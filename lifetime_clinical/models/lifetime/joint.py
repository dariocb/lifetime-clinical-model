"""Joint visit-flow / residual-GP prototype with a shared causal Transformer.

Measurement times are conditioned upon; this is not a measurement-occurrence
model. The GP prior/residual context is fixed at the prefix origin, while the
visit context refreshes after visits. Each recorded value is scored once.
"""
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from ..tpp.conditional_flow import ConditionalFlowTPP
from ..longitudinal.residual_flow_gp import ResidualFlowGP
from ...components.history import CausalHistoryEncoder,clinical_feature_names
from ...data.longitudinal import GPBlock
from ...data.preprocessing import VARIABLES, prepare_continuous_targets
from ...data.synthetic import EVENT_TYPES, YEAR_DAYS


@dataclass
class LifetimePatient:
    block: GPBlock
    tokens: np.ndarray
    token_days: np.ndarray
    times: np.ndarray
    marks: np.ndarray
    end: float
    origin_age: float

    @property
    def patient_id(self): return self.block.patient_id


def prepare_lifetime_data(dataset, preprocessor, splits, history_fraction=.25):
    """Index each table once, retaining strictly causal record tokens."""
    targets = prepare_continuous_targets(dataset, preprocessor)
    groups = {"measure": dict(tuple(targets.groupby("patient_id"))),
              "event": dict(tuple(dataset.events.groupby("patient_id"))),
              "update": dict(tuple(dataset.dynamic_covariates.groupby("patient_id")))}
    patients = dataset.patients.set_index("patient_id", drop=False)
    offsets = np.array([preprocessor.measurement_scales[v][0] for v in VARIABLES])
    scales = np.array([preprocessor.measurement_scales[v][1] for v in VARIABLES])
    result = {}
    for split, ids in splits.items():
        records_out = []
        for pid in ids:
            patient = patients.loc[pid]
            measures = groups["measure"].get(pid, targets.iloc[:0]).sort_values("measurement_date")
            known = measures[measures.target_mask]
            panels = known.measurement_date.drop_duplicates().tolist()
            if len(panels) < 2: continue
            count = max(1, min(int(len(panels)*history_fraction), len(panels)-1))
            origin = pd.Timestamp(panels[count-1])
            if origin >= patient.study_end_date: continue
            rows = []
            events = groups["event"].get(pid, dataset.events.iloc[:0]).sort_values("event_date")
            for e in events.itertuples():
                rows.append((e.event_date, preprocessor.token("event",e.event_date,patient,
                            event_type=e.event_type,iat=e.iat_days)))
            for m in measures.itertuples():
                rows.append((m.measurement_date,preprocessor.token("measurement",m.measurement_date,
                            patient,variable=m.variable,value=m.value if m.target_mask else None)))
            updates = groups["update"].get(pid, dataset.dynamic_covariates.iloc[:0])
            for u in updates[updates.source!="baseline"].itertuples():
                rows.append((u.recorded_at,preprocessor.token("covariate",u.recorded_at,patient,
                            variable=u.variable,value=u.value)))
            rows.sort(key=lambda row:row[0])
            tokens = np.asarray([row[1] for row in rows],dtype=float)
            days = np.asarray([(pd.Timestamp(row[0])-origin).total_seconds()/86400 for row in rows])
            after = known[known.measurement_date > origin]
            if after.empty: continue
            block = GPBlock(pid,preprocessor.static_row(patient).astype(float),tokens[days<=0].copy(),
                            after.query_age_scaled.to_numpy(float),after.variable_id.to_numpy(int),
                            after.target_z.to_numpy(float),offsets.copy(),scales.copy(),origin).validate()
            visits = events[events.event_date>origin]
            times = (visits.event_date-origin).dt.total_seconds().to_numpy()/86400
            if len(times) and np.any(np.diff(times)<=0):
                raise ValueError("Visit times must be distinct for this prototype")
            marks = visits.event_type.map({v:i for i,v in enumerate(EVENT_TYPES)}).to_numpy(int)
            age = (origin-patient.date_of_birth).total_seconds()/(86400*YEAR_DAYS)
            records_out.append(LifetimePatient(block,tokens,days,times,marks,
                               (patient.study_end_date-origin).total_seconds()/86400,age))
        result[split] = records_out
    return result


SharedCausalEncoder = CausalHistoryEncoder


class SharedPhysiologyGP(ResidualFlowGP):
    def build_gp_inputs(self, block, ages=None, variables=None):
        self._check([block],allow_empty=True)
        ages=block.ages if ages is None else np.atleast_1d(np.asarray(ages,float))
        variables=block.variables if variables is None else np.atleast_1d(np.asarray(variables,int))
        if ages.shape!=variables.shape or not np.isfinite(ages).all():
            raise ValueError("Matching finite query ages and variable IDs required")
        cached=getattr(self,"_cached_prefix",None)
        state=cached[1] if cached is not None and cached[0]==block.patient_id else self.history_encoder.states(block.history)[-1]
        baseline=torch.cat([self._tensor(block.static),state])
        identity=F.one_hot(torch.as_tensor(variables,dtype=torch.long,device=state.device),self.n_variables).to(state.dtype)
        return torch.cat([baseline.expand(len(ages),-1),self._tensor(ages)[:,None],identity],dim=1)


class VisitFlow(ConditionalFlowTPP):
    """TPP-A timing flows with the shared history state supplied externally."""
    def build_context(self, inputs, history=()):
        state=self._tensor(history)
        if state.shape!=(self.hidden_dim,): raise ValueError("Supply a shared history state")
        return torch.cat([self._tensor(inputs),state])

    def batch_marked_log_prob(self, gaps, contexts, marks):
        raw=self.net(contexts)
        size=4*self.flow_layers*self.n_marks
        pars=raw[:,:size].reshape(-1,self.flow_layers,self.n_marks,4)
        value=torch.log(self._tensor(gaps)/self.time_scale)[:,None]
        ld=torch.zeros_like(value)
        for layer in range(self.flow_layers):
            shift=pars[:,layer,:,0]; scale=torch.exp(.8*torch.tanh(pars[:,layer,:,1]))
            skew=.8*torch.tanh(pars[:,layer,:,2]); tail=torch.exp(.3*torch.tanh(pars[:,layer,:,3]))
            a=(value-shift)/scale; b=tail*(torch.asinh(a)+skew)
            logcosh=torch.abs(b)+F.softplus(-2*torch.abs(b))-np.log(2.)
            ld=ld+torch.log(tail)-torch.log(scale)+logcosh-.5*torch.log1p(a*a)
            value=torch.sinh(b)
        pdf=-.5*value.square()-.5*np.log(2*np.pi)+ld-torch.log(self._tensor(gaps))[:,None]
        joint=pdf+torch.log_softmax(raw[:,size:],dim=-1)
        return joint.gather(1,torch.as_tensor(marks,dtype=torch.long,device=joint.device)[:,None]).squeeze(1)


class LifetimeModel(nn.Module):
    """Joint ELBO; shared gradients from visits and physiological observations."""
    def __init__(self, static_dim, token_dim, hidden_dim=12, seed=42):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.shared=SharedCausalEncoder(token_dim,hidden_dim,
                        feature_names=clinical_feature_names("history",token_dim))
        self.gp=SharedPhysiologyGP(input_dim=static_dim,history_dim=token_dim,hidden_dim=hidden_dim,
                                  kernel_rank=6,flow_layers=2,inducing_per_variable=3,elbo_samples=8,seed=seed)
        self.gp.history_encoder=self.shared
        self.tpp=VisitFlow(input_dim=static_dim,hidden_dim=hidden_dim,n_marks=3,seed=seed,
                           flow_layers=2,transformer_layers=1)
        # Only the shared Transformer supplies history; remove the unused private one.
        del self.tpp.encoder
        self.fit_history=[]

    def patient_terms(self, patient):
        states=self.shared.states(patient.tokens)
        origin_count=np.searchsorted(patient.token_days,0.,side="right")
        self.gp._cached_prefix=(patient.patient_id,states[origin_count])
        try:
            physiology=-self.gp.elbo([patient.block])[0]
        finally:
            self.gp._cached_prefix=None
        origins=np.r_[0.,patient.times]
        indices=np.searchsorted(patient.token_days,origins,side="right")
        contexts=torch.cat([self.tpp._tensor(patient.block.static).expand(len(indices),-1),states[indices]],dim=1)
        event_ll=self.tpp._tensor(0.)
        if len(patient.times):
            gaps=np.diff(origins)
            event_ll=self.tpp.batch_marked_log_prob(gaps,contexts[:-1],patient.marks).sum()
        _,sf,_=self.tpp._log_terms(patient.end-origins[-1],contexts[-1])
        return -event_ll-sf,physiology

    def objective(self, patients):
        terms=torch.stack([torch.stack(self.patient_terms(patient)) for patient in patients])
        return terms.mean(0)

    def score(self, patients):
        self.eval()
        with torch.no_grad(): event,gp=self.objective(patients).cpu().numpy()
        return dict(event_nll=float(event),gp_negative_elbo=float(gp),joint_negative_elbo=float(event+gp))

    def fit(self, train, validation, epochs=5, device="cpu", seed=42):
        if not train or not validation or epochs<1: raise ValueError("Nonempty splits and positive epochs required")
        if {p.patient_id for p in train}&{p.patient_id for p in validation}: raise ValueError("Patient leakage")
        self.to(device)
        optimizer=torch.optim.Adam(self.parameters(),lr=.004)
        rng=np.random.default_rng(seed); best=np.inf; best_state=None
        for epoch in range(epochs+1):
            if epoch:
                self.train()
                order=rng.permutation(len(train))
                for start in range(0,len(order),3):
                    batch=[train[i] for i in order[start:start+3]]
                    optimizer.zero_grad(); loss=self.objective(batch).sum()
                    if not torch.isfinite(loss): raise FloatingPointError("Nonfinite joint ELBO")
                    loss.backward(); nn.utils.clip_grad_norm_(self.parameters(),10.,error_if_nonfinite=True)
                    optimizer.step()
            training=self.score(train); monitoring=self.score(validation)
            self.fit_history.append(dict(epoch=epoch,train=training,validation=monitoring))
            print(f"Epoch {epoch}/{epochs}: train negative ELBO {training['joint_negative_elbo']:.3f}; validation {monitoring['joint_negative_elbo']:.3f}",flush=True)
            if monitoring['joint_negative_elbo']<best:
                best=monitoring['joint_negative_elbo']; best_state=deepcopy(self.state_dict())
        self.load_state_dict(best_state); self.eval()
        return self.fit_history

    def predict_events(self, patient, horizons, cutoff_days):
        if not 0<=cutoff_days<=patient.end: raise ValueError("Cutoff outside observation window")
        observed=patient.times[patient.times<=cutoff_days]
        last=float(observed[-1]) if len(observed) else 0.
        with torch.no_grad():
            tokens=patient.tokens[patient.token_days<=last]
            state=self.shared.states(tokens)[-1]
            return self.tpp.next_event_distribution(patient.block.static,horizons,history=state,
                                                    elapsed_days=cutoff_days-last)
