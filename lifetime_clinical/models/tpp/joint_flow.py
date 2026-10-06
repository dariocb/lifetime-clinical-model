"""Model B: continuous feature coupling flow and conditional timing flow.

feature_indices explicitly names real-valued density coordinates. Remaining inputs
are fixed conditioning metadata, not continuous density targets.
"""
import numpy as np
import torch
from torch import nn
from .conditional_flow import ConditionalFlowTPP
from ...components.history import CausalHistoryEncoder,NeuralInputs,clinical_feature_names
from ...data.synthetic import EVENT_TYPES

class FeatureCoupling(nn.Module):
    def __init__(self,dimension,hidden_dim,layers=4,context_dim=0):
        super().__init__();self.dimension=dimension;self.context_dim=context_dim
        self.nets=nn.ModuleList();self.masks=[]
        for j in range(layers):
            mask=((torch.arange(dimension)+j)%2).double()
            if dimension==1:mask=torch.zeros(1,dtype=torch.float64)
            self.register_buffer("mask_"+str(j),mask)
            net=nn.Sequential(nn.Linear(dimension+context_dim,hidden_dim),nn.Tanh(),nn.Linear(hidden_dim,2*dimension)).double()
            nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias);self.nets.append(net)
    def forward(self,x,inverse=False,context=None):
        ld=torch.zeros(x.shape[:-1],dtype=x.dtype,device=x.device)
        indices=range(len(self.nets)-1,-1,-1) if inverse else range(len(self.nets))
        for j in indices:
            mask=getattr(self,"mask_"+str(j));fixed=x*mask
            if self.context_dim:
                if context is None:raise ValueError("Conditional feature flow needs temporal context")
                fixed_input=torch.cat([fixed,context.expand(*x.shape[:-1],self.context_dim)],dim=-1)
            else:fixed_input=fixed
            scale,shift=self.nets[j](fixed_input).chunk(2,dim=-1)
            scale=.8*torch.tanh(scale)*(1-mask);shift=shift*(1-mask)
            x=fixed+(1-mask)*((x-shift)*torch.exp(-scale) if inverse else x*torch.exp(scale)+shift)
            ld=ld+(-scale if inverse else scale).sum(-1)
        return x,ld

class JointFlowTPP(ConditionalFlowTPP):
    model_name="tpp_b"
    history_encoder="transformer"
    def __init__(self,feature_indices=None,feature_layers=4,input_mode="static",**kwargs):
        if feature_indices is None or not len(feature_indices):
            raise ValueError("Explicit continuous feature_indices required; one-hot/masks are not density coordinates")
        if len(set(feature_indices))!=len(feature_indices) or any(int(i)!=i or i<0 for i in feature_indices):
            raise ValueError("Invalid feature_indices")
        if feature_layers<1:raise ValueError("feature_layers must be positive")
        self.feature_indices=tuple(int(i) for i in feature_indices);self.feature_layers=int(feature_layers)
        if input_mode not in ("static","dynamic"):raise ValueError("input_mode must be static or dynamic")
        self.input_mode=input_mode
        self.input_field="contexts" if input_mode=="dynamic" else "static"
        super().__init__(**kwargs)
    def _initialize(self,input_dim):
        if max(self.feature_indices)>=input_dim:raise ValueError("Feature index outside input schema")
        super()._initialize(input_dim)
        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            self.feature_flow=FeatureCoupling(len(self.feature_indices),self.hidden_dim,self.feature_layers,
                                             self.hidden_dim if self.input_mode=="dynamic" else 0)
            if self.input_mode=="dynamic":
                del self.encoder  # Dynamic timing uses feature-aware history instead.
                self.feature_encoder=CausalHistoryEncoder(len(self.feature_indices),self.hidden_dim,
                                                           self.transformer_heads,self.transformer_layers)
                timing_dim=input_dim if getattr(self,"use_private",False) else input_dim-len(self.feature_indices)+getattr(self,"shared_dim",len(self.feature_indices))
                self.dynamic_encoder=CausalHistoryEncoder(timing_dim+1,self.hidden_dim,
                    self.transformer_heads,self.transformer_layers,
                    feature_names=None if self._timing_names() is None else self._timing_names()+["log_iat"])
                names=self._timing_names()
                encoded=NeuralInputs(timing_dim+self.hidden_dim,self.hidden_dim,
                     None if names is None else names+[f"H_{i}" for i in range(self.hidden_dim)])
                self.net=nn.Sequential(encoded,nn.Linear(encoded.output_dim,self.hidden_dim),nn.Tanh(),
                                       nn.Linear(self.hidden_dim,self.output_dim)).double()
                nn.init.normal_(self.net[-1].weight,std=.01);nn.init.zeros_(self.net[-1].bias)

    def _timing_names(self):
        names=clinical_feature_names("dynamic",self.input_dim)
        if names is None:return None
        if hasattr(self,"shared_dim"):
            count=len(self.feature_indices) if self.use_private else self.shared_dim
            return [f"feature_latent_{i}" for i in range(count)]+[n for i,n in enumerate(names) if i not in self.feature_indices]
        return names

    def _dynamic_state(self,history=None):
        history={} if history is None else history
        if not isinstance(history,dict):
            if len(history):raise ValueError("Dynamic history must be a dict with features and iats")
            history={}
        features=np.asarray(history.get("features",np.empty((0,self.input_dim))),dtype=float)
        gaps=np.asarray(history.get("iats",np.zeros(len(features))),dtype=float)
        if features.shape!=(len(gaps),self.input_dim) or not np.isfinite(features).all() or not np.isfinite(gaps).all() or (gaps<0).any():
            raise ValueError("Dynamic history needs aligned finite feature rows and nonnegative elapsed gaps")
        # Feature-history states use earlier observed physiology, not future latents.
        x=self._tensor(features)
        names=clinical_feature_names("dynamic",self.input_dim)
        ages=x[:,names.index("current_age_scaled")]*100 if names is not None else None
        prefix=self.feature_encoder.states(x[:,list(self.feature_indices)],positions=ages)
        if len(features):
            z,_=self.feature_flow(x[:,list(self.feature_indices)],context=prefix[:-1])
            tokens=torch.cat([self._latent_timing(x,z),torch.log1p(self._tensor(gaps)/self.time_scale)[:,None]],dim=-1)
        else:
            tokens=self._tensor(np.empty((0,self.dynamic_encoder.input_dim)))
        return prefix[-1],self.dynamic_encoder(tokens)

    def _latent_timing(self,x,z):
        result=x.clone();result[...,list(self.feature_indices)]=z
        return result

    def build_context(self,inputs,history=()):
        if self.input_mode=="static":return super().build_context(inputs,history)
        x=self._tensor(inputs)
        if x.shape!=(self.input_dim,) or not torch.isfinite(x).all():raise ValueError("Invalid dynamic input vector")
        fstate,tstate=self._dynamic_state(history)
        z,_=self.feature_flow(x[list(self.feature_indices)],context=fstate)
        return torch.cat([self._latent_timing(x,z),tstate])
    def encode_features(self,inputs,history=None):
        x=self._tensor(inputs)
        if x.shape[-1]!=self.input_dim or not torch.isfinite(x).all():raise ValueError("Invalid features")
        context=self._dynamic_state(history)[0] if self.input_mode=="dynamic" else None
        return self.feature_flow(x[...,list(self.feature_indices)],context=context)[0]
    def _timing_features(self,inputs):
        x=self._tensor(inputs);z=self.encode_features(x);result=x.clone()
        result[...,list(self.feature_indices)]=z
        return result
    def feature_log_prob(self,inputs,history=None):
        x=self._tensor(inputs)
        if x.shape[-1]!=self.input_dim or not torch.isfinite(x).all():raise ValueError("Invalid features")
        context=self._dynamic_state(history)[0] if self.input_mode=="dynamic" else None
        z,ld=self.feature_flow(x[...,list(self.feature_indices)],context=context)
        return -.5*(z*z+np.log(2*np.pi)).sum(-1)+ld
    def _sequence_log_likelihood(self,seq):
        if self.input_mode=="dynamic":return self._dynamic_log_likelihood(seq,True)
        return super()._sequence_log_likelihood(seq)+self.feature_log_prob(seq.static)
    def _dynamic_log_likelihood(self,seq,include_features):
        gaps=np.diff(np.r_[0.,seq.times]);ll=self._tensor(0.)
        for j,row in enumerate(seq.contexts):
            historical_gaps=gaps[:j].copy()
            if self.entry_policy=="condition_on_first" and j:historical_gaps[0]=0.
            past=dict(features=seq.contexts[:j],iats=historical_gaps)
            if include_features:ll=ll+self.feature_log_prob(row,past)
            if j<len(gaps):
                if j==0 and self.entry_policy=="condition_on_first":continue
                ll=ll+self._marked_log_prob(gaps[j],self.build_context(row,past),
                                             int(seq.marks[j]) if self.n_marks>1 else 0)
            elif len(gaps) or self.entry_policy=="reset_at_entry":
                _,sf,_=self._log_terms(seq.end-(seq.times[-1] if len(gaps) else 0.),self.build_context(row,past))
                ll=ll+sf
        return ll
    def conditional_nll(self,sequences):
        if self.input_mode=="static":return super().conditional_nll(sequences)
        self._check(sequences)
        return -torch.stack([self._dynamic_log_likelihood(s,False) for s in sequences]).mean()
    def joint_nll(self,sequences):return self.nll(sequences)
    def sample_patients(self,n_samples=1,conditioning_inputs=None,seed=None,history=None):
        if n_samples<1:raise ValueError("n_samples must be positive")
        remainder=set(range(self.input_dim))-set(self.feature_indices)
        if remainder and conditioning_inputs is None:
            raise ValueError("Supply conditioning_inputs for coordinates without a density model")
        template=np.zeros(self.input_dim) if conditioning_inputs is None else np.asarray(conditioning_inputs,dtype=float)
        if template.shape!=(self.input_dim,) or not np.isfinite(template).all():raise ValueError("Invalid conditioning template")
        rng=np.random.default_rng(seed)
        with torch.no_grad():
            x=np.tile(template,(n_samples,1));z=self._tensor(rng.normal(size=(n_samples,len(self.feature_indices))))
            context=self._dynamic_state(history)[0] if self.input_mode=="dynamic" else None
            x[:,list(self.feature_indices)]=self.feature_flow(z,inverse=True,context=context)[0].cpu().numpy()
        return x

    def infer_features(self,inputs,missing_indices,history=(),next_iat=None,censor_days=None,
                       n_samples=500,burn_in=300,proposal_scale=.25,seed=42):
        """Random-walk MH posterior for missing continuous coordinates.

        Targets p(x_missing | x_observed, past, optional next-IAT evidence).
        Metadata remains fixed. No claim of closed-form or exact Monte Carlo inference.
        """
        from ...inference.features import infer_features
        return infer_features(self,inputs,missing_indices,history,next_iat,censor_days,
                              n_samples,burn_in,proposal_scale,seed)

    def simulate_timeline(self,inputs,end_days,history=(),current_time=0.,elapsed_days=0.,seed=None,max_events=10000):
        if self.input_mode=="static":return super().simulate_timeline(inputs,end_days,history,current_time,elapsed_days,seed,max_events)
        if not np.isfinite([end_days,current_time,elapsed_days]).all() or not 0<=current_time<=end_days or elapsed_days<0:
            raise ValueError("Invalid simulation window")
        self._dynamic_state(history)
        past={} if not isinstance(history,dict) else history
        rows=list(np.asarray(past.get("features",np.empty((0,self.input_dim)))))
        gaps=list(past.get("iats",np.zeros(len(rows))))
        rng=np.random.default_rng(seed);x=np.asarray(inputs,dtype=float).copy();times=[];marks=[];snapshots=[x.copy()]
        time=current_time;elapsed=elapsed_days
        for _ in range(max_events):
            state=dict(features=np.asarray(rows).reshape(-1,self.input_dim),iats=np.asarray(gaps))
            waits,event_marks=self.sample_next_event(x,state,1,elapsed,seed=int(rng.integers(2**31)))
            wait=float(waits[0])
            if not np.isfinite(wait) or wait<=0:raise FloatingPointError("Invalid sampled waiting time")
            time+=wait
            if time>end_days:return dict(times=np.asarray(times),marks=np.asarray(marks,dtype=int),features=np.asarray(snapshots))
            times.append(time);marks.append(int(event_marks[0]));rows.append(x.copy());gaps.append(elapsed+wait)
            state=dict(features=np.asarray(rows),iats=np.asarray(gaps))
            x=self.sample_patients(1,conditioning_inputs=x,history=state,seed=int(rng.integers(2**31)))[0]
            # Advance the absolute clock of the generated snapshot. Future
            # history positions must refer to its generated age, not the cutoff.
            names=clinical_feature_names("dynamic",self.input_dim)
            if names is not None:
                x[names.index("current_age_scaled")]+=wait/self.time_scale/100.
                i=names.index("log1p_days_since_entry")
                x[i]=np.log1p(np.expm1(x[i])+wait)
                i=names.index("log1p_event_count")
                x[i]=np.log1p(np.expm1(x[i])+1)
                x[names.index("log1p_last_iat")]=np.log1p(elapsed+wait)
                x[names.index("last_iat_known")]=1.
                for k,name in enumerate(EVENT_TYPES):
                    x[names.index("last_event="+name)]=float(k==event_marks[0])
            snapshots.append(x.copy());elapsed=0.
        raise RuntimeError("Simulation exceeded max_events")
