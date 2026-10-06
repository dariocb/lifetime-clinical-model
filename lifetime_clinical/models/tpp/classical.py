"""Small Poisson/exponential-Hawkes baselines with exact censored likelihoods.

Parameters are patient-specific constants. MLPs use baseline covariates only;
Hawkes excitation uses the observed event history directly, without an encoder.
All calculations use years since study entry.
"""
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import minimize
from ...data.synthetic import YEAR_DAYS
from ...components.history import NeuralInputs,clinical_feature_names


def _tensors(seq, device='cpu'):
    return (torch.as_tensor(seq.times/YEAR_DAYS, dtype=torch.float64, device=device),
            torch.as_tensor(seq.marks, dtype=torch.long, device=device), seq.end/YEAR_DAYS)


def process_nll(seq, mu, alpha=None, delta=None, start=0., end=None, timing_only=False):
    """Exact conditional NLL over (start,end], retaining all pre-start excitation.

    start/end are YEARS since entry. timing_only sums mark intensities before
    taking logs; the integral always sums them. Includes the terminal no-event
    interval. Initial history at study entry is empty, not oracle pre-entry data.
    """
    mu = torch.as_tensor(mu, dtype=torch.float64).reshape(-1)
    times, marks, horizon = _tensors(seq, mu.device)
    end = horizon if end is None else float(end)
    if not 0 <= start < end <= horizon+1e-10:
        raise ValueError('Require 0 <= start < end <= observation end')
    selected = (times > start) & (times <= end)
    target = times[selected]
    if len(mu) == 1:
        target_marks = torch.zeros(len(target), dtype=torch.long, device=mu.device)
        source_marks = torch.zeros_like(marks)
    else:
        target_marks, source_marks = marks[selected], marks
    intensity = mu[None, :].expand(len(target), -1)
    integral = mu.sum()*(end-start)
    if alpha is not None:
        alpha = torch.as_tensor(alpha, dtype=torch.float64, device=mu.device).reshape(len(mu), len(mu))
        delta = torch.as_tensor(delta, dtype=torch.float64, device=mu.device)
        lags = target[:, None]-times[None, :]
        kernel = torch.exp(-delta*lags.clamp_min(0))*(lags > 0)
        intensity = intensity + kernel@alpha[:, source_marks].T
        eligible = times < end
        source = times[eligible]
        lower = (start-source).clamp_min(0)
        upper = end-source
        mass = (torch.exp(-delta*lower)-torch.exp(-delta*upper))/delta
        integral = integral+(alpha[:, source_marks[eligible]].sum(0)*mass).sum()
    event_intensity = intensity.sum(1) if timing_only else intensity[
        torch.arange(len(target), device=mu.device), target_marks]
    return integral-torch.log(event_intensity.clamp_min(1e-12)).sum()


def fit_patient_poisson(seq, cutoff):
    """MLE with a 1e-6/year numerical rate floor for zero-event prefixes."""
    count = int(np.sum(seq.times/YEAR_DAYS <= cutoff))
    return np.array([max(1e-6, count/cutoff)]), None, None


def fit_patient_hawkes(seq, cutoff):
    """Constrained MLE: mu,delta>0 and alpha/delta<0.95."""
    rate = max(.02, np.sum(seq.times/YEAR_DAYS <= cutoff)/cutoff)
    def unpack(raw):
        mu = np.exp(raw[0]); delta = np.exp(raw[1])
        rho = .95/(1+np.exp(-raw[2]))
        return np.array([mu]), np.array([[rho*delta]]), delta
    def objective(raw):
        return float(process_nll(seq, *unpack(raw), end=cutoff))
    candidates = [minimize(objective, [np.log(rate), np.log(d), -2.],
                           method='L-BFGS-B', bounds=[(-14,5),(-5,5),(-12,8)],
                           options={'maxiter':150}) for d in (.5, 3.)]
    result = min(candidates, key=lambda r:r.fun)
    return unpack(result.x)


class CovariatePointProcess(nn.Module):
    """Shared small MLP -> each patient's constant process parameters."""
    def __init__(self, static_dim, kind='poisson', hidden_dim=12, seed=42):
        super().__init__()
        if kind not in ('poisson', 'hawkes', 'marked_hawkes'):
            raise ValueError('Unknown baseline kind')
        torch.manual_seed(seed)
        self.kind = kind
        self.n_marks = 3 if kind == 'marked_hawkes' else 1
        size = 1 if kind == 'poisson' else (3 if kind == 'hawkes' else 16)
        self.input_encoder=NeuralInputs(static_dim,hidden_dim,clinical_feature_names("static",static_dim))
        self.net = nn.Sequential(nn.Linear(self.input_encoder.output_dim,hidden_dim),nn.Tanh(),nn.Linear(hidden_dim,size)).double()
        self.register_buffer('x_mean',torch.zeros(static_dim,dtype=torch.float64))
        self.register_buffer('x_scale',torch.ones(static_dim,dtype=torch.float64))
        self.fit_history = []

    def parameters_for(self, seq):
        x = torch.as_tensor(seq.static,dtype=torch.float64,device=self.x_mean.device)
        raw = self.net(self.input_encoder(x))
        if self.kind == 'poisson': return F.softplus(raw[:1])+1e-6,None,None
        if self.kind == 'hawkes':
            mu,delta = F.softplus(raw[0:1])+1e-6,F.softplus(raw[1])+1e-4
            alpha = delta*.95*torch.sigmoid(raw[2])
            return mu,alpha.reshape(1,1),delta
        mu = F.softplus(raw[:3])+1e-6
        delta = F.softplus(raw[3])+1e-4
        rho = .95*torch.sigmoid(raw[4:7])
        weights = torch.softmax(raw[7:16].reshape(3,3),dim=0)
        # Each source column has branching mass <.95, hence spectral radius <1.
        alpha = delta*weights*rho[None,:]
        return mu,alpha,delta

    def score(self, sequences, cutoffs=None, timing_only=False):
        with torch.no_grad():
            losses = [float(process_nll(s,*self.parameters_for(s),
                         start=0. if cutoffs is None else cutoffs[s.patient_id],
                         timing_only=timing_only)) for s in sequences]
        return float(np.mean(losses))

    def fit(self, train, validation, epochs=5, device='cpu', lr=.02):
        self.to(device)
        x = torch.as_tensor(np.stack([s.static for s in train]),dtype=torch.float64,device=device)
        self.x_mean.copy_(x.mean(0)); self.x_scale.copy_(x.std(0,unbiased=False).clamp_min(.1))
        optimizer = torch.optim.Adam(self.parameters(),lr=lr)
        rng = np.random.default_rng(42)
        best,best_state = np.inf,None
        for epoch in range(epochs+1):
            if epoch:
                self.train()
                for indices in np.array_split(rng.permutation(len(train)),max(1,int(np.ceil(len(train)/4)))):
                    optimizer.zero_grad()
                    loss = torch.stack([process_nll(train[j],*self.parameters_for(train[j])) for j in indices]).mean()
                    loss.backward(); nn.utils.clip_grad_norm_(self.parameters(),5.); optimizer.step()
            self.eval()
            val = self.score(validation)
            self.fit_history.append({'epoch':epoch,'train_nll':self.score(train),'validation_nll':val})
            if val < best: best,best_state = val,deepcopy(self.state_dict())
        self.load_state_dict(best_state)
        return self


def baseline_comparison(test, poisson_params, hawkes_params, models, mark_probabilities, fraction=.6):
    """Held-out suffixes; per-patient fits use prefixes, MLPs use training patients."""
    import pandas as pd
    cutoffs = {s.patient_id:fraction*s.end/YEAR_DAYS for s in test}
    rows = []
    mark_cost = float(np.mean([-np.log(mark_probabilities[s.marks[s.times/YEAR_DAYS > cutoffs[s.patient_id]]]).sum() for s in test]))
    for name,parameters in [('Poisson / patient',poisson_params),('Hawkes / patient',hawkes_params)]:
        values = [float(process_nll(s,*parameters[s.patient_id],start=cutoffs[s.patient_id])) for s in test]
        rows.append({'Model':name,'Timing NLL (nats/patient)':np.mean(values),
                     'Timing + marks NLL (nats/patient)':np.mean(values)+mark_cost})
    for name,model in models.items():
        rows.append({'Model':name,'Timing NLL (nats/patient)':model.score(test,cutoffs,timing_only=True),
                     'Timing + marks NLL (nats/patient)':model.score(test,cutoffs) if model.n_marks>1 else model.score(test,cutoffs)+mark_cost})
    return pd.DataFrame(rows).set_index('Model')


def plot_baseline_intensities(seq, fitted, cutoff, ax):
    grid = np.linspace(0,seq.end/YEAR_DAYS,900)
    times = seq.times/YEAR_DAYS
    for name,parameters in fitted.items():
        mu,alpha,delta = [None if p is None else np.asarray(
            p.detach().cpu() if torch.is_tensor(p) else p) for p in parameters]
        mu = mu.reshape(-1)
        intensity = np.full(len(grid),mu.sum())
        if alpha is not None:
            source = seq.marks if len(mu)>1 else np.zeros_like(seq.marks)
            lags = grid[:,None]-times[None,:]
            intensity += (np.exp(-float(delta)*np.maximum(lags,0))*(lags>0))@alpha.reshape(len(mu),len(mu))[:,source].sum(0)
        ax.plot(grid,intensity,label=name,linewidth=1.3)
    ax.scatter(times,np.zeros_like(times),marker='x',color='red',s=25,label='Observed visits',zorder=4)
    ax.axvline(cutoff,color='gray',linestyle='--',label='Prefix / held-out cutoff')
    ax.set(xlabel='Years since entry',ylabel=r'Conditional intensity ($\lambda$; events/year)',
           title=f'{seq.patient_id}: baseline conditional intensities')
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.13),ncol=3,frameon=False,fontsize=8)
