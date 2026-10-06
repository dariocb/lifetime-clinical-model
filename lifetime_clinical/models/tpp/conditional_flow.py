"""Model A: mark-conditioned positive-IAT flows with causal Transformer history."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import ndtri_exp
from .neural_base import NeuralTPP
from ...components.history import CausalHistoryEncoder,NeuralInputs,clinical_feature_names

class ConditionalFlowTPP(NeuralTPP):
    model_name = "tpp_a"
    history_encoder = "transformer"
    def __init__(self,input_dim=None,hidden_dim=32,n_marks=1,seed=42,time_scale=365.2425,
                 entry_policy="reset_at_entry",flow_layers=2,transformer_heads=2,
                 transformer_layers=1):
        if entry_policy not in ("reset_at_entry","condition_on_first"):
            raise ValueError("Unknown entry policy")
        if flow_layers < 1: raise ValueError("flow_layers must be positive")
        if (int(transformer_heads)!=transformer_heads or transformer_heads<1
                or int(transformer_layers)!=transformer_layers or transformer_layers<1):
            raise ValueError("Positive integer Transformer heads/layers required")
        if self.history_encoder == "transformer" and hidden_dim % transformer_heads:
            raise ValueError("hidden_dim must be divisible by transformer_heads")
        self.transformer_heads,self.transformer_layers=int(transformer_heads),int(transformer_layers)
        self.entry_policy,self.flow_layers=entry_policy,int(flow_layers)
        self.output_dim=4*self.flow_layers*n_marks+n_marks
        super().__init__(input_dim,hidden_dim,n_marks,seed,time_scale)

    def _initialize(self,input_dim):
        super()._initialize(input_dim)
        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            self.encoder=CausalHistoryEncoder(2,self.hidden_dim,self.transformer_heads,self.transformer_layers)
            names=clinical_feature_names("static",input_dim)
            context_names=None if names is None else names+[f"H_{i}" for i in range(self.hidden_dim)]
            encoded=NeuralInputs(input_dim+self.hidden_dim,self.hidden_dim,context_names)
            self.net=nn.Sequential(encoded,nn.Linear(encoded.output_dim,self.hidden_dim),nn.Tanh(),
                                   nn.Linear(self.hidden_dim,self.output_dim)).double()
            nn.init.normal_(self.net[-1].weight,std=.01);nn.init.zeros_(self.net[-1].bias)

    def encode_history(self,history,start_age=0.):
        history=np.asarray(history,dtype=float)
        if history.ndim != 1 or not np.isfinite(history).all() or (history<=0).any():
            raise ValueError("History must contain positive finite IATs in days")
        return self.encode_history_states(history,start_age)[-1]

    def encode_history_states(self,history,start_age=0.):
        """One causal state per prefix, including an explicit empty-history token.

        Tokens contain log-scaled IAT and elapsed time since the renewal origin.
        Learned absolute-age positions preserve the irregular clinical clock.
        A triangular mask prevents any
        prefix state from attending to a later IAT. Dropout is zero so density
        evaluation and its inverse use the same deterministic context.
        """
        values=np.asarray(history,dtype=float)
        if values.ndim!=1 or not np.isfinite(values).all() or (values<=0).any():
            raise ValueError("History must contain positive finite IATs in days")
        durations=self._tensor(values)
        features=torch.stack([torch.log1p(durations/self.time_scale),
                              torch.log1p(torch.cumsum(durations,0)/self.time_scale)],dim=-1)
        ages=float(start_age)+torch.cumsum(durations,0)/self.time_scale
        return self.encoder.states(features,positions=ages)

    def _timing_features(self,inputs): return self._tensor(inputs)

    def build_context(self,inputs,history=()):
        x=self._tensor(inputs)
        if x.shape != (self.input_dim,) or not torch.isfinite(x).all():
            raise ValueError("Expected one finite patient feature vector")
        start_age=float(x[0].detach())*100 if clinical_feature_names("static",self.input_dim) is not None else 0.
        return torch.cat([self._timing_features(x),self.encode_history(history,start_age)])

    def _params(self,context):
        raw=self.net(context)
        flow_size=4*self.flow_layers*self.n_marks
        pars=raw[:flow_size].reshape(self.flow_layers,self.n_marks,4)
        # Identity at zero: affine log-IAT followed by sinh-arcsinh warping.
        shift=pars[:,:,0];scale=torch.exp(.8*torch.tanh(pars[:,:,1]))
        skew=.8*torch.tanh(pars[:,:,2]);tail=torch.exp(.3*torch.tanh(pars[:,:,3]))
        return shift,scale,skew,tail,torch.log_softmax(raw[flow_size:],dim=-1)

    def _transform(self,logt,context,inverse=False):
        shift,scale,skew,tail,_=self._params(context)
        value=logt;logdet=torch.zeros_like(value)
        indices=range(self.flow_layers-1,-1,-1) if inverse else range(self.flow_layers)
        for j in indices:
            if inverse:
                value=scale[j]*torch.sinh(torch.asinh(value)/tail[j]-skew[j])+shift[j]
            else:
                a=(value-shift[j])/scale[j]
                b=tail[j]*(torch.asinh(a)+skew[j])
                # Stable log(cosh(b)).
                logcosh=torch.abs(b)+F.softplus(-2*torch.abs(b))-np.log(2.)
                logdet=logdet+torch.log(tail[j])-torch.log(scale[j])+logcosh-.5*torch.log1p(a*a)
                value=torch.sinh(b)
        return value,logdet

    def _component_log_terms(self,days,context):
        """Per-mark log density/survival, with mark on the final axis."""
        t=self._tensor(days)[...,None]
        safe=t.clamp_min(torch.finfo(t.dtype).tiny)
        z,ld=self._transform(torch.log(safe/self.time_scale),context)
        pdf=-.5*z*z-.5*np.log(2*np.pi)+ld-torch.log(safe)
        sf=torch.special.log_ndtr(-z)
        pdf=torch.where(t>0,pdf,torch.full_like(pdf,-torch.inf))
        sf=torch.where(t>0,sf,torch.zeros_like(sf))
        return pdf,sf,self._params(context)[-1]

    def _log_terms(self,days,context):
        """Marginal IAT terms; unknown marks are integrated out."""
        pdf,sf,marks=self._component_log_terms(days,context)
        return torch.logsumexp(pdf+marks,dim=-1),torch.logsumexp(sf+marks,dim=-1),marks

    def _marked_log_prob(self,days,context,mark):
        pdf,_,marks=self._component_log_terms(days,context)
        return pdf[...,mark]+marks[mark]

    def iat_log_prob(self,days,inputs,history=()):
        if not np.isfinite(days).all() or (np.asarray(days)<=0).any():raise ValueError("Positive finite IAT required")
        return self._log_terms(days,self.build_context(inputs,history))[0]

    def iat_cdf(self,days,inputs,history=()):
        return -torch.expm1(self._log_terms(self._horizons(days),self.build_context(inputs,history))[1])

    def iat_survival(self,days,inputs,history=()):
        return self._log_terms(self._horizons(days),self.build_context(inputs,history))[1].exp()

    def _sequence_log_likelihood(self,seq):
        gaps=np.diff(np.r_[0.,seq.times]);start=int(self.entry_policy=="condition_on_first")
        if start and not len(gaps):return self.build_context(seq.static).sum()*0
        history=[];ll=self._tensor(0.)
        for j,gap in enumerate(gaps):
            if j>=start:
                ll=ll+self._marked_log_prob(gap,self.build_context(seq.static,history),
                                             int(seq.marks[j]) if self.n_marks>1 else 0)
            # Unknown first lifetime IAT never enters history under condition_on_first.
            if j>=start:history.append(gap)
        _,sf,_=self._log_terms(seq.end-(seq.times[-1] if len(gaps) else 0.),self.build_context(seq.static,history))
        return ll+sf

    def conditional_nll(self,sequences):
        self._check(sequences)
        return -torch.stack([ConditionalFlowTPP._sequence_log_likelihood(self,s) for s in sequences]).mean()

    def next_event_distribution(self,inputs,horizons,history=(),elapsed_days=0.):
        """Mixture forecast conditional on no event in the elapsed interval.

        mark_probabilities are the eventual-type probabilities at the query
        origin. mark_probabilities_at_event additionally condition on event
        time; mark_survival_probabilities condition on waiting through a horizon.
        """
        if not np.isfinite(elapsed_days) or elapsed_days<0:raise ValueError("Invalid elapsed time")
        h=self._horizons(horizons)
        with torch.no_grad():
            ctx=self.build_context(inputs,history)
            _,component_origin,marks=self._component_log_terms(elapsed_days,ctx)
            origin=torch.logsumexp(component_origin+marks,dim=-1)
            pdf,sf,_=self._component_log_terms(elapsed_days+h,ctx)
            log_sf=torch.logsumexp(sf+marks,dim=-1)
            ls=(log_sf-origin).cpu().numpy()
            marked_density=(pdf+marks-origin).exp().cpu().numpy()
            marked_hazard=(pdf+marks-log_sf[...,None]).exp().cpu().numpy()
            density=marked_density.sum(-1)
            hazard=marked_hazard.sum(-1)
            probs=torch.softmax(component_origin+marks,dim=-1).cpu().numpy()
            at_event=torch.softmax(pdf+marks,dim=-1).cpu().numpy()
            # An event exactly at zero has zero density in these positive-IAT flows.
            at_event=np.where(np.isfinite(at_event),at_event,probs)
            surviving=torch.softmax(sf+marks,dim=-1).cpu().numpy()
        return dict(horizons_days=h,survival=np.exp(ls),cdf=-np.expm1(ls),density=density,
                    hazard=hazard,integrated_hazard=-ls,mark_probabilities=probs,
                    mark_probabilities_at_event=at_event,mark_survival_probabilities=surviving,
                    marked_density=marked_density,marked_hazard=marked_hazard,
                    mark_conditional_density=(pdf-component_origin).exp().cpu().numpy(),
                    mark_conditional_survival=(sf-component_origin).exp().cpu().numpy())

    def quantile(self,probabilities,inputs,history=(),elapsed_days=0.):
        """Marginal mixture quantiles, found by bisection in log time."""
        p=np.asarray(probabilities,dtype=float)
        if not np.isfinite(p).all() or (p<=0).any() or (p>=1).any():raise ValueError("Quantiles strictly inside (0,1)")
        if not np.isfinite(elapsed_days) or elapsed_days<0:raise ValueError("Invalid elapsed time")
        with torch.no_grad():
            ctx=self.build_context(inputs,history)
            _,sf,marks=self._component_log_terms(elapsed_days,ctx)
            z=-ndtri_exp(np.log1p(-p)[...,None]+sf.cpu().numpy())
            logt,_=self._transform(self._tensor(z),ctx,inverse=True)
            lo,hi=logt.min(dim=-1).values,logt.max(dim=-1).values
            origin=torch.logsumexp(sf+marks,dim=-1)
            target=self._tensor(np.log1p(-p))+origin
            for _ in range(60 if self.n_marks>1 else 0):
                mid=(lo+hi)/2
                transformed,_=self._transform(mid[...,None],ctx)
                log_survival=torch.logsumexp(torch.special.log_ndtr(-transformed)+marks,dim=-1)
                lo=torch.where(log_survival>target,mid,lo)
                hi=torch.where(log_survival>target,hi,mid)
            result=self.time_scale*((lo+hi)/2).exp().cpu().numpy()-elapsed_days
        if not np.isfinite(result).all():raise FloatingPointError("Flow quantile exceeds numeric range")
        return np.maximum(result,0.)

    def sample_next_event(self,inputs,history=(),n_samples=1,elapsed_days=0.,seed=None):
        """Jointly sample eventual mark then its truncated mark-specific IAT."""
        if n_samples<1:raise ValueError("n_samples must be positive")
        if not np.isfinite(elapsed_days) or elapsed_days<0:raise ValueError("Invalid elapsed time")
        rng=np.random.default_rng(seed)
        with torch.no_grad():
            ctx=self.build_context(inputs,history)
            _,sf,log_marks=self._component_log_terms(elapsed_days,ctx)
            weights=torch.softmax(sf+log_marks,dim=-1).cpu().numpy()
            marks=rng.choice(self.n_marks,size=n_samples,p=weights)
            p=rng.uniform(np.finfo(float).eps,1-np.finfo(float).eps,n_samples)
            z=-ndtri_exp(np.log1p(-p)[:,None]+sf.cpu().numpy())
            logt,_=self._transform(self._tensor(z),ctx,inverse=True)
            selected=logt.cpu().numpy()[np.arange(n_samples),marks]
            waits=self.time_scale*np.exp(selected)-elapsed_days
        if not np.isfinite(waits).all() or (waits<=0).any():
            raise FloatingPointError("Sampled waiting time exceeds numeric range")
        return waits,marks

    def sample_next_iat(self,inputs,history=(),n_samples=1,elapsed_days=0.,seed=None):
        return self.sample_next_event(inputs,history,n_samples,elapsed_days,seed)[0]

    def predict_distribution(self,history,query):
        """Common adapter: query supplies inputs, horizon days and elapsed days."""
        return self.next_event_distribution(query['inputs'],query['horizons'],history=history,
                                            elapsed_days=query.get('elapsed_days',0.))

    def sample(self,history,query,n_samples=100,seed=None):
        return self.sample_next_iat(query['inputs'],history=history,n_samples=n_samples,
                                    elapsed_days=query.get('elapsed_days',0.),seed=seed)

    def simulate_timeline(self,inputs,end_days,history=(),current_time=0.,elapsed_days=0.,seed=None,max_events=10000):
        if not np.isfinite([end_days,current_time,elapsed_days]).all() or not 0<=current_time<=end_days or elapsed_days<0:
            raise ValueError("Invalid simulation window")
        rng=np.random.default_rng(seed);times=[];marks=[];past=list(history);time=current_time;elapsed=elapsed_days
        for _ in range(max_events):
            waits,event_marks=self.sample_next_event(inputs,past,1,elapsed,seed=int(rng.integers(2**31)))
            wait=float(waits[0])
            if wait<=0:raise FloatingPointError("Sampled wait below numeric resolution")
            time+=wait
            if time>end_days:return {"times":np.array(times),"marks":np.array(marks,dtype=int)}
            times.append(time);marks.append(int(event_marks[0]));past.append(elapsed+wait);elapsed=0.
        raise RuntimeError("Simulation exceeded max_events")
