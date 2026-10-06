"""Model 3: sparse variational GP physiology plus independent flow residuals.

Amortized diagonal q(v), with whitened inducing variables u=m_u+L_uu v.
The GP function marginal has full query covariance despite diagonal q(v).
"""
import zlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .gp_base import NeuralGP
from ...components.history import CausalHistoryEncoder,NeuralInputs
from .distributions import GPForecast
from ...components.observation_flows import ObservationFlow
from ...components.gp_linalg import cholesky, triangular_solve

class ResidualFlowGP(NeuralGP):
    model_name = "gp_3"
    def __init__(self, flow_layers=2, inducing_per_variable=3, elbo_samples=16,
                 inducing_age_range=(0.,1.), **kwargs):
        if flow_layers<1 or inducing_per_variable<1 or elbo_samples<2:
            raise ValueError("Positive flow/inducing sizes and at least two ELBO samples required")
        if len(inducing_age_range)!=2 or not np.isfinite(inducing_age_range).all() or inducing_age_range[0]>=inducing_age_range[1]:
            raise ValueError("Ordered finite inducing ages required, in age/100 units")
        self.flow_layers,self.inducing_per_variable,self.elbo_samples=int(flow_layers),int(inducing_per_variable),int(elbo_samples)
        self.inducing_age_range=tuple(inducing_age_range)
        super().__init__(**kwargs)
    def _extra_initialize(self):
        self.raw_noise.requires_grad_(False)  # Flow residual replaces Gaussian observation noise.
        self.residual_flow=ObservationFlow(self.query_dim,self.hidden_dim,self.flow_layers,self.query_feature_names)
        recognition_names=["query_age_scaled"]+[f"output={i}" for i in range(self.n_variables)]+["value_z"]
        self.posterior_encoder=CausalHistoryEncoder(self.n_variables+2,self.hidden_dim,
            heads=2 if self.hidden_dim%2==0 else 1,feature_names=recognition_names)
        posterior_inputs=NeuralInputs(self.hidden_dim+self.input_dim+self.hidden_dim,self.hidden_dim,
            [f"recognition_{i}" for i in range(self.hidden_dim)]+self.query_feature_names[:self.input_dim+self.hidden_dim])
        self.posterior_head=nn.Sequential(posterior_inputs,nn.Linear(posterior_inputs.output_dim,
                                      2*self.inducing_per_variable*self.n_variables)).double()
        nn.init.normal_(self.posterior_head[-1].weight,std=.002);nn.init.zeros_(self.posterior_head[-1].bias)
        with torch.no_grad():
            self.posterior_head[-1].bias[self.inducing_per_variable*self.n_variables:]=np.log(np.expm1(.99))
    def _inducing_inputs(self,block):
        ages=np.repeat(np.linspace(*self.inducing_age_range,self.inducing_per_variable),self.n_variables)
        ids=np.tile(np.arange(self.n_variables),self.inducing_per_variable)
        return self.build_gp_inputs(block,ages,ids)
    def _recognition(self,block):
        count=self.inducing_per_variable*self.n_variables
        if not len(block.y):return self._tensor(np.zeros(count)),self._tensor(np.ones(count))
        tokens=np.column_stack([block.ages,np.eye(self.n_variables)[block.variables],block.y])
        state=self.posterior_encoder(tokens)
        context=self.build_gp_inputs(block)[:1,:self.input_dim+self.hidden_dim][0]
        raw=self.posterior_head(torch.cat([state,context]));m,raw_s=raw.chunk(2)
        return m,F.softplus(raw_s)+.01
    def _q_moments(self,block,query,diagonal=False):
        inducing=self._inducing_inputs(block)
        factor=cholesky(self.kernel(inducing),self.jitter)
        a=triangular_solve(factor,self.kernel(inducing,query)).T
        m,s=self._recognition(block)
        mean=self.mean(query)+a@m
        if diagonal:
            variance=self.kernel_diagonal(query)-a.square().sum(-1)+(a*s).square().sum(-1)
            return mean,variance.clamp_min(self.jitter),m,s
        covariance=self.kernel(query)-a@a.T+(a*s)@(a*s).T
        return mean,(covariance+covariance.T)*.5,m,s
    def residual_log_prob(self,residual,queries):
        return self.residual_flow.log_prob(self._tensor(residual),queries)
    def _block_elbo(self,block,n_samples):
        query=self.build_gp_inputs(block)
        mean,variance,m,s=self._q_moments(block,query,diagonal=True)
        seed=(self.seed+zlib.crc32(block.patient_id.encode()))%(2**32)
        rng=np.random.default_rng(seed)
        half=rng.normal(size=((n_samples+1)//2,len(query)))
        noise=np.concatenate([half,-half])[:n_samples]
        # The residual likelihood is a sum of independent per-row log densities.
        # Its expectation depends only on q(f_j) marginals; joint f sampling is
        # unnecessary here. Forecasts still retain the full function covariance.
        f=mean+self._tensor(noise)*torch.sqrt(variance)
        expected=self.residual_log_prob(self._tensor(block.y)-f,query).sum(-1).mean()
        kl=.5*(m.square()+s.square()-2*torch.log(s)-1).sum()
        return expected-kl-self._tensor(np.log(block.scales[block.variables])).sum()
    def elbo(self,blocks,n_samples=None):
        self._check(blocks)
        samples=self.elbo_samples if n_samples is None else int(n_samples)
        if samples<2:raise ValueError("At least two ELBO samples required")
        return torch.stack([self._block_elbo(b,samples) for b in blocks])
    def nll(self,blocks):
        """Common optimizer interface: negative ELBO, not exact marginal NLL."""
        return -self.elbo(blocks).mean()
    def log_likelihood(self,blocks,observation_windows=None):
        raise ValueError("Model 3 has no analytic marginal likelihood; use elbo or forecast(...).log_prob")
    def variational_posterior(self,block,ages,variables):
        query=self.build_gp_inputs(block,ages,variables)
        mean,covariance,_,_=self._q_moments(block,query)
        return mean,covariance
    def sample_residuals(self,block,ages,variables,n_samples=100,seed=None):
        if n_samples<1:raise ValueError("Positive sample count required")
        query=self.build_gp_inputs(block,ages,variables)
        noise=np.random.default_rng(seed).normal(size=(n_samples,len(query)))
        with torch.no_grad():result,_=self.residual_flow(self._tensor(noise),query,inverse=True)
        ids=np.atleast_1d(np.asarray(variables,dtype=int))
        return result.cpu().numpy()*block.scales[ids]  # Residuals have no raw-unit mean offset.
    def forecast(self,block,ages,variables,retrospective=False):
        self._check([block],allow_empty=True)
        ages=np.atleast_1d(np.asarray(ages,dtype=float))
        if not retrospective and len(block.y) and np.any(ages<block.ages.max()):
            raise ValueError("Earlier queries require explicit retrospective mode")
        with torch.no_grad():
            query=self.build_gp_inputs(block,ages,variables)
            mean,covariance,_,_=self._q_moments(block,query)
            ids=np.atleast_1d(np.asarray(variables,dtype=int))
            return GPForecast("residual",mean,covariance,torch.zeros_like(mean),block.offsets[ids],
                              block.scales[ids],self.residual_flow,query,self.jitter)
