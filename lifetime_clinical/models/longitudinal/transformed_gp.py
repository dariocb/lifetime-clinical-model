"""Model 2: exact invertible transformation of Gaussian GP measurements."""
import numpy as np
import torch
from .deep_kernel_gp import DeepKernelGP
from .distributions import GPForecast
from ...components.observation_flows import ObservationFlow

class TransformedObservationGP(DeepKernelGP):
    model_name = "gp_2"
    def __init__(self, flow_layers=2, **kwargs):
        if flow_layers < 1: raise ValueError("Positive flow_layers required")
        self.flow_layers = int(flow_layers)
        super().__init__(**kwargs)
    def _extra_initialize(self):
        self.observation_flow = ObservationFlow(self.query_dim, self.hidden_dim, self.flow_layers,self.query_feature_names)
    def _observations(self, block, queries):
        values, jacobian = self.observation_flow(self._tensor(block.y), queries)
        return values, jacobian.sum()
    def inverse_observations(self, values, queries):
        return self.observation_flow(self._tensor(values), queries)[0]
    def inverse_log_det(self, values, queries):
        return self.observation_flow(self._tensor(values), queries)[1]
    def posterior_latent(self, block, ages, variables): return self.posterior(block, ages, variables)
    def transformed_density(self, block, ages, variables, values):
        return self.forecast(block, ages, variables).log_prob(values)
    def forecast(self, block, ages, variables, retrospective=False):
        self._check([block], allow_empty=True)
        ages = np.atleast_1d(np.asarray(ages, dtype=float))
        if not retrospective and len(block.y) and np.any(ages < block.ages.max()):
            raise ValueError("Earlier queries require explicit retrospective mode")
        with torch.no_grad():
            query = self.build_gp_inputs(block, ages, variables)
            values, _ = self._observations(block, self.build_gp_inputs(block))
            mean, covariance = self._gaussian_posterior(block, query, values)
            ids = np.atleast_1d(np.asarray(variables, dtype=int))
            return GPForecast("transformed", mean, covariance, self.noise_variance(query),
                              block.offsets[ids], block.scales[ids], self.observation_flow, query, self.jitter)
