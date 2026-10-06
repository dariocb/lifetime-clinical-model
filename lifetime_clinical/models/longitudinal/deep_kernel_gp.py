"""Model 1: exact deep-kernel GP with independent Gaussian measurement noise."""
import numpy as np
import torch
from .gp_base import NeuralGP
from .distributions import GPForecast
from ...components.gp_linalg import normal_log_prob

class DeepKernelGP(NeuralGP):
    model_name = "gp_1"
    def _observations(self, block, queries):
        return self._tensor(block.y), self._tensor(0.)
    def _block_log_likelihood(self, block):
        queries = self.build_gp_inputs(block)
        values, jacobian = self._observations(block, queries)
        covariance = self.kernel(queries)+torch.diag(self.noise_variance(queries))
        return (normal_log_prob(values, self.mean(queries), covariance, self.jitter)
                +jacobian-self._tensor(np.log(block.scales[block.variables])).sum())
    def exact_marginal_nll(self, blocks): return self.nll(blocks)
    def posterior(self, block, ages, variables):
        query = self.build_gp_inputs(block, ages, variables)
        values, _ = self._observations(block, self.build_gp_inputs(block))
        return self._gaussian_posterior(block, query, values)
    def forecast(self, block, ages, variables, retrospective=False):
        self._check([block], allow_empty=True)
        ages = np.atleast_1d(np.asarray(ages, dtype=float))
        if not retrospective and len(block.y) and np.any(ages < block.ages.max()):
            raise ValueError("Earlier queries require explicit interpolate/retrospective mode")
        with torch.no_grad():
            query = self.build_gp_inputs(block, ages, variables)
            values, _ = self._observations(block, self.build_gp_inputs(block))
            mean, covariance = self._gaussian_posterior(block, query, values)
            ids = np.atleast_1d(np.asarray(variables, dtype=int))
            return GPForecast("gaussian", mean, covariance, self.noise_variance(query),
                              block.offsets[ids], block.scales[ids], jitter=self.jitter)
