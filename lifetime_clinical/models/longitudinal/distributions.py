"""Joint trajectory sampling and raw-unit predictive inference for GP 1/2/3."""
from copy import deepcopy
import numpy as np
import scipy.linalg as sla
from scipy.special import ndtr, ndtri, logsumexp
from scipy.optimize import brentq
from numpy.polynomial.hermite import hermgauss
import torch


class GPForecast:
    def __init__(self, kind, latent_mean, latent_covariance, noise_variance, offsets, scales,
                 flow=None, context=None, jitter=1e-6):
        self.kind = kind
        self.latent_mean_z = latent_mean.detach().cpu().numpy()
        self.latent_covariance_z = latent_covariance.detach().cpu().numpy()
        self.noise_variance_z = noise_variance.detach().cpu().numpy()
        self.offsets, self.scales = np.asarray(offsets), np.asarray(scales)
        self.flow = deepcopy(flow).eval() if flow is not None else None
        self.context = context.detach().clone() if context is not None else None
        self.jitter = jitter

    def _transform(self, values, inverse=False):
        with torch.no_grad():
            tensor = torch.as_tensor(values, dtype=self.context.dtype, device=self.context.device)
            result, jacobian = self.flow(tensor, self.context, inverse=inverse)
        return result.cpu().numpy(), jacobian.cpu().numpy()

    def _value(self, values):
        value = np.asarray(values, dtype=float)
        if value.shape[-1:] != self.offsets.shape or not np.isfinite(value).all():
            raise ValueError("Supply finite values with one column per query")
        return (value-self.offsets)/self.scales

    def _measurement_covariance(self):
        return self.latent_covariance_z+np.diag(self.noise_variance_z)

    def _factor(self, covariance):
        covariance = (covariance+covariance.T)*.5
        return sla.cholesky(covariance+self.jitter*np.eye(len(covariance)), lower=True)

    def latent_sample(self, n_samples=100, seed=None, raw_units=True):
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        rng = np.random.default_rng(seed)
        samples = self.latent_mean_z+rng.normal(size=(n_samples, len(self.offsets))) @ self._factor(self.latent_covariance_z).T
        # Model 2 latent coordinates are Gaussian-space values, not raw physiology.
        if raw_units and self.kind != "transformed":
            samples = self.offsets+self.scales*samples
        return samples

    def sample(self, n_samples=100, seed=None):
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        rng = np.random.default_rng(seed)
        if self.kind == "residual":
            f = self.latent_mean_z+rng.normal(size=(n_samples, len(self.offsets))) @ self._factor(self.latent_covariance_z).T
            residual, _ = self._transform(rng.normal(size=f.shape), inverse=True)
            samples = f+residual
        else:
            samples = self.latent_mean_z+rng.normal(size=(n_samples, len(self.offsets))) @ self._factor(self._measurement_covariance()).T
            if self.kind == "transformed":
                samples, _ = self._transform(samples, inverse=True)
        return self.offsets+self.scales*samples

    def _latent_quadrature(self, nodes=40):
        points, weights = hermgauss(nodes)
        sd = np.sqrt(np.maximum(np.diag(self.latent_covariance_z), 0))
        return self.latent_mean_z+np.sqrt(2)*points[:, None]*sd, weights/np.sqrt(np.pi)

    def marginal_log_prob(self, values):
        """Per-query density; residual model uses one-dimensional quadrature."""
        y = self._value(values)
        if y.ndim != 1:
            raise ValueError("Marginal scoring expects one value per query")
        if self.kind == "residual":
            f, weights = self._latent_quadrature()
            z, ld = self._transform(y-f)
            result = logsumexp(-.5*z*z-.5*np.log(2*np.pi)+ld+np.log(weights)[:, None], axis=0)
        else:
            variance = np.maximum(np.diag(self._measurement_covariance()), self.jitter)
            ld = 0.
            if self.kind == "transformed":
                y, ld = self._transform(y)
            result = -.5*((y-self.latent_mean_z)**2/variance+np.log(2*np.pi*variance))+ld
        return result-np.log(self.scales)

    def log_prob(self, values, n_samples=2048, seed=42):
        """Joint score; Model 3 is a Monte Carlo posterior-mixture estimate."""
        y = self._value(values)
        if y.ndim != 1:
            raise ValueError("Joint scoring expects one future trajectory")
        if self.kind == "residual":
            if n_samples < 1:
                raise ValueError("Positive Monte Carlo count required")
            f = self.latent_sample(n_samples, seed, raw_units=False)
            z, ld = self._transform(y-f)
            terms = (-.5*z*z-.5*np.log(2*np.pi)+ld).sum(axis=1)
            result = logsumexp(terms)-np.log(n_samples)
        else:
            ld = 0.
            if self.kind == "transformed":
                y, jacobian = self._transform(y)
                ld = jacobian.sum()
            factor = self._factor(self._measurement_covariance())
            whitened = sla.solve_triangular(factor, y-self.latent_mean_z, lower=True)
            result = -.5*(whitened @ whitened+2*np.log(factor.diagonal()).sum()+len(y)*np.log(2*np.pi))+ld
        return float(result-np.log(self.scales).sum())

    def cdf(self, values):
        """Scalar marginal CDFs, not a multivariate joint CDF."""
        y = self._value(values)
        if y.ndim != 1:
            raise ValueError("CDF expects one threshold per query")
        if self.kind == "residual":
            f, weights = self._latent_quadrature()
            z, _ = self._transform(y-f)
            return (ndtr(z)*weights[:, None]).sum(axis=0)
        if self.kind == "transformed":
            y, _ = self._transform(y)
        sd = np.sqrt(np.maximum(np.diag(self._measurement_covariance()), self.jitter))
        return ndtr((y-self.latent_mean_z)/sd)

    def quantile(self, probabilities):
        p = np.atleast_1d(np.asarray(probabilities, dtype=float))
        if p.ndim != 1 or not np.isfinite(p).all() or (p <= 0).any() or (p >= 1).any():
            raise ValueError("Quantile probabilities must be inside (0,1)")
        if self.kind != "residual":
            sd = np.sqrt(np.maximum(np.diag(self._measurement_covariance()), self.jitter))
            values = self.latent_mean_z+ndtri(p)[:, None]*sd
            if self.kind == "transformed":
                values, _ = self._transform(values, inverse=True)
            return self.offsets+self.scales*values
        values = np.empty((len(p), len(self.offsets)))
        center = self.mean
        for j in range(len(self.offsets)):
            def scalar_cdf(value):
                threshold = center.copy()
                threshold[j] = value
                return self.cdf(threshold)[j]
            for i, probability in enumerate(p):
                width = 8*self.scales[j]
                for _ in range(40):
                    lo, hi = center[j]-width, center[j]+width
                    if scalar_cdf(lo) <= probability <= scalar_cdf(hi):
                        break
                    width *= 2
                else:
                    raise FloatingPointError("Unable to bracket predictive quantile")
                values[i, j] = brentq(lambda v: scalar_cdf(v)-probability, lo, hi)
        return values

    def _moments(self):
        points, weights = hermgauss(40)
        if self.kind == "gaussian":
            return self.latent_mean_z, np.diag(self._measurement_covariance())
        if self.kind == "transformed":
            sd = np.sqrt(np.maximum(np.diag(self._measurement_covariance()), 0))
            base = self.latent_mean_z+np.sqrt(2)*points[:, None]*sd
        else:
            base = np.sqrt(2)*points[:, None]*np.ones_like(self.latent_mean_z)
        observed, _ = self._transform(base, inverse=True)
        mean = ((weights/np.sqrt(np.pi))[:, None]*observed).sum(axis=0)
        variance = ((weights/np.sqrt(np.pi))[:, None]*(observed-mean)**2).sum(axis=0)
        if self.kind == "residual":
            mean = mean+self.latent_mean_z
            variance = variance+np.diag(self.latent_covariance_z)
        return mean, variance

    @property
    def mean(self):
        return self.offsets+self.scales*self._moments()[0]

    @property
    def variance(self):
        return self.scales**2*self._moments()[1]

    def covariance(self, n_samples=5000, seed=42):
        if self.kind == "gaussian":
            return self._measurement_covariance()*self.scales[:, None]*self.scales[None, :]
        if self.kind == "residual":
            covariance = self.latent_covariance_z*self.scales[:, None]*self.scales[None, :]
            np.fill_diagonal(covariance, self.variance)
            return covariance
        if n_samples < 2:
            raise ValueError("Covariance estimation needs at least two samples")
        return np.atleast_2d(np.cov(self.sample(n_samples, seed), rowvar=False))
