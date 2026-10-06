"""Differentiable SPD linear algebra, including CPU Torch builds without LAPACK.

SciPy fallback implements first-order autograd; higher-order derivatives are not
supported by that path. Ordinary LAPACK-enabled CPU/CUDA Torch uses native ops.
"""
import numpy as np
import scipy.linalg as sla
import torch

_NATIVE = {}


def native_available(device):
    key = str(device)
    if key not in _NATIVE:
        try:
            a = torch.eye(2, dtype=torch.double, device=device)
            torch.linalg.cholesky(a)
            torch.linalg.solve_triangular(a, a, upper=False)
            _NATIVE[key] = True
        except RuntimeError:
            _NATIVE[key] = False
    return _NATIVE[key]


class _Cholesky(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix):
        array = sla.cholesky(matrix.detach().cpu().numpy(), lower=True)
        factor = torch.as_tensor(array.copy(), dtype=matrix.dtype, device=matrix.device)
        ctx.save_for_backward(factor)
        return factor

    @staticmethod
    def backward(ctx, gradient):
        (factor,) = ctx.saved_tensors
        l = factor.detach().cpu().numpy()
        g = gradient.detach().cpu().numpy()
        p = np.tril(l.T @ g)
        p[np.diag_indices_from(p)] *= .5
        result = sla.solve_triangular(l.T, p, lower=False)
        result = sla.solve_triangular(l.T, result.T, lower=False).T
        result = (result + result.T) * .5
        return torch.as_tensor(result.copy(), dtype=gradient.dtype, device=gradient.device)


class _TriangularSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, factor, rhs, upper):
        result = sla.solve_triangular(factor.detach().cpu().numpy(),
                                     rhs.detach().cpu().numpy(), lower=not upper)
        output = torch.as_tensor(result.copy(), dtype=rhs.dtype, device=rhs.device)
        ctx.upper = upper
        ctx.save_for_backward(factor, output)
        return output

    @staticmethod
    def backward(ctx, gradient):
        factor, output = ctx.saved_tensors
        l = factor.detach().cpu().numpy()
        db = sla.solve_triangular(l.T, gradient.detach().cpu().numpy(), lower=ctx.upper)
        da = -db @ output.detach().cpu().numpy().T
        da = np.triu(da) if ctx.upper else np.tril(da)
        return (torch.as_tensor(da.copy(), dtype=factor.dtype, device=factor.device),
                torch.as_tensor(db.copy(), dtype=gradient.dtype, device=gradient.device), None)


def cholesky(matrix, jitter=1e-6):
    """Symmetrize and add explicit numerical jitter; fail on invalid covariances."""
    matrix = (matrix + matrix.T) * .5
    identity = torch.eye(len(matrix), dtype=matrix.dtype, device=matrix.device)
    for multiplier in (1., 10., 100., 1000.):
        adjusted = matrix + identity * jitter * multiplier
        try:
            return (torch.linalg.cholesky(adjusted) if native_available(matrix.device)
                    else _Cholesky.apply(adjusted))
        except (RuntimeError, np.linalg.LinAlgError):
            continue
    raise FloatingPointError("GP covariance is not numerically positive definite")


def triangular_solve(factor, rhs, upper=False):
    vector = rhs.ndim == 1
    if vector:
        rhs = rhs[:, None]
    result = (torch.linalg.solve_triangular(factor, rhs, upper=upper)
              if native_available(factor.device) else _TriangularSolve.apply(factor, rhs, upper))
    return result[:, 0] if vector else result


def spd_solve(factor, rhs):
    return triangular_solve(factor.T, triangular_solve(factor, rhs), upper=True)


def normal_log_prob(value, mean, covariance, jitter=1e-6):
    factor = cholesky(covariance, jitter)
    whitened = triangular_solve(factor, value - mean)
    return -.5 * (whitened.square().sum() + 2 * torch.log(factor.diag()).sum()
                   + len(value) * np.log(2 * np.pi))
