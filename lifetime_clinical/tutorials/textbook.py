"""Minimal textbook constructions for the introductory notebooks."""
from __future__ import annotations
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def simulate_poisson(rate=1.0, horizon=10.0, seed=0):
    rng=np.random.default_rng(seed); t=0.; events=[]
    while True:
        t += rng.exponential(1/rate)
        if t>horizon: break
        events.append(t)
    return np.asarray(events)


def hawkes_intensity(grid, events, mu=.35, alpha=.8, beta=1.5):
    grid=np.asarray(grid,float); events=np.asarray(events,float)
    out=np.full_like(grid,mu)
    for ti in events:
        mask=grid>ti
        out[mask]+=alpha*np.exp(-beta*(grid[mask]-ti))
    return out


def simulate_hawkes(mu=.35, alpha=.7, beta=1.4, horizon=12., seed=3):
    """Simple Ogata thinning simulation for an exponential Hawkes process."""
    if alpha>=beta:
        raise ValueError("Use alpha < beta for a stable textbook example")
    rng=np.random.default_rng(seed); t=0.; events=[]
    while t<horizon:
        upper=mu+sum(alpha*np.exp(-beta*(t-ti)) for ti in events)
        t += rng.exponential(1/max(upper,1e-12))
        if t>=horizon: break
        lam=mu+sum(alpha*np.exp(-beta*(t-ti)) for ti in events)
        if rng.uniform()<=lam/upper: events.append(t)
    return np.asarray(events)


def rbf_kernel(x, y, variance=1.0, lengthscale=1.0):
    x=np.atleast_1d(x)[:,None]; y=np.atleast_1d(y)[None,:]
    return variance*np.exp(-.5*((x-y)/lengthscale)**2)


def gp_posterior(x_train,y_train,x_test,noise=.12,variance=1.,lengthscale=1.2):
    x_train=np.asarray(x_train,float); y_train=np.asarray(y_train,float); x_test=np.asarray(x_test,float)
    K=rbf_kernel(x_train,x_train,variance,lengthscale)+noise**2*np.eye(len(x_train))
    Ks=rbf_kernel(x_train,x_test,variance,lengthscale)
    Kss=rbf_kernel(x_test,x_test,variance,lengthscale)
    cf=cho_factor(K,lower=True)
    mean=Ks.T@cho_solve(cf,y_train)
    cov=Kss-Ks.T@cho_solve(cf,Ks)
    return mean,(cov+cov.T)*.5


def scaled_sinh_arcsinh_forward(z, skew=.7, tail=1.25):
    z=np.asarray(z,float)
    return np.sinh(tail*(np.arcsinh(z)+skew))


def scaled_sinh_arcsinh_inverse(y, skew=.7, tail=1.25):
    y=np.asarray(y,float)
    return np.sinh(np.arcsinh(y)/tail-skew)


def causal_attention(tokens, seed=0):
    """One-head attention demonstration with identity-sized learned projections."""
    x=np.asarray(tokens,float); d=x.shape[1]; rng=np.random.default_rng(seed)
    Wq=rng.normal(scale=.45,size=(d,d)); Wk=rng.normal(scale=.45,size=(d,d)); Wv=rng.normal(scale=.45,size=(d,d))
    q,k,v=x@Wq,x@Wk,x@Wv
    scores=q@k.T/np.sqrt(d)
    mask=np.triu(np.ones_like(scores,dtype=bool),1); scores=np.where(mask,-np.inf,scores)
    scores=scores-np.max(scores,axis=1,keepdims=True)
    weights=np.exp(scores); weights/=weights.sum(axis=1,keepdims=True)
    return weights,weights@v
