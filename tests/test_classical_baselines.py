import numpy as np
import torch
from scipy.integrate import quad
from lifetime_clinical.data.baselines import TPPSequence
from lifetime_clinical.models.tpp.classical import process_nll,CovariatePointProcess
from lifetime_clinical.data.synthetic import YEAR_DAYS


def sequence():
    return TPPSequence('p',np.array([.5,1.]),np.zeros((4,2)),
                       np.array([.2,.6,1.1])*YEAR_DAYS,np.array([0,1,2]),1.8*YEAR_DAYS)


def test_poisson_and_hawkes_conditional_integrals():
    s=sequence();start=.7
    assert np.isclose(float(process_nll(s,[2.],start=start)),2*(1.8-start)-np.log(2.))
    mu,alpha,delta=.4,.5,1.2
    intensity=lambda t: mu+sum(alpha*np.exp(-delta*(t-u)) for u in s.times/YEAR_DAYS if u<t)
    integral=quad(intensity,start,1.8,points=[1.1])[0]
    expected=integral-np.log(intensity(1.1))
    assert np.isclose(float(process_nll(s,[mu],[[alpha]],delta,start=start)),expected)


def test_marked_stability_gradients_and_score_decomposition():
    s=sequence();model=CovariatePointProcess(2,'marked_hawkes')
    mu,alpha,delta=model.parameters_for(s)
    assert torch.all(alpha.sum(0)/delta<.95)
    full=process_nll(s,mu,alpha,delta)
    timing=process_nll(s,mu,alpha,delta,timing_only=True)
    assert full>=timing
    full.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
