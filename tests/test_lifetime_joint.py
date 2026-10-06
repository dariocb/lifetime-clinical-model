import numpy as np
import torch
from lifetime_clinical.models.lifetime.joint import LifetimeModel, LifetimePatient
from lifetime_clinical.data.longitudinal import GPBlock


def example():
    rng=np.random.default_rng(8)
    tokens=rng.normal(size=(7,6))
    days=np.array([-2.,0.,2.,3.,5.,7.,8.])
    block=GPBlock("p",np.array([.4,1.]),tokens[:2],np.array([.51,.52,.53]),
                  np.array([0,1,2]),np.array([.1,-.2,.3]),np.array([120.,90.,1.]),np.array([10.,8.,.1]))
    return LifetimePatient(block,tokens,days,np.array([3.,7.]),np.array([0,2]),10.,50.)


def test_causality_and_shared_gradients():
    torch.set_num_threads(1)
    patient=example(); model=LifetimeModel(2,6)
    whole=model.shared.states(patient.tokens)
    prefix=model.shared.states(patient.tokens[:3])
    assert torch.allclose(whole[:4],prefix,atol=1e-10)
    event,gp=model.patient_terms(patient)
    for term in (event,gp):
        gradient=torch.autograd.grad(term,model.shared.projection.weight,retain_graph=True)[0]
        assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
    assert torch.allclose(model.objective([patient]).sum(),event+gp)


def test_batched_flow_and_forecast_survival():
    torch.set_num_threads(1)
    patient=example(); model=LifetimeModel(2,6)
    contexts=torch.randn(2,14,dtype=torch.float64)
    batch=model.tpp.batch_marked_log_prob([2.,4.],contexts,[0,2])
    scalar=torch.stack([model.tpp._marked_log_prob(2.,contexts[0],0),
                        model.tpp._marked_log_prob(4.,contexts[1],2)])
    assert torch.allclose(batch,scalar,atol=1e-12)
    prediction=model.predict_events(patient,[0.,1.,10.],8.)
    assert np.isclose(prediction['survival'][0],1.)
    assert np.all(np.diff(prediction['survival'])<=0)
    # Future record values cannot affect a forecast from the last observed visit.
    patient.tokens[-1]+=100
    unchanged=model.predict_events(patient,[0.,1.,10.],8.)
    assert np.allclose(prediction['density'],unchanged['density'])
