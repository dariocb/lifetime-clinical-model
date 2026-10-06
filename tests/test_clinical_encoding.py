import numpy as np
import torch
from lifetime_clinical.components.history import CausalHistoryEncoder,clinical_feature_names
from lifetime_clinical.tutorials.experiments import load_toy,tpp_data,gp_data
from lifetime_clinical.models.tpp.conditional_flow import ConditionalFlowTPP
from lifetime_clinical.models.tpp.joint_flow import JointFlowTPP
from lifetime_clinical.models.tpp.shared_private_flow import SharedPrivateFlowTPP
from lifetime_clinical.models.longitudinal.deep_kernel_gp import DeepKernelGP
from lifetime_clinical.models.longitudinal.transformed_gp import TransformedObservationGP
from lifetime_clinical.models.longitudinal.residual_flow_gp import ResidualFlowGP
from lifetime_clinical.models.lifetime import LifetimeModel,prepare_lifetime_data
from lifetime_clinical.data.tutorial import physiology_feature_indices
from pathlib import Path
from lifetime_clinical.data.synthetic import YEAR_DAYS


def test_absolute_age_positions_and_causal_prefixes():
    torch.set_num_threads(1)
    encoder=CausalHistoryEncoder(31,12,feature_names=clinical_feature_names('history',31))
    x=torch.zeros(4,31,dtype=torch.float64)
    x[:,1]=1; x[:,3]=torch.tensor([.2,.21,.23,.25]); x[:,12]=1
    changed=x.clone();changed[:,3]+=.1
    # Age never enters the content projection, but changes its positional code.
    assert torch.equal(encoder.input_encoder(x),encoder.input_encoder(changed))
    assert not torch.allclose(encoder.states(x),encoder.states(changed))
    assert torch.allclose(encoder.states(x[:2]),encoder.states(x)[:3],atol=1e-10)
    assert torch.isfinite(encoder(np.empty((0,31)))).all()


def test_all_seven_models_train_their_categorical_embeddings():
    torch.set_num_threads(1)
    dataset,splits,pre=load_toy(Path(__file__).resolve().parents[1]/'data/toy')
    static=tpp_data(dataset,splits,pre)['train'][:1]
    dynamic=tpp_data(dataset,splits,pre,dynamic=True)['train'][:1]
    blocks=gp_data(dataset,splits,pre)['train'][:1]
    feature_indices=physiology_feature_indices(pre)
    cases=[(ConditionalFlowTPP(input_dim=20,hidden_dim=12,n_marks=3),static)]
    for cls,kwargs in ((JointFlowTPP,{}),(SharedPrivateFlowTPP,dict(shared_dim=2))):
        cases.append((cls(input_dim=55,hidden_dim=12,n_marks=3,feature_indices=feature_indices,
                          input_mode='dynamic',**kwargs),dynamic))
    for cls in (DeepKernelGP,TransformedObservationGP,ResidualFlowGP):
        cases.append((cls(input_dim=20,history_dim=31,hidden_dim=12),blocks))
    for model,data in cases:
        loss=model.nll(data);loss.backward()
        gradients=[m.weight.grad for m in model.modules() if isinstance(m,torch.nn.Embedding)]
        assert torch.isfinite(loss)
        assert any(g is not None and g.abs().sum()>0 for g in gradients)
    lifetime=LifetimeModel(20,31)
    records=prepare_lifetime_data(dataset,pre,splits)['train'][:1]
    lifetime.objective(records).sum().backward()
    assert lifetime.gp.history_encoder is lifetime.shared
    assert any(m.weight.grad is not None and m.weight.grad.abs().sum()>0
               for m in lifetime.modules() if isinstance(m,torch.nn.Embedding))


def test_generated_snapshots_advance_the_absolute_age_clock():
    torch.set_num_threads(1)
    names=clinical_feature_names('dynamic',55)
    context=np.zeros(55)
    context[0]=.2
    context[names.index('current_age_scaled')]=.3
    context[names.index('log1p_days_since_entry')]=np.log1p(10*YEAR_DAYS)
    model=JointFlowTPP(input_dim=55,hidden_dim=12,n_marks=3,
                       feature_indices=[38,41,44],input_mode='dynamic')
    result=model.simulate_timeline(context,10*YEAR_DAYS,seed=42)
    assert len(result['times'])>0
    ages=result['features'][:,names.index('current_age_scaled')]
    assert np.allclose(ages,.3+np.r_[0.,result['times']]/YEAR_DAYS/100.)
