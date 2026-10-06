import numpy as np
import torch
from lifetime_clinical.components.history import CausalHistoryEncoder
from lifetime_clinical.models.tpp.conditional_flow import ConditionalFlowTPP
from lifetime_clinical.models.tpp.joint_flow import JointFlowTPP
from lifetime_clinical.models.tpp.shared_private_flow import SharedPrivateFlowTPP
from lifetime_clinical.models.longitudinal.deep_kernel_gp import DeepKernelGP
from lifetime_clinical.models.longitudinal.transformed_gp import TransformedObservationGP
from lifetime_clinical.models.longitudinal.residual_flow_gp import ResidualFlowGP


def test_all_models_use_common_causal_encoder():
    torch.set_num_threads(1)
    for model in (ConditionalFlowTPP(input_dim=5,hidden_dim=8),
                  JointFlowTPP(input_dim=5,hidden_dim=8,feature_indices=[0,1,2],input_mode='dynamic'),
                  SharedPrivateFlowTPP(input_dim=5,hidden_dim=8,feature_indices=[0,1,2],
                                       shared_dim=2,input_mode='dynamic')):
        if getattr(model,'input_mode',None)=='dynamic':
            assert isinstance(model.feature_encoder,CausalHistoryEncoder)
            assert isinstance(model.dynamic_encoder,CausalHistoryEncoder)
        else: assert isinstance(model.encoder,CausalHistoryEncoder)
    for cls in (DeepKernelGP,TransformedObservationGP,ResidualFlowGP):
        model=cls(input_dim=5,history_dim=6,hidden_dim=8)
        assert isinstance(model.history_encoder,CausalHistoryEncoder)
        if isinstance(model,ResidualFlowGP): assert isinstance(model.posterior_encoder,CausalHistoryEncoder)


def test_prefix_states_ignore_future_tokens_and_empty_history_is_finite():
    torch.set_num_threads(1)
    encoder=CausalHistoryEncoder(3,8)
    tokens=np.random.default_rng(9).normal(size=(6,3))
    prefix=encoder.states(tokens[:3])
    whole=encoder.states(tokens)
    assert torch.allclose(prefix,whole[:4],atol=1e-10)
    assert torch.isfinite(encoder(np.empty((0,3)))).all()
