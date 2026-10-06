"""Compact experiment builders used by the model notebooks."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from ..data import load_dataset, load_splits
from ..data.preprocessing import fit_static_transform
from ..data.baselines import prepare_baseline_sequences
from ..data.tutorial import prepare_dynamic_flow_sequences, physiology_feature_indices
from ..data.longitudinal import prepare_gp_blocks, split_forecast_block
from ..models.tpp.conditional_flow import ConditionalFlowTPP
from ..models.tpp.joint_flow import JointFlowTPP
from ..models.tpp.shared_private_flow import SharedPrivateFlowTPP
from ..models.longitudinal.deep_kernel_gp import DeepKernelGP
from ..models.longitudinal.transformed_gp import TransformedObservationGP
from ..models.longitudinal.residual_flow_gp import ResidualFlowGP


def load_toy(root):
    root=Path(root)
    dataset=load_dataset(root); splits=load_splits(root)
    pre=fit_static_transform(dataset,splits['train'],use_ehr=False)
    return dataset,splits,pre


def tpp_data(dataset,splits,preprocessor,dynamic=False):
    maker=prepare_dynamic_flow_sequences if dynamic else prepare_baseline_sequences
    return {name:maker(dataset,preprocessor,ids) for name,ids in splits.items()}


def fit_tpp_a(data,epochs=8,seed=42,device="cpu"):
    model=ConditionalFlowTPP(hidden_dim=24,n_marks=3,seed=seed,flow_layers=2,
                             transformer_heads=2,transformer_layers=1)
    model.fit(data['train'],validation_sequences=data['validation'],epochs=epochs,lr=.008,batch_size=8,patience=max(3,epochs),device=device)
    return model


def fit_tpp_b(data,preprocessor,epochs=8,seed=42,device="cpu"):
    model=JointFlowTPP(feature_indices=physiology_feature_indices(preprocessor),input_mode='dynamic',
                       hidden_dim=24,n_marks=3,seed=seed,flow_layers=2,feature_layers=3)
    model.fit(data['train'],validation_sequences=data['validation'],epochs=epochs,lr=.006,batch_size=6,patience=max(3,epochs),device=device)
    return model


def fit_tpp_c(data,preprocessor,epochs=8,seed=42,device="cpu"):
    model=SharedPrivateFlowTPP(feature_indices=physiology_feature_indices(preprocessor),input_mode='dynamic',
                              shared_dim=2,use_private=False,hidden_dim=24,n_marks=3,seed=seed,
                              flow_layers=2,feature_layers=3)
    model.fit(data['train'],validation_sequences=data['validation'],epochs=epochs,lr=.006,batch_size=6,patience=max(3,epochs),device=device)
    return model


def gp_data(dataset,splits,preprocessor,history_fraction=.25):
    return {name:prepare_gp_blocks(dataset,preprocessor,ids,history_fraction=history_fraction)
            for name,ids in splits.items()}


def fit_gp_a(data,epochs=8,seed=42,device="cpu"):
    model=DeepKernelGP(hidden_dim=12,kernel_rank=6,seed=seed)
    model.fit(data['train'],validation_blocks=data['validation'],epochs=epochs,lr=.006,batch_size=4,patience=max(3,epochs),device=device)
    return model


def fit_gp_b(data,epochs=8,seed=42,device="cpu"):
    model=TransformedObservationGP(hidden_dim=12,kernel_rank=6,flow_layers=2,seed=seed)
    model.fit(data['train'],validation_blocks=data['validation'],epochs=epochs,lr=.005,batch_size=4,patience=max(3,epochs),device=device)
    return model


def fit_gp_c(data,epochs=8,seed=42,device="cpu"):
    model=ResidualFlowGP(hidden_dim=12,kernel_rank=6,flow_layers=2,inducing_per_variable=3,
                         elbo_samples=8,seed=seed)
    model.fit(data['train'],validation_blocks=data['validation'],epochs=epochs,lr=.004,batch_size=3,patience=max(3,epochs),device=device)
    return model


def representative_tpp_sequence(sequences,min_events=8):
    candidates=[s for s in sequences if len(s.times)>=min_events]
    return sorted(candidates,key=lambda s:(abs(len(s.times)-15),s.patient_id))[0]


def representative_gp_block(blocks,min_panels=5):
    candidates=[]
    for b in blocks:
        if len(np.unique(b.ages))>=min_panels: candidates.append(b)
    block=sorted(candidates,key=lambda b:(abs(len(np.unique(b.ages))-10),b.patient_id))[0]
    return split_forecast_block(block,.6)
