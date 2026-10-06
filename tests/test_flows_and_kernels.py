from pathlib import Path
import numpy as np
import torch
from lifetime_clinical.tutorials.experiments import load_toy, gp_data
from lifetime_clinical.models.longitudinal.deep_kernel_gp import DeepKernelGP

ROOT=Path(__file__).resolve().parents[1]


def test_learned_kernel_is_psd_numerically():
    dataset,splits,pre=load_toy(ROOT/'data'/'toy')
    blocks=gp_data(dataset,splits,pre,.25)
    model=DeepKernelGP(hidden_dim=8,kernel_rank=4)
    model._check([blocks['train'][0]])
    q=model.build_gp_inputs(blocks['train'][0])
    K=model.kernel(q).detach().numpy()
    eig=np.linalg.eigvalsh((K+K.T)/2)
    assert eig.min()>-1e-8
