"""Model C: full feature bijection; only shared coordinates enter timing."""
import numpy as np
import torch
from torch import nn
from .joint_flow import JointFlowTPP
from ...components.history import NeuralInputs

class SharedPrivateFlowTPP(JointFlowTPP):
    model_name="tpp_c"
    def __init__(self,shared_dim=1,use_private=False,**kwargs):
        if int(shared_dim)!=shared_dim:raise ValueError("shared_dim must be an integer")
        self.shared_dim=int(shared_dim)
        self.use_private=bool(use_private)
        if self.shared_dim<1:raise ValueError("shared_dim must be positive")
        super().__init__(**kwargs)
        if self.shared_dim>=len(self.feature_indices):
            raise ValueError("Model C needs both shared and private continuous coordinates")
    def _initialize(self,input_dim):
        super()._initialize(input_dim)
        dimension=input_dim if self.use_private else input_dim-len(self.feature_indices)+self.shared_dim
        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            names=self._timing_names() if self.input_mode=="dynamic" else None
            encoded=NeuralInputs(dimension+self.hidden_dim,self.hidden_dim,
                None if names is None else names+[f"H_{i}" for i in range(self.hidden_dim)])
            self.net=nn.Sequential(encoded,nn.Linear(encoded.output_dim,self.hidden_dim),nn.Tanh(),
                                   nn.Linear(self.hidden_dim,self.output_dim)).double()
            nn.init.normal_(self.net[-1].weight,std=.01);nn.init.zeros_(self.net[-1].bias)
    def split_latents(self,inputs,history=None):
        z=self.encode_features(inputs,history)
        return z[...,:self.shared_dim],z[...,self.shared_dim:]
    def _latent_timing(self,x,z):
        rest=[i for i in range(self.input_dim) if i not in self.feature_indices]
        return torch.cat([z if self.use_private else z[...,:self.shared_dim],x[...,rest]],dim=-1)
    def _timing_features(self,inputs):
        x=self._tensor(inputs);shared,_=self.split_latents(x)
        rest=[i for i in range(self.input_dim) if i not in self.feature_indices]
        if self.use_private:shared=self.encode_features(x)
        return torch.cat([shared,x[...,rest]],dim=-1)
