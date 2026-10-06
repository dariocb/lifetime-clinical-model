"""Scalar monotone sinh-arcsinh flows for real-valued GP measurements/residuals."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .history import NeuralInputs


class ObservationFlow(nn.Module):
    def __init__(self, input_dim, hidden_dim=24, layers=2,feature_names=None):
        super().__init__()
        self.layers = layers
        inputs=NeuralInputs(input_dim,hidden_dim,feature_names)
        self.net = nn.Sequential(inputs,nn.Linear(inputs.output_dim, hidden_dim), nn.Tanh(),
                                 nn.Linear(hidden_dim, 4*layers)).double()
        nn.init.normal_(self.net[-1].weight, std=.002)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, value, context, inverse=False):
        raw = self.net(context).reshape(len(context), self.layers, 4)
        logdet = torch.zeros_like(value)
        for j in (range(self.layers-1, -1, -1) if inverse else range(self.layers)):
            shift = raw[:, j, 0]
            scale = torch.exp(.8*torch.tanh(raw[:, j, 1]))
            skew = .8*torch.tanh(raw[:, j, 2])
            tail = torch.exp(.3*torch.tanh(raw[:, j, 3]))
            if inverse:
                a = torch.asinh(value)/tail-skew
                logcosh = a.abs()+F.softplus(-2*a.abs())-np.log(2.)
                logdet = logdet+torch.log(scale)-torch.log(tail)+logcosh-.5*torch.log1p(value.square())
                value = scale*torch.sinh(a)+shift
            else:
                a = (value-shift)/scale
                b = tail*(torch.asinh(a)+skew)
                logcosh = b.abs()+F.softplus(-2*b.abs())-np.log(2.)
                logdet = logdet+torch.log(tail)-torch.log(scale)+logcosh-.5*torch.log1p(a.square())
                value = torch.sinh(b)
        return value, logdet

    def log_prob(self, residual, context):
        z, logdet = self(residual, context)
        return -.5*z.square()-.5*np.log(2*np.pi)+logdet
