"""Shared CPU/GPU optimization and schema validation for neural TPP baselines."""
from copy import deepcopy
import numpy as np
import torch
from torch import nn

class NeuralTPP(nn.Module):
    input_field = "static"

    def __init__(self, input_dim=None, hidden_dim=32, n_marks=1, seed=42, time_scale=365.2425):
        super().__init__()
        if hidden_dim < 1 or n_marks not in (1,3) or time_scale <= 0 or not np.isfinite(time_scale):
            raise ValueError("hidden_dim > 0, n_marks 1/3, positive finite time_scale required")
        self.input_dim,self.hidden_dim,self.n_marks = input_dim,hidden_dim,n_marks
        self.seed,self.time_scale = seed,float(time_scale)
        self.net = None
        self.fit_history = []
        if input_dim is not None: self._initialize(input_dim)

    def _initialize(self, input_dim):
        if input_dim < 1: raise ValueError("input_dim must be positive")
        self.input_dim = int(input_dim)
        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            self.net = nn.Sequential(nn.Linear(input_dim,self.hidden_dim),nn.Tanh(),
                                     nn.Linear(self.hidden_dim,self.output_dim)).double()
            nn.init.normal_(self.net[-1].weight,std=0.01)
            nn.init.zeros_(self.net[-1].bias)

    def _tensor(self, values):
        if self.net is None: raise RuntimeError("Initialize input_dim or call fit first")
        parameter = next(self.parameters())
        return torch.as_tensor(values,dtype=parameter.dtype,device=parameter.device)

    def _check(self,sequences):
        if not sequences: raise ValueError("No patient sequences")
        dim = getattr(sequences[0],self.input_field).shape[-1]
        for seq in sequences:
            seq.validate()
            if getattr(seq,self.input_field).shape[-1] != dim: raise ValueError("Inconsistent input dimensions")
        if self.net is None: self._initialize(dim)
        if dim != self.input_dim: raise ValueError("Inputs do not match the fitted feature schema")

    def log_likelihood(self, sequences, observation_windows=None):
        """Per-patient differentiable log likelihood, including terminal exposure."""
        self._check(sequences)
        return torch.stack([self._sequence_log_likelihood(seq) for seq in sequences])

    def nll(self, sequences):
        return -self.log_likelihood(sequences).mean()

    def score(self,sequences):
        self.eval()
        with torch.no_grad(): return float(self.nll(sequences))

    def fit(self,sequences,observation_windows=None,validation_sequences=None,epochs=100,
            lr=0.01,batch_size=32,patience=20,device="cpu"):
        if epochs < 1 or batch_size < 1 or patience < 1 or not np.isfinite(lr) or lr <= 0:
            raise ValueError("Positive optimization settings required")
        self._check(sequences)
        if validation_sequences:
            self._check(validation_sequences)
            if set(s.patient_id for s in sequences) & set(s.patient_id for s in validation_sequences):
                raise ValueError("Train/validation patients must be disjoint")
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(),lr=lr)
        rng = np.random.default_rng(self.seed)
        monitor = validation_sequences or sequences
        best,best_state,stale = self.score(monitor),deepcopy(self.state_dict()),0
        self.fit_history = [{"epoch":0,"train_nll":self.score(sequences),"monitor_nll":best}]
        for epoch in range(1,epochs+1):
            self.train()
            order = rng.permutation(len(sequences))
            for start in range(0,len(order),batch_size):
                batch = [sequences[i] for i in order[start:start+batch_size]]
                optimizer.zero_grad();loss=self.nll(batch)
                if not torch.isfinite(loss): raise FloatingPointError("Nonfinite TPP likelihood")
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(),10.,error_if_nonfinite=True)
                optimizer.step()
            train_score,monitor_score=self.score(sequences),self.score(monitor)
            self.fit_history.append({"epoch":epoch,"train_nll":train_score,"monitor_nll":monitor_score})
            if monitor_score < best-1e-8:
                best,best_state,stale=monitor_score,deepcopy(self.state_dict()),0
            else: stale+=1
            if stale >= patience: break
        self.load_state_dict(best_state);self.eval()
        return self.fit_history

    @staticmethod
    def _horizons(horizons):
        h=np.atleast_1d(np.asarray(horizons,dtype=float))
        if not np.isfinite(h).all() or (h < 0).any(): raise ValueError("Horizons must be finite nonnegative days")
        return h
