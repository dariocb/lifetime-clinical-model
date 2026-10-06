"""Shared deep kernel and patient-level training for the three GP formulations."""
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ...components.history import CausalHistoryEncoder,NeuralInputs,clinical_feature_names
from ...components.gp_linalg import cholesky, spd_solve, triangular_solve, normal_log_prob


class NeuralGP(nn.Module):
    model_name = "gp_1"

    def __init__(self, input_dim=None, history_dim=None, hidden_dim=16, kernel_rank=8,
                 n_variables=3, seed=42, jitter=1e-6,kernel_kind="learned"):
        super().__init__()
        if (hidden_dim < 1 or kernel_rank < 1 or n_variables < 1
                or not np.isfinite(jitter) or jitter <= 0):
            raise ValueError("Positive architecture sizes and jitter required")
        if (input_dim is None) != (history_dim is None):
            raise ValueError("Specify both input_dim and history_dim or infer both during fit")
        self.input_dim, self.history_dim = input_dim, history_dim
        self.hidden_dim, self.kernel_rank, self.n_variables = hidden_dim, kernel_rank, n_variables
        self.seed, self.jitter = seed, jitter
        if kernel_kind not in ("learned","rbf","matern32"):raise ValueError("Unknown kernel kind")
        self.kernel_kind=kernel_kind
        self.feature_net = None
        self.fit_history = []
        if input_dim is not None:
            self._initialize(input_dim, history_dim)

    def _initialize(self, input_dim, history_dim):
        if input_dim < 1 or history_dim < 1:
            raise ValueError("Positive input and history dimensions required")
        self.input_dim, self.history_dim = int(input_dim), int(history_dim)
        self.query_dim = self.input_dim+self.hidden_dim+1+self.n_variables
        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            self.history_encoder = CausalHistoryEncoder(self.history_dim,self.hidden_dim,
                heads=2 if self.hidden_dim%2==0 else 1,
                feature_names=clinical_feature_names("history",self.history_dim))
            static_names=clinical_feature_names("static",input_dim) or [f"static_{i}" for i in range(input_dim)]
            self.query_feature_names=static_names+[f"H_{i}" for i in range(self.hidden_dim)]+["query_age_scaled"]+[f"output={i}" for i in range(self.n_variables)]
            feature_inputs=NeuralInputs(self.query_dim,self.hidden_dim,self.query_feature_names)
            mean_inputs=NeuralInputs(self.query_dim,self.hidden_dim,self.query_feature_names)
            self.feature_net = nn.Sequential(feature_inputs,nn.Linear(feature_inputs.output_dim, self.hidden_dim), nn.Tanh(),
                                             nn.Linear(self.hidden_dim, self.kernel_rank)).double()
            self.mean_net = nn.Sequential(mean_inputs,nn.Linear(mean_inputs.output_dim, self.hidden_dim), nn.Tanh(),
                                          nn.Linear(self.hidden_dim, 1)).double()
            nn.init.zeros_(self.mean_net[-1].weight)
            nn.init.zeros_(self.mean_net[-1].bias)
            self.raw_noise = nn.Parameter(torch.full((self.n_variables,), -1., dtype=torch.double))
            self.raw_length = nn.Parameter(torch.tensor(-2., dtype=torch.double))
            self.raw_amplitude = nn.Parameter(torch.tensor(0., dtype=torch.double))
            self.variable_factor = nn.Parameter(torch.eye(self.n_variables, dtype=torch.double))
            self._extra_initialize()
            if self.kernel_kind!="learned":
                for p in self.feature_net.parameters():p.requires_grad_(False)

    def _extra_initialize(self):
        pass

    def _tensor(self, value):
        parameter = next(self.parameters())
        return torch.as_tensor(value, dtype=parameter.dtype, device=parameter.device)

    def _check(self, blocks, allow_empty=False):
        if not blocks:
            raise ValueError("No GP patient blocks")
        for block in blocks:
            block.validate()
            if not allow_empty and not len(block.y):
                raise ValueError("Training needs observed targets")
            if len(block.scales) != self.n_variables:
                raise ValueError("Variable scaling metadata does not match model")
        if self.feature_net is None:
            self._initialize(len(blocks[0].static), blocks[0].history.shape[1])
        for block in blocks:
            if len(block.static) != self.input_dim or block.history.shape[1] != self.history_dim:
                raise ValueError("Inconsistent fitted input schema")

    def build_gp_inputs(self, block, ages=None, variables=None):
        self._check([block], allow_empty=True)
        ages = block.ages if ages is None else np.atleast_1d(np.asarray(ages, dtype=float))
        variables = block.variables if variables is None else np.atleast_1d(np.asarray(variables))
        if (np.shape(ages) != np.shape(variables) or not np.isfinite(ages).all()
                or not np.isfinite(variables).all() or np.any(variables < 0)
                or np.any(variables >= self.n_variables)
                or not np.equal(variables, np.floor(variables)).all()):
            raise ValueError("Queries require finite age/100 and valid variable IDs of equal shape")
        state = self.history_encoder(block.history)
        baseline = torch.cat([self._tensor(block.static), state])
        variable_tensor = torch.as_tensor(variables, dtype=torch.long, device=state.device)
        return torch.cat([baseline.expand(len(ages), -1), self._tensor(ages)[:, None],
                          F.one_hot(variable_tensor, self.n_variables).to(state.dtype)], dim=1)

    def kernel(self, queries, other=None):
        """Feature Gram plus RBF-age × PSD variable-coregionalization kernel."""
        other = queries if other is None else other
        a, b = self.feature_net(queries), self.feature_net(other)
        age_column = -self.n_variables-1
        difference = queries[:, age_column, None]-other[:, age_column][None, :]
        length = F.softplus(self.raw_length)+.01
        temporal = torch.exp(-.5*(difference/length).square())
        ids_a = queries[:, -self.n_variables:]
        ids_b = other[:, -self.n_variables:]
        covariance = self.variable_factor @ self.variable_factor.T
        if self.kernel_kind=="matern32":
            r=np.sqrt(3)*torch.abs(difference)/length;temporal=(1+r)*torch.exp(-r)
        gram=a@b.T/self.kernel_rank if self.kernel_kind=="learned" else torch.zeros_like(temporal)
        return gram + (F.softplus(self.raw_amplitude)+.01)*temporal*(ids_a @ covariance @ ids_b.T)

    def mean(self, queries):
        return self.mean_net(queries)[:, 0]

    def noise_variance(self, queries):
        noise_std = F.softplus(self.raw_noise)+.03
        return queries[:, -self.n_variables:] @ noise_std.square()

    def kernel_diagonal(self, queries):
        features = self.feature_net(queries)
        covariance = self.variable_factor @ self.variable_factor.T
        ids = queries[:, -self.n_variables:]
        gram=features.square().sum(-1)/self.kernel_rank if self.kernel_kind=="learned" else torch.zeros(len(queries),dtype=queries.dtype,device=queries.device)
        return gram + (F.softplus(self.raw_amplitude)+.01)*(ids @ covariance.diag())

    def _gaussian_posterior(self, block, query, values):
        mean = self.mean(query)
        covariance = self.kernel(query)
        if not len(block.y):
            return mean, covariance
        training = self.build_gp_inputs(block)
        c = self.kernel(training)+torch.diag(self.noise_variance(training))
        factor = cholesky(c, self.jitter)
        cross = self.kernel(query, training)
        mean = mean+cross @ spd_solve(factor, values-self.mean(training))
        projected = triangular_solve(factor, cross.T)
        covariance = covariance-projected.T @ projected
        return mean, (covariance+covariance.T)*.5

    def log_likelihood(self, blocks, observation_windows=None):
        self._check(blocks)
        return torch.stack([self._block_log_likelihood(block) for block in blocks])

    def nll(self, blocks):
        return -self.log_likelihood(blocks).mean()

    def score(self, blocks):
        self.eval()
        with torch.no_grad():
            return float(self.nll(blocks))

    def fit(self, blocks, observation_windows=None, validation_blocks=None, epochs=50,
            lr=.005, batch_size=8, patience=15, device="cpu"):
        if epochs < 1 or batch_size < 1 or patience < 1 or not np.isfinite(lr) or lr <= 0:
            raise ValueError("Positive optimization settings required")
        self._check(blocks)
        if validation_blocks:
            self._check(validation_blocks)
            if set(b.patient_id for b in blocks) & set(b.patient_id for b in validation_blocks):
                raise ValueError("Train/validation patients must be disjoint")
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        rng = np.random.default_rng(self.seed)
        monitor = validation_blocks or blocks
        best, state, stale = self.score(monitor), deepcopy(self.state_dict()), 0
        self.fit_history = [dict(epoch=0, train_objective=self.score(blocks), monitor_objective=best)]
        for epoch in range(1, epochs+1):
            self.train()
            order = rng.permutation(len(blocks))
            for start in range(0, len(order), batch_size):
                selected = [blocks[i] for i in order[start:start+batch_size]]
                optimizer.zero_grad()
                loss = self.nll(selected)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite GP training objective")
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 20., error_if_nonfinite=True)
                optimizer.step()
            train, validation = self.score(blocks), self.score(monitor)
            self.fit_history.append(dict(epoch=epoch, train_objective=train, monitor_objective=validation))
            if validation < best-1e-8:
                best, state, stale = validation, deepcopy(self.state_dict()), 0
            else:
                stale += 1
            if stale >= patience:
                break
        self.load_state_dict(state)
        self.eval()
        return self.fit_history

    def predict_distribution(self, history, query):
        if history is None or not isinstance(query, dict):
            raise ValueError("Provide an observed GPBlock and query ages/variables.")
        return self.forecast(history, query["ages"], query["variables"])

    def sample(self, history, query, n_samples=100, seed=None):
        return self.predict_distribution(history, query).sample(n_samples, seed)

    def interpolate(self, block, ages, variables):
        return self.forecast(block, ages, variables, retrospective=True)
