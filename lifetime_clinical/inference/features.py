"""Metropolis-Hastings conditioning on feature and timing evidence."""
import numpy as np
import torch

def infer_features(model,inputs,missing_indices,history=(),next_iat=None,censor_days=None,
                   n_samples=500,burn_in=300,proposal_scale=.25,seed=42):
    x=np.asarray(inputs,dtype=float).copy();indices=list(missing_indices)
    if x.shape!=(model.input_dim,) or not np.isfinite(x).all():raise ValueError("Provide a finite initialization vector")
    if not indices or len(set(indices))!=len(indices) or not set(indices)<=set(model.feature_indices):
        raise ValueError("Infer only unique selected continuous feature coordinates")
    if n_samples<1 or burn_in<0 or not np.isfinite(proposal_scale) or proposal_scale<=0:raise ValueError("Invalid sampler settings")
    if next_iat is not None and censor_days is not None:raise ValueError("Use observed IAT or censored evidence, not both")
    if next_iat is not None and (not np.isfinite(next_iat) or next_iat<=0):raise ValueError("Positive next_iat required")
    if censor_days is not None and (not np.isfinite(censor_days) or censor_days<0):raise ValueError("Nonnegative censor_days required")
    def score(row):
        with torch.no_grad():
            value=model.feature_log_prob(row,history if model.input_mode=="dynamic" else None)
            if model.input_mode=="static":
                for j,gap in enumerate(history):
                    value=value+model.iat_log_prob(gap,row,history[:j])
            if next_iat is not None:value=value+model.iat_log_prob(next_iat,row,history)
            if censor_days is not None:value=value+model._log_terms(censor_days,model.build_context(row,history))[1]
            return float(value)
    rng=np.random.default_rng(seed);current=score(x);samples=[];accepted=0
    if not np.isfinite(current):raise FloatingPointError("Initial posterior log density is nonfinite")
    for iteration in range(burn_in+n_samples):
        proposal=x.copy();proposal[indices]+=rng.normal(scale=proposal_scale,size=len(indices))
        candidate=score(proposal)
        if np.log(rng.uniform())<candidate-current:x,current=proposal,candidate;accepted+=1
        if iteration>=burn_in:samples.append(x.copy())
    return dict(samples=np.asarray(samples),acceptance_rate=accepted/(burn_in+n_samples),
                missing_indices=indices,method="random_walk_metropolis",seed=seed)
