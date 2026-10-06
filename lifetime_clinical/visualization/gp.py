from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from ..data.preprocessing import VARIABLES


def plot_additive_contributions(model, conditioning_block, heldout_block, age_grid, cutoff_age):
    """GP-C forecast mean and variance contributions in clinical units."""
    fig, axes = plt.subplots(len(VARIABLES), 2, figsize=(13, 10), sharex=True, dpi=160)
    for vid, variable in enumerate(VARIABLES):
        ids = np.full(len(age_grid), vid, dtype=int)
        prediction = model.forecast(conditioning_block, np.asarray(age_grid)/100., ids)
        # Offset belongs to physiology; the additive residual has no offset.
        mean_f = prediction.offsets + prediction.scales*prediction.latent_mean_z
        var_f = prediction.scales**2*np.diag(prediction.latent_covariance_z)
        mean_y, var_y = prediction.mean, prediction.variance
        var_eps = np.maximum(var_y-var_f, 0.)
        left, right = axes[vid]
        left.plot(age_grid, mean_f, color="tab:blue", linewidth=2.4, label=r"Physiology mean $E[f]$")
        left.plot(age_grid, mean_y, color="black", linewidth=2.4, label=r"Observation mean $E[y]$")
        left.fill_between(age_grid, mean_f, mean_y, color="tab:orange", alpha=.35,
                          label=r"Signed residual shift $E[\epsilon]=E[y]-E[f]$")
        ages, values = _raw_observations(heldout_block, vid)
        left.scatter(ages, values, color="red", marker="x", s=30, label="held-out observations")
        left.set_xlim(age_grid[0], age_grid[-1])
        right.stackplot(age_grid, var_f, var_eps, colors=["tab:blue", "tab:orange"], alpha=.65,
                        labels=[r"Physiology variance $\mathrm{Var}(f)$", r"Residual variance $\mathrm{Var}(\epsilon)$"])
        right.plot(age_grid, var_y, color="black", linewidth=2.4, label=r"Total variance $\mathrm{Var}(y)$")
        name, unit = {"systolic_bp": ("Systolic BP", "mmHg"), "glucose": ("Glucose", "mg/dL"),
                      "creatinine": ("Creatinine", "mg/dL")}[variable]
        left.set(title=f"{name}: additive mean contributions", ylabel=f"{name} ({unit})")
        right.set(title=f"{name}: additive uncertainty contributions", ylabel=f"Variance ({unit})²")
        for ax in (left, right):
            ax.axvline(cutoff_age, color="tab:red", linestyle="--", linewidth=1.3)
            ax.grid(alpha=.15)
    handles, labels = [], []
    for ax in axes[0]:
        h, lab = ax.get_legend_handles_labels(); handles.extend(h); labels.extend(lab)
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, frameon=False)
    for ax in axes[-1]: ax.set_xlabel("Patient age (years)")
    fig.tight_layout(rect=(0, 0, 1, .91))
    return axes


def _raw_observations(block, variable_id):
    mask=block.variables==variable_id
    return block.ages[mask]*100, block.raw_y[mask]


def _density_image(samples, y_grid=None, bins=90):
    samples=np.asarray(samples,float)
    if samples.ndim!=2: raise ValueError("samples must be [n_samples,n_times]")
    lo,hi=np.quantile(samples,[.005,.995])
    if y_grid is None: y_grid=np.linspace(lo,hi,bins)
    edges=np.linspace(y_grid[0],y_grid[-1],len(y_grid)+1)
    density=np.zeros((len(y_grid),samples.shape[1]))
    for j in range(samples.shape[1]):
        hist,_=np.histogram(samples[:,j],bins=edges,density=True)
        density[:,j]=gaussian_filter1d(hist.astype(float),1.2)
    return y_grid,density


def plot_gp_feature(model, conditioning_block, heldout_block, variable_id, age_grid, ax=None, title=None):
    """Classical GP visualization for GP-A: mean and Gaussian uncertainty."""
    ax=ax or plt.subplots(figsize=(9.5,3.7))[1]
    ids=np.full(len(age_grid),variable_id,dtype=int)
    forecast=model.forecast(conditioning_block,np.asarray(age_grid)/100.,ids,retrospective=True)
    mean=forecast.mean; sd=np.sqrt(np.maximum(forecast.variance,0))
    ax.fill_between(age_grid,mean-1.96*sd,mean+1.96*sd,alpha=.16,label="95% predictive interval")
    ax.fill_between(age_grid,mean-sd,mean+sd,alpha=.22,label="±1 predictive SD")
    ax.plot(age_grid,mean,label="predictive mean")
    x,y=_raw_observations(conditioning_block,variable_id); ax.scatter(x,y,s=25,label="conditioning observations")
    xh,yh=_raw_observations(heldout_block,variable_id); ax.scatter(xh,yh,s=28,marker="x",label="held-out observations")
    ax.set_xlabel("age (years)"); ax.set_ylabel(VARIABLES[variable_id].replace("_"," "))
    ax.set_title(title or f"GP-A — {VARIABLES[variable_id]}")
    ax.grid(alpha=.18); ax.legend(frameon=False,ncol=2,fontsize=8)
    return ax


def plot_sampled_predictive_density(model, conditioning_block, heldout_block, variable_id, age_grid,
                                    ax=None,n_samples=1600,seed=42,title=None,bins=90,cmap=None):
    """Predictive density heatmap from samples; used for non-Gaussian GP-B/GP-C."""
    ax=ax or plt.subplots(figsize=(9.5,3.8))[1]
    ids=np.full(len(age_grid),variable_id,dtype=int)
    forecast=model.forecast(conditioning_block,np.asarray(age_grid)/100.,ids,retrospective=True)
    samples=forecast.sample(n_samples,seed)
    y,dens=_density_image(samples,bins=bins)
    ax.imshow(dens,origin="lower",aspect="auto",extent=[age_grid[0],age_grid[-1],y[0],y[-1]],
              alpha=.72,cmap=cmap,interpolation="bilinear")
    q=np.quantile(samples,[.1,.5,.9],axis=0)
    ax.plot(age_grid,q[1],linewidth=2.4,label="sample median")
    ax.plot(age_grid,q[0],linewidth=1.8,linestyle="-",label="10/90% quantiles")
    ax.plot(age_grid,q[2],linewidth=1.8,linestyle="-")
    x,yobs=_raw_observations(conditioning_block,variable_id); ax.scatter(x,yobs,s=23,label="conditioning observations")
    xh,yh=_raw_observations(heldout_block,variable_id); ax.scatter(xh,yh,s=28,marker="x",label="held-out observations")
    ax.set_xlabel("age (years)"); ax.set_ylabel(VARIABLES[variable_id].replace("_"," "))
    ax.set_title(title or f"Sampled predictive density — {VARIABLES[variable_id]}")
    ax.legend(frameon=False,ncol=2,fontsize=8); ax.grid(alpha=.1)
    return ax


def plot_residual_decomposition(model, conditioning_block, heldout_block, variable_id, age_grid,
                                axes=None,n_samples=1600,seed=42,title=None,bins=90,cmap=None):
    """GP-C: full observation density, latent physiology, and residual distribution."""
    if axes is None:
        _,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
    ids=np.full(len(age_grid),variable_id,dtype=int)
    forecast=model.forecast(conditioning_block,np.asarray(age_grid)/100.,ids,retrospective=True)
    obs=forecast.sample(n_samples,seed)
    latent=forecast.latent_sample(n_samples,seed+1,raw_units=True)
    residual=model.sample_residuals(conditioning_block,np.asarray(age_grid)/100.,ids,n_samples=n_samples,seed=seed+2)
    for ax,samples,label in [(axes[0],obs,"combined observation y=f+ε"),(axes[2],residual,"flow residual ε")]:
        yg,dens=_density_image(samples,bins=bins)
        ax.imshow(dens,origin="lower",aspect="auto",extent=[age_grid[0],age_grid[-1],yg[0],yg[-1]],
                  alpha=.72,cmap=cmap,interpolation="bilinear")
        q=np.quantile(samples,[.1,.5,.9],axis=0); ax.plot(age_grid,q[1],linewidth=2.4,label="median")
        ax.plot(age_grid,q[0],linestyle="-",linewidth=1.8,label="10/90% quantiles")
        ax.plot(age_grid,q[2],linestyle="-",linewidth=1.8)
        ax.set_ylabel(label)
    q=np.quantile(latent,[.1,.5,.9],axis=0)
    axes[1].fill_between(age_grid,q[0],q[2],alpha=.2,label="80% latent physiology")
    axes[1].plot(age_grid,q[1],label="latent median")
    axes[1].set_ylabel("latent physiology f")
    x,y=_raw_observations(conditioning_block,variable_id); axes[0].scatter(x,y,s=22,label="conditioning observations")
    xh,yh=_raw_observations(heldout_block,variable_id); axes[0].scatter(xh,yh,s=26,marker="x",label="held-out observations")
    axes[-1].set_xlabel("age (years)")
    for ax in axes: ax.grid(alpha=.12); ax.legend(frameon=False,fontsize=8,loc="best")
    axes[0].set_title(title or f"GP-C decomposition — {VARIABLES[variable_id]}")
    return axes
