from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from ..data.preprocessing import VARIABLES
from ..data.synthetic import EVENT_TYPES
from ..data.tutorial import physiology_feature_indices, context_feature_names

YEAR_DAYS = 365.2425


def _dynamic_history(seq, j):
    gaps = np.diff(np.r_[0., seq.times])
    return {"features": seq.contexts[:j], "iats": gaps[:j]}


def _origin_age(seq, preprocessor=None):
    if seq.contexts.ndim == 2 and preprocessor is not None:
        names = context_feature_names(preprocessor)
        if "current_age_scaled" in names:
            return float(seq.contexts[0, names.index("current_age_scaled")] * 100)
    return float(seq.static[0] * 100)


def plot_tpp_timeline_fit(model, seq, preprocessor=None, ax=None, grid_per_interval=50, title=None):
    """Total intensity, with each segment coloured by its starting event's mark."""
    ax = ax or plt.subplots(figsize=(11, 3.5))[1]
    origin_age = _origin_age(seq, preprocessor)
    gaps = np.diff(np.r_[0., seq.times])
    starts = np.r_[0., seq.times]
    ends = np.r_[seq.times, seq.end]
    marked = getattr(model, "n_marks", 1) == 3
    event_colours = ("tab:blue", "tab:orange", "tab:green")
    for j, (start, end) in enumerate(zip(starts, ends)):
        duration = max(float(end-start), 1e-6)
        horizons = np.linspace(0, duration, grid_per_interval)
        if getattr(model, "input_mode", "static") == "dynamic":
            inputs = seq.contexts[j]
            history = _dynamic_history(seq, j)
        else:
            inputs = seq.static
            history = list(gaps[:j])
        pred = model.next_event_distribution(inputs, horizons, history=history)
        intensity = pred["hazard"] * YEAR_DAYS  # events/year for readability
        ages = origin_age + (start+horizons)/YEAR_DAYS
        colour = "grey" if j == 0 else (event_colours[int(seq.marks[j-1])] if marked else "red")
        ax.plot(ages, intensity, color=colour, linewidth=1.35)
    event_ages = origin_age + seq.times/YEAR_DAYS
    if marked:
        for mark, (kind, colour, marker) in enumerate(zip(
            EVENT_TYPES, event_colours, ("o", "^", "s")
        )):
            ages = event_ages[np.asarray(seq.marks) == mark]
            if len(ages):
                ax.scatter(ages, np.zeros_like(ages), color=colour, marker=marker,
                           s=35, zorder=3, label=f"observed {kind}")
    else:
        ax.scatter(event_ages, np.zeros_like(event_ages), color="red", marker="x",
                   s=35, zorder=3, label="observed events")
    ax.axvline(origin_age+seq.end/YEAR_DAYS, linestyle="--", linewidth=1, alpha=.7, label="right censoring")
    ax.set_xlabel(r"Patient age ($t$; years)")
    ax.set_ylabel(r"Conditional intensity ($\lambda$; events per year)")
    ax.set_title(title or f"{getattr(model,'model_name','TPP')}: fitted conditional intensity through the observation window")
    ax.legend(frameon=False)
    ax.grid(alpha=.18)
    return ax


def plot_next_event_forecast(model, seq, preprocessor=None, horizon_days=730, axes=None, title=None):
    """Forecast from censoring, conditional on no event since the last visit."""
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    h = np.linspace(1e-3, horizon_days, 320)
    gaps = np.diff(np.r_[0., seq.times])
    if getattr(model, "input_mode", "static") == "dynamic":
        inputs = seq.contexts[-1]
        history = _dynamic_history(seq, len(seq.contexts)-1)
    else:
        inputs = seq.static
        history = list(gaps)
    elapsed = float(seq.end - (seq.times[-1] if len(seq.times) else 0.))
    pred = model.next_event_distribution(inputs, h, history=history, elapsed_days=elapsed)
    x = h/YEAR_DAYS
    axes[0].plot(x, pred["density"]*YEAR_DAYS, color="black", linewidth=2, label="total mixture")
    axes[0].set_ylabel(r"Density ($f(\tau)$; per year)")
    axes[0].set_title(r"Next-IAT mixture density ($f$)")
    axes[1].plot(x, pred["survival"], color="black", linewidth=2, label="total mixture")
    axes[1].set_ylim(-.02,1.02)
    axes[1].set_ylabel(r"Survival ($S(\tau)$; probability)")
    axes[1].set_title(r"Survival ($S$)")
    axes[2].plot(x, pred["hazard"]*YEAR_DAYS, color="black", linewidth=2, label="total intensity")
    axes[2].set_ylabel(r"Hazard ($h(\tau)$; events per year)")
    axes[2].set_title(r"Hazard ($h$) / intensity ($\lambda$)")
    if getattr(model, "n_marks", 1) == 3:
        for k, (kind, colour) in enumerate(zip(EVENT_TYPES, ("tab:blue", "tab:orange", "tab:green"))):
            name=kind.replace(" visit", "")
            axes[0].plot(x, pred["marked_density"][:,k]*YEAR_DAYS, "--", color=colour,
                         label=f"{name} contribution")
            axes[1].plot(x, pred["mark_conditional_survival"][:,k], "--", color=colour,
                         label=f"{name} $S_k$")
            axes[2].plot(x, pred["marked_hazard"][:,k]*YEAR_DAYS, "--", color=colour,
                         label=rf"{name} $\lambda_k$")
    for ax in axes:
        ax.set_xlabel(r"Time after censoring ($t-t_C$; years)"); ax.grid(alpha=.18)
        ax.legend(frameon=False, fontsize=7)
    if title:
        axes[0].figure.suptitle(title)
    axes[0].figure.tight_layout()
    return axes


def feature_flow_predictions(model, seq, preprocessor, n_samples=600, seed=42):
    """One-step feature-state predictions before each observed dynamic snapshot."""
    indices = physiology_feature_indices(preprocessor)
    rng = np.random.default_rng(seed)
    rows=[]
    for j, observed in enumerate(seq.contexts):
        past = _dynamic_history(seq, j)
        samples = model.sample_patients(n_samples, conditioning_inputs=observed,
                                        history=past, seed=int(rng.integers(2**31)))
        age = float(observed[context_feature_names(preprocessor).index("current_age_scaled")]*100)
        for k, variable in enumerate(VARIABLES):
            mean, std = preprocessor.measurement_scales[variable]
            raw = mean + std*samples[:, indices[k]]
            truth = mean + std*observed[indices[k]]
            rows.append((j, age, variable, truth,
                         np.quantile(raw,.1), np.quantile(raw,.5), np.quantile(raw,.9)))
    return rows


def plot_feature_flow_fit(model, seq, preprocessor, dataset=None, patient_id=None, axes=None,
                          n_samples=600, seed=42, title=None):
    rows = feature_flow_predictions(model, seq, preprocessor, n_samples=n_samples, seed=seed)
    if axes is None:
        _, axes = plt.subplots(len(VARIABLES), 1, figsize=(10, 8), sharex=True)
    for d, (ax, variable) in enumerate(zip(axes, VARIABLES), start=1):
        subset=[r for r in rows if r[2]==variable]
        ages=np.array([r[1] for r in subset]); truth=np.array([r[3] for r in subset])
        q10=np.array([r[4] for r in subset]); med=np.array([r[5] for r in subset]); q90=np.array([r[6] for r in subset])
        ax.fill_between(ages,q10,q90,alpha=.18,label="80% one-step predictive interval")
        ax.plot(ages,med,label="predictive median")
        ax.plot(ages,truth,"o-",markersize=3,alpha=.7,label="state used by TPP")
        if dataset is not None and patient_id is not None:
            m=dataset.measurements[(dataset.measurements.patient_id==patient_id)&(dataset.measurements.variable==variable)]
            ax.scatter(m.measurement_age,m.value,s=18,marker="x",label="raw measurements")
        name, unit = {"systolic_bp": ("Systolic BP", "mmHg"),
                      "glucose": ("Glucose", "mg/dL"),
                      "creatinine": ("Creatinine", "mg/dL")}[variable]
        ax.set_ylabel(f"{name} ($x_{d}$; {unit})")
        ax.grid(alpha=.16)
    axes[-1].set_xlabel(r"Patient age ($t$; years)")
    handles, labels=axes[0].get_legend_handles_labels(); axes[0].legend(handles,labels,frameon=False,ncol=2,fontsize=8)
    if title: axes[0].figure.suptitle(title)
    return axes


def forecast_next_feature_snapshot(model, seq, preprocessor, n_samples=1500, seed=42):
    j=len(seq.contexts)-1
    past=_dynamic_history(seq,j)
    current=seq.contexts[-1]
    rng=np.random.default_rng(seed)
    elapsed=float(seq.end-(seq.times[-1] if len(seq.times) else 0.))
    waits=model.sample_next_iat(current,past,n_samples=n_samples,elapsed_days=elapsed,
                                seed=int(rng.integers(2**31)))
    # In the current TPP-B/C implementation, the feature flow is conditioned on the
    # feature-history state; elapsed gaps affect the timing state but not the feature
    # encoder. Therefore the whole next-state distribution can be sampled in one batch.
    gaps=np.diff(np.r_[0.,seq.times])
    hist={"features":seq.contexts.copy(), "iats":np.r_[gaps[:j], elapsed+float(np.median(waits))]}
    samples=model.sample_patients(n_samples,conditioning_inputs=current,history=hist,
                                  seed=int(rng.integers(2**31)))
    indices=physiology_feature_indices(preprocessor)
    values={}
    for k,v in enumerate(VARIABLES):
        mu,sd=preprocessor.measurement_scales[v]
        values[v]=mu+sd*samples[:,indices[k]]
    return waits, values
