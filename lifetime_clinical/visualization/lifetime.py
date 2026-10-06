"""Lifetime-horizon views, with explicit boundaries of available information."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from ..data.synthetic import EVENT_TYPES, YEAR_DAYS
from ..data.preprocessing import VARIABLES

COLOURS = ("tab:blue", "tab:orange", "tab:green")
MARKERS = ("o", "^", "s")


def _marginal_quantiles(prediction, probabilities=(.1, .9)):
    """Vectorised bisection of GP-C quadrature CDFs, avoiding sampling jitter."""
    center = prediction.mean
    result = []
    for probability in probabilities:
        width = 8*prediction.scales
        for _ in range(40):
            lower, upper = center-width, center+width
            outside = ((prediction.cdf(lower) > probability)
                       | (prediction.cdf(upper) < probability))
            if not outside.any():
                break
            width = np.where(outside, width*2, width)
        else:
            raise FloatingPointError("Could not bracket predictive quantiles")
        for _ in range(48):
            middle = (lower+upper)/2
            below = prediction.cdf(middle) < probability
            lower = np.where(below, middle, lower)
            upper = np.where(below, upper, middle)
        result.append((lower+upper)/2)
    return np.asarray(result)


def lifetime_predictions(model, patient, dataset, preprocessor, resolution=180):
    """Leave pre-entry blank; fitted physiology and future use observed records.

    The observed physiology view is retrospective; the future view uses all
    observed measurements and the same fixed-prefix GP. Event forecasts concern
    the next visit only, with no simulated later visits or missing-history events.
    """
    row = dataset.patients.set_index("patient_id").loc[patient.patient_id]
    entry, censor = float(row.start_study_age), float(row.censor_age)
    if not 0 < entry < censor < 100:
        raise ValueError("This illustration requires 0 < entry < censor < 100")
    segments = []
    for label, left, right, block in (
        ("Retrospective fit", entry, censor, patient.block),
        ("Post-censor forecast", censor, 100., patient.block),
    ):
        ages = np.linspace(left, right, resolution)
        prediction = model.gp.forecast(block, np.repeat(ages / 100, 3),
                                       np.tile(np.arange(3), len(ages)), retrospective=True)
        segments.append(dict(label=label, ages=ages, prediction=prediction,
                             mean=prediction.mean.reshape(-1, 3),
                             quantiles=_marginal_quantiles(prediction).reshape(2, -1, 3),
                             latent=prediction.latent_sample(3, seed=101).reshape(3, -1, 3)))

    # Use a dense local grid inside each observed inter-visit interval even when
    # the full age span is large. Curves are separate at information boundaries.
    event_segments = []
    def interval(left, right, state, static, elapsed=0., label="fitted"):
        if right <= left:
            return
        ages = np.linspace(left, right, resolution)
        prediction = model.tpp.next_event_distribution(
            static, (ages-left)*YEAR_DAYS, state, elapsed_days=elapsed)
        event_segments.append(dict(ages=ages, prediction=prediction, label=label))

    model.eval()
    import torch
    with torch.no_grad():
        origins = np.r_[0., patient.times]
        ends = np.r_[patient.times, patient.end]
        for origin, end in zip(origins, ends):
            state = model.shared.states(patient.tokens[patient.token_days <= origin])[-1]
            interval(patient.origin_age+origin/YEAR_DAYS,
                     patient.origin_age+end/YEAR_DAYS, state, patient.block.static)
        # At C the survival is reset to one by conditioning on the known terminal
        # event-free gap; type probabilities also update through that survival.
        last = float(origins[-1])
        state = model.shared.states(patient.tokens[patient.token_days <= last])[-1]
        interval(censor, 100., state, patient.block.static, patient.end-last,
                 label="censor-origin next-visit forecast")
    return dict(entry=entry, censor=censor, origin=patient.origin_age,
                physiology=segments, events=event_segments)


def _boundaries(ax, view):
    ax.axvspan(0, view["entry"], color="#dbe7f5", alpha=.3, zorder=-10)
    ax.axvspan(view["entry"], view["censor"], color="#ddeddc", alpha=.25, zorder=-10)
    ax.axvspan(view["censor"], 100, color="#f6e0c7", alpha=.25, zorder=-10)
    ax.axvline(view["entry"], color="tab:blue", linestyle="--", linewidth=1.5)
    ax.axvline(view["origin"], color="gray", linestyle=":", linewidth=1.3)
    ax.axvline(view["censor"], color="tab:red", linestyle="--", linewidth=1.5)
    ax.set_xlim(0, 100)
    ax.grid(alpha=.12)


def _region_legend():
    return [Patch(facecolor="#dbe7f5", label="Pre-entry: unobserved (no predictions)"),
            Patch(facecolor="#ddeddc", label="Observed window: retrospective fit"),
            Patch(facecolor="#f6e0c7", label="After censoring: conditional forecast"),
            Line2D([], [], color="gray", linestyle=":", label="Model prefix origin"),
            Line2D([], [], color="tab:red", linestyle="--", label="Right censoring C")]


def plot_lifetime_events(view, patient, dataset, age_limits=None):
    """Marked intensity, next-IAT density and waiting-time survival in age units."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for segment in view["events"]:
        age, pred = segment["ages"], segment["prediction"]
        for k, colour in enumerate(COLOURS):
            axes[0].plot(age, pred["marked_hazard"][:, k]*YEAR_DAYS, color=colour, lw=1.5)
            axes[1].plot(age, pred["marked_density"][:, k]*YEAR_DAYS, color=colour, lw=1.5)
        axes[2].plot(age, pred["survival"], color="black", lw=1.5)
    observed = dataset.events[dataset.events.patient_id == patient.patient_id]
    row = dataset.patients.set_index("patient_id").loc[patient.patient_id]
    ages = (observed.event_date-row.date_of_birth).dt.total_seconds().to_numpy()/(86400*YEAR_DAYS)
    for k, (name, colour, marker) in enumerate(zip(EVENT_TYPES, COLOURS, MARKERS)):
        event_ages = ages[observed.event_type.to_numpy() == name]
        for ax, y in zip(axes, (0., 0., 1.)):
            ax.scatter(event_ages, np.full(len(event_ages), y), c=colour, marker=marker, s=22, zorder=5)
    for ax in axes:
        _boundaries(ax, view)
        if age_limits is not None: ax.set_xlim(*age_limits)
    axes[0].set(title=r"Conditional marked intensity $\lambda_k(t\mid\mathcal{H})$", ylabel="Events per year")
    axes[1].set(title=r"Next-visit density $\pi_k f_k$: waiting time mapped to arrival age", ylabel="Density per year")
    axes[2].set(title=r"No next visit yet: survival $S$ (not survival from death)", ylabel="Probability", ylim=(-.03, 1.08))
    axes[-1].set_xlabel("Patient age (years)")
    handles = _region_legend()+[Line2D([], [], color=c, marker=m, label=n)
                                for n, c, m in zip(EVENT_TYPES, COLOURS, MARKERS)]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .9))
    return fig


def plot_lifetime_physiology(view, patient, dataset, density=False, density_resolution=180):
    """Measured-value marginals and jointly sampled latent physiology per region.

    Density values use GP-C's 40-node marginal quadrature in clinical units;
    grayscale_r agrees with the standalone GP notebooks. Region predictions are
    deliberately not extended into the unknown pre-entry region.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    names = (("Systolic BP", "mmHg"), ("Glucose", "mg/dL"), ("Creatinine", "mg/dL"))
    measured = dataset.measurements[dataset.measurements.patient_id == patient.patient_id]
    for d, (ax, variable, (name, unit)) in enumerate(zip(axes, VARIABLES, names)):
        if density:
            lower = min(s["quantiles"][0, :, d].min() for s in view["physiology"])
            upper = max(s["quantiles"][1, :, d].max() for s in view["physiology"])
            span = upper-lower
            values = np.linspace(lower-.4*span, upper+.4*span, density_resolution)
            matrices = []
            for seg in view["physiology"]:
                pred = seg["prediction"]
                matrix = []
                center = pred.mean
                for value in values:
                    thresholds = center.copy()
                    thresholds[d::3] = value
                    matrix.append(np.exp(pred.marginal_log_prob(thresholds)[d::3]))
                matrices.append(np.asarray(matrix))
            vmax = max(m.max() for m in matrices)
            for seg, matrix in zip(view["physiology"], matrices):
                mesh = ax.pcolormesh(seg["ages"], values, matrix, cmap="gray_r", vmin=0, vmax=vmax, shading="auto", rasterized=True)
            fig.colorbar(mesh, ax=ax, pad=.01, label=f"Density (1/{unit})")
            ax.set_ylim(values[0], values[-1])
        for seg in view["physiology"]:
            age, q = seg["ages"], seg["quantiles"]
            if not density:
                ax.fill_between(age, q[0, :, d], q[1, :, d], color="gray", alpha=.22)
                for trajectory in seg["latent"]:
                    ax.plot(age, trajectory[:, d], color="tab:blue", alpha=.4, lw=.8)
            else:
                ax.plot(age, q[0, :, d], color="tab:orange", lw=2)
                ax.plot(age, q[1, :, d], color="tab:orange", lw=2)
            ax.plot(age, seg["mean"][:, d], color="black" if not density else "tab:blue", lw=2)
        rows = measured[(measured.variable == variable) & measured.value.notna()]
        ax.scatter(rows.measurement_age, rows.value, color="red", marker="x", s=24, zorder=5)
        _boundaries(ax, view)
        ax.set(title=f"{name}: {'measured-value predictive density' if density else 'physiology and measured-value predictions'}", ylabel=f"{name} ({unit})")
    handles = _region_legend()+[Line2D([], [], c="red", marker="x", ls="", label="Recorded measurements")]
    if density:
        handles += [Line2D([], [], c="tab:blue", lw=2, label="Predictive mean of y"),
                    Line2D([], [], c="tab:orange", lw=2, label="10/90% quantiles of y")]
    else:
        handles += [Line2D([], [], c="black", lw=2, label="Predictive mean of y"),
                    Line2D([], [], c="tab:blue", lw=.8, label="Joint latent f draws"),
                    Patch(facecolor="gray", alpha=.3, label="80% predictive interval of y")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=8)
    axes[-1].set_xlabel("Patient age (years)")
    fig.tight_layout(rect=(0, 0, 1, .9))
    return fig
