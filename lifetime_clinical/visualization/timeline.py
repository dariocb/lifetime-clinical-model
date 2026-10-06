from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from ..data.preprocessing import VARIABLES

_EVENT_Y = {"outpatient visit": 2.7, "emergency visit": 3.0, "inpatient visit": 3.3}
_MEAS_Y = {"systolic_bp": 1.55, "glucose": 1.25, "creatinine": .95}


def plot_patient_timeline(dataset, patient_id, ax=None, show_hidden=True, title=None):
    p = dataset.patients.loc[dataset.patients.patient_id == patient_id]
    if len(p) != 1:
        raise ValueError("patient_id must identify exactly one patient")
    p = p.iloc[0]
    ax = ax or plt.subplots(figsize=(12, 4.2))[1]
    entry, end = float(p.start_study_age), float(p.censor_age)
    ax.axvspan(entry, end, alpha=.08, label="observation window")
    ax.axvline(entry, linestyle="--", linewidth=1.2)
    ax.axvline(end, linestyle="--", linewidth=1.2)
    ax.text(entry, 3.68, "entry", ha="center", va="bottom", fontsize=9)
    ax.text(end, 3.68, "censor/end", ha="center", va="bottom", fontsize=9)

    if show_hidden:
        hidden = dataset.pre_study_events[dataset.pre_study_events.patient_id == patient_id]
        if len(hidden):
            ax.scatter(hidden.event_age, np.full(len(hidden), 3.58), marker="x", s=24,
                       alpha=.45, label="hidden pre-study events (oracle only)")

    events = dataset.events[dataset.events.patient_id == patient_id]
    for kind, rows in events.groupby("event_type"):
        ax.scatter(rows.event_age, np.full(len(rows), _EVENT_Y.get(kind, 3.0)), s=34,
                   label=kind)

    measures = dataset.measurements[dataset.measurements.patient_id == patient_id]
    for variable in VARIABLES:
        rows = measures[measures.variable == variable]
        if len(rows):
            ax.scatter(rows.measurement_age, np.full(len(rows), _MEAS_Y[variable]), marker="|",
                       s=160, linewidths=1.6, label=variable)

    updates = dataset.dynamic_covariates[(dataset.dynamic_covariates.patient_id == patient_id) &
                                         (dataset.dynamic_covariates.source != "baseline")]
    if len(updates):
        ax.scatter(updates.recorded_age, np.full(len(updates), .42), marker=".", s=18,
                   alpha=.55, label="dynamic covariate update")

    ax.set_yticks([.42, .95, 1.25, 1.55, 2.7, 3.0, 3.3, 3.58])
    ax.set_yticklabels(["covariates", "creatinine", "glucose", "systolic BP",
                        "outpatient", "emergency", "inpatient", "pre-study"])
    left = max(0, min(entry - 4, float(p.age) - 25)) if show_hidden else entry - .5
    ax.set_xlim(left, end + .5)
    ax.set_ylim(.15, 3.9)
    ax.set_xlabel("age (years)")
    ax.set_title(title or f"Patient {patient_id}: observed window and irregular clinical timeline")
    handles, labels = ax.get_legend_handles_labels()
    # Deduplicate labels while keeping order.
    seen = set(); hh=[]; ll=[]
    for h,l in zip(handles, labels):
        if l not in seen:
            hh.append(h); ll.append(l); seen.add(l)
    ax.legend(hh, ll, frameon=False, ncol=3, fontsize=8, loc="upper left")
    ax.grid(axis="x", alpha=.15)
    return ax


def plot_patient_clinical_timeline(dataset, patient_id, *, show_hidden=True,
                                   figsize=(14, 9), title=None):
    """Plot visits, censoring and measured physiology on a shared patient-age axis.

    Returns ``(figure, axes)``: the event/update panel followed by three
    measurement panels. Legends appear above the panels. Measurement values
    are shown at their recorded ages without interpolation. Hidden pre-study
    events are oracle truth; pass ``show_hidden=False`` to omit them.
    A patient with no observed visits has a censored wait starting at entry.
    """
    patients = dataset.patients.loc[dataset.patients.patient_id == patient_id]
    if len(patients) != 1:
        raise ValueError("patient_id must identify exactly one patient")
    patient = patients.iloc[0]
    visits = dataset.events[dataset.events.patient_id == patient_id].sort_values("event_age")
    hidden = dataset.pre_study_events[dataset.pre_study_events.patient_id == patient_id]
    measured = dataset.measurements[dataset.measurements.patient_id == patient_id]
    updates = dataset.dynamic_covariates[
        (dataset.dynamic_covariates.patient_id == patient_id)
        & (dataset.dynamic_covariates.source == "update")
    ]
    entry, endpoint = float(patient.start_study_age), float(patient.censor_age)
    last_visit = float(visits.event_age.max()) if len(visits) else entry
    tail_years = endpoint - last_visit
    left = entry - .7
    if show_hidden and len(hidden):
        left = min(left, float(hidden.event_age.min()) - .7)
    left = max(0, left)

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1, 1, 1]})
    for ax in axes:
        ax.axvspan(entry, endpoint, color="tab:blue", alpha=.07)
        ax.axvline(entry, color="tab:blue", linestyle="--", linewidth=1.2)
        ax.axvline(endpoint, color="black", linestyle="--", linewidth=1.2)
        ax.set(xlim=(left, endpoint + .7))
        ax.grid(axis="x", alpha=.15)

    top = axes[0]
    if show_hidden and len(hidden):
        top.scatter(hidden.event_age, np.full(len(hidden), 4), color="grey",
                    marker="x", s=32, label="hidden visits (oracle only)")
    visit_styles = [
        ("outpatient visit", 1, "tab:blue", "o"),
        ("emergency visit", 2, "tab:orange", "^"),
        ("inpatient visit", 3, "tab:green", "s"),
    ]
    for kind, lane, colour, symbol in visit_styles:
        rows = visits[visits.event_type == kind]
        if len(rows):
            top.scatter(rows.event_age, np.full(len(rows), lane), color=colour,
                        marker=symbol, s=48, label=kind, zorder=3)
    if len(updates):
        top.scatter(updates.recorded_age, np.zeros(len(updates)), color="tab:purple",
                    marker=".", s=28, label="recorded covariate updates")
    top.axvspan(last_visit, endpoint, color="tab:orange", alpha=.13)
    top.annotate("", xy=(endpoint, 4.8), xytext=(last_visit, 4.8),
                 arrowprops={"arrowstyle": "<->", "color": "darkorange", "linewidth": 1.5})
    top.text((last_visit + endpoint)/2, 5.02,
             f"right-censored wait: {tail_years:.2f} years", ha="center",
             fontsize=9, color="saddlebrown")
    top.text(entry, 5.6, "observation begins", ha="right", fontsize=9, color="tab:blue")
    top.text(endpoint, 5.6, "follow-up ends\n(no event here)", ha="right", fontsize=9)
    lanes = [0, 1, 2, 3] + ([4] if show_hidden and len(hidden) else [])
    labels = ["covariate updates", "outpatient", "emergency", "inpatient"]
    if show_hidden and len(hidden):
        labels.append("hidden pre-study")
    top.set(yticks=lanes, yticklabels=labels, ylim=(-.5, 6.1))

    for ax, variable, name, unit in zip(
        axes[1:], ["systolic_bp", "glucose", "creatinine"],
        ["Systolic BP", "Glucose", "Creatinine"], ["mmHg", "mg/dL", "mg/dL"]
    ):
        rows = measured[measured.variable == variable]
        for source, colour, marker, label in [
            ("event_linked", "tab:blue", "o", "event-linked measurement"),
            ("independent", "tab:purple", "D", "independent measurement"),
        ]:
            selected = rows[rows.measurement_source == source]
            if len(selected):
                ax.scatter(selected.measurement_age, selected.value, color=colour,
                           marker=marker, s=30, label=label, zorder=3)
        ax.set_ylabel(f"{name}\n({unit})")
    axes[-1].set_xlabel("patient age (years)")
    # One deduplicated figure legend for visits, updates and measurement sources.
    legend_items = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            legend_items.setdefault(label, handle)
    if legend_items:
        fig.legend(list(legend_items.values()), list(legend_items),
                   loc="upper center", bbox_to_anchor=(.5, .955),
                   ncol=4, frameon=False, fontsize=9)
    fig.suptitle(title or f"Patient {patient_id}: observed visits and irregular physiology",
                 fontsize=14, y=.995)
    fig.tight_layout(rect=(0, .04, 1, .92))
    note = "Orange shading means no observed visits, not no measurements."
    if show_hidden and len(hidden):
        note = "Pre-study events are hidden simulator truth. " + note
    fig.text(.12, .015, note, fontsize=9)
    return fig, axes
