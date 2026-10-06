"""Adapters used by the tutorial notebooks.

They do not define new statistical models. They only expose the canonical dataset
in compact forms that make the six model assumptions easy to inspect.
"""
from __future__ import annotations
import numpy as np
from .baselines import TPPSequence, prepare_baseline_sequences
from .preprocessing import VARIABLES, CATEGORIES
from .synthetic import EVENT_TYPES


def context_feature_names(preprocessor):
    names = list(preprocessor.static_feature_names)
    for variable in ("marital_status", "employment", "ehr_history"):
        if variable == "ehr_history":
            names.extend([f"ehr_{i}" for i in range(preprocessor.ehr_dimension)])
        else:
            names.extend([f"{variable}={v}" for v in CATEGORIES[variable] + ("<unseen>",)])
        names.extend([f"{variable}_missing", f"{variable}_log1p_days_since_record"])
    for variable in VARIABLES:
        names.extend([f"{variable}_value_z", f"{variable}_known", f"{variable}_log1p_staleness"])
    names.extend(["current_age_scaled", "log1p_days_since_entry", "log1p_event_count",
                  "log1p_last_iat", "last_iat_known"])
    names.extend([f"last_event={v}" for v in EVENT_TYPES])
    return names


def physiology_feature_indices(preprocessor):
    names = context_feature_names(preprocessor)
    return tuple(names.index(f"{v}_value_z") for v in VARIABLES)


def physiology_known_indices(preprocessor):
    names = context_feature_names(preprocessor)
    return tuple(names.index(f"{v}_known") for v in VARIABLES)


def prepare_dynamic_flow_sequences(dataset, preprocessor, patient_ids=None):
    """TPP-B/C sequences beginning once all continuous physiology is observed.

    The three continuous flow coordinates are the latest standardized values of
    systolic BP, glucose and creatinine. Missingness indicators and staleness are
    retained as conditioning metadata but are not assigned continuous density.
    Starting after the first complete physiological state avoids treating an
    imputation placeholder as an observed continuous target.
    """
    sequences = prepare_baseline_sequences(dataset, preprocessor, patient_ids)
    known = physiology_known_indices(preprocessor)
    result = []
    for seq in sequences:
        complete = np.flatnonzero(np.all(seq.contexts[:, known] > 0.5, axis=1))
        if not len(complete):
            continue
        j = int(complete[0])
        origin = 0.0 if j == 0 else float(seq.times[j - 1])
        times = seq.times[j:] - origin
        marks = seq.marks[j:]
        contexts = seq.contexts[j:]
        end = seq.end - origin
        if end <= 0 or len(contexts) != len(times) + 1:
            continue
        result.append(TPPSequence(seq.patient_id, seq.static.copy(), contexts.copy(),
                                  times.copy(), marks.copy(), float(end)).validate())
    return result
