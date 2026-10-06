"""Patient sequences and causal interval contexts for implemented TPP baselines."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .preprocessing import encode_dynamic_covariates, VARIABLES
from .synthetic import EVENT_TYPES

@dataclass
class TPPSequence:
    patient_id: str
    static: np.ndarray
    contexts: np.ndarray  # at entry, then immediately after each observed visit
    times: np.ndarray  # positive, increasing days since entry
    marks: np.ndarray  # schema IDs 0,1,2 (ignored by unmarked models)
    end: float  # observation duration in days

    def validate(self):
        if not np.isfinite(self.end) or self.end <= 0:
            raise ValueError("Observation duration must be positive")
        if (not np.isfinite(self.times).all() or np.any(self.times <= 0)
                or np.any(np.diff(self.times) <= 0) or np.any(self.times > self.end)):
            raise ValueError("Events must be strictly ordered inside (0,end]")
        if self.marks.shape != self.times.shape or np.any(self.marks < 0) or np.any(self.marks >= len(EVENT_TYPES)):
            raise ValueError("Invalid event marks")
        if self.contexts.ndim != 2 or len(self.contexts) != len(self.times)+1 or self.static.ndim != 1:
            raise ValueError("Invalid context dimensions")
        if not np.isfinite(self.static).all() or not np.isfinite(self.contexts).all():
            raise ValueError("Nonfinite model inputs")
        return self


def prepare_baseline_sequences(dataset, preprocessor, patient_ids=None):
    """Use observed tables only; all durations are in days, all ages age/100.

    Hawkes consumes static inputs. Renewal MLPs consume baseline plus dynamic state,
    latest known physiology/masks/staleness, origin age and event-history summaries.
    Context at event+1ns includes that observed event/panel but no later records.
    Parameters are then frozen for the following interval. No complete-grid filling.
    """
    patients = dataset.patients
    if patient_ids is not None:
        ids = tuple(patient_ids)
        if len(set(ids)) != len(ids) or not set(ids) <= set(patients.patient_id):
            raise ValueError("Patient IDs must be unique and present")
        patients = patients[patients.patient_id.isin(ids)]
    result = []
    for patient in patients.itertuples():
        p = pd.Series(patient._asdict())
        static = preprocessor.static_row(p)
        events = dataset.events[dataset.events.patient_id == p.patient_id].sort_values("event_date")
        if not events.event_type.isin(EVENT_TYPES).all(): raise ValueError("Unknown event type")
        times = (pd.to_datetime(events.event_date)-p.study_start_date).dt.total_seconds().to_numpy()/86400
        end = (p.study_end_date-p.study_start_date).total_seconds()/86400
        marks = events.event_type.map({v:i for i,v in enumerate(EVENT_TYPES)}).to_numpy(dtype=int)
        contexts = []
        origins = [p.study_start_date]+events.event_date.tolist()
        for j,origin in enumerate(origins):
            cutoff = origin if j == 0 else min(origin+pd.Timedelta(nanoseconds=1),p.study_end_date)
            dyn = encode_dynamic_covariates(dataset,preprocessor,p.patient_id,cutoff)["values"]
            phys = []
            available = dataset.measurements[(dataset.measurements.patient_id == p.patient_id) &
                (dataset.measurements.measurement_date < cutoff)]
            for variable in VARIABLES:
                rows = available[available.variable == variable].sort_values("measurement_date")
                value,known,stale = 0.,0.,0.
                if len(rows):
                    row = rows.iloc[-1]
                    from .preprocessing import UNITS
                    if row.unit != UNITS[variable]: raise ValueError("Measurement unit mismatch")
                    if pd.notna(row.value) and np.isfinite(float(row.value)):
                        mean,std = preprocessor.measurement_scales[variable]
                        value,known = (float(row.value)-mean)/std,1.
                    stale = np.log1p((origin-row.measurement_date).total_seconds()/86400)
                phys += [value,known,stale]
            last_iat = times[j-1]-times[j-2] if j >= 2 else 0.
            last_mark = np.zeros(len(EVENT_TYPES))
            if j: last_mark[marks[j-1]]=1
            summary = [preprocessor.age_scaled((origin-p.date_of_birth).total_seconds()/(86400*365.2425)),
                       np.log1p((origin-p.study_start_date).total_seconds()/86400),
                       np.log1p(j),np.log1p(last_iat),float(j >= 2)]
            contexts.append(np.concatenate([static,dyn,phys,summary,last_mark]).astype(np.float64))
        result.append(TPPSequence(p.patient_id,static.astype(np.float64),np.stack(contexts),times,marks,end).validate())
    return result
