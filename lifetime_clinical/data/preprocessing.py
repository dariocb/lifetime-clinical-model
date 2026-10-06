"""Implemented train-only transforms and strictly causal, irregular model inputs.

No fitting or predictive model is implemented here. Arrays are NumPy float32;
padding/missingness masks are explicit, with raw targets retained for likelihoods.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .synthetic import MARITAL, EMPLOYMENT, SEX, EVENT_TYPES, YEAR_DAYS

VARIABLES = ("systolic_bp", "glucose", "creatinine")
UNITS = {"systolic_bp":"mmHg", "glucose":"mg/dL", "creatinine":"mg/dL"}
CATEGORIES = {"sex":SEX, "marital_status":MARITAL, "employment":EMPLOYMENT}

def _missing(value):
    return pd.isna(value) or str(value).strip() in ("", "unknown/missing")

def _category(value, name):
    levels = CATEGORIES[name] + ("<unseen>",)
    if _missing(value):
        value = "unknown/missing" if "unknown/missing" in levels else "<unseen>"
    key = str(value) if str(value) in levels else "<unseen>"
    return np.asarray([float(key == level) for level in levels],dtype=np.float32)

def _scale(values):
    values = np.asarray(values,dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (0.0,1.0)
    return (float(values.mean()),float(values.std()) or 1.0)

@dataclass
class ClinicalPreprocessor:
    train_ids: tuple
    measurement_scales: dict
    text_encoder: object = None
    ehr_dimension: int = 0

    @property
    def static_feature_names(self):
        names = ["entry_age_scaled"]
        for name in CATEGORIES:
            names += [f"{name}={level}" for level in CATEGORIES[name]+("<unseen>",)]
            names += [f"{name}_missing"]
        return names + [f"ehr_{i}" for i in range(self.ehr_dimension)] + ["ehr_missing"]

    @property
    def token_feature_names(self):
        names = ["kind=event","kind=measurement","kind=covariate", "age_scaled", "log1p_elapsed_days",
                 "log1p_iat_days","iat_known", "measurement_value_z", "value_known"]
        names += [f"event={v}" for v in EVENT_TYPES]
        names += [f"variable={v}" for v in VARIABLES+("marital_status","employment","ehr_history")]
        for name in ("marital_status","employment"):
            names += [f"{name}={level}" for level in CATEGORIES[name]+("<unseen>",)]
        return names + [f"ehr_{i}" for i in range(self.ehr_dimension)] + ["missing"]

    def age_scaled(self, age):
        return age / 100.0

    def text(self, value):
        if not self.ehr_dimension:
            return np.empty(0,dtype=np.float32)
        if _missing(value):
            return np.zeros(self.ehr_dimension,dtype=np.float32)
        vector = np.asarray(self.text_encoder.encode([str(value)],normalize_embeddings=True,
            convert_to_numpy=True,show_progress_bar=False),dtype=np.float32)[0]
        if vector.shape != (self.ehr_dimension,) or not np.isfinite(vector).all():
            raise ValueError("Invalid EHR embedding")
        return vector

    def static_row(self, patient):
        parts = [np.array([self.age_scaled(patient.start_study_age)],dtype=np.float32)]
        for name in CATEGORIES:
            value = getattr(patient,name)
            parts += [_category(value,name),np.array([float(_missing(value))])]
        parts += [self.text(patient.ehr_history),np.array([float(_missing(patient.ehr_history))])]
        return np.concatenate(parts).astype(np.float32)

    def token(self, kind, date, patient, variable=None, value=None, event_type=None, iat=np.nan):
        features = dict.fromkeys(self.token_feature_names,0.0)
        features[f"kind={kind}"] = 1
        features["age_scaled"] = self.age_scaled((date-patient.date_of_birth).total_seconds()/(86400*YEAR_DAYS))
        elapsed = (date-patient.study_start_date).total_seconds()/86400
        features["log1p_elapsed_days"] = np.log1p(elapsed)
        if event_type is not None:
            if event_type not in EVENT_TYPES:
                raise ValueError(f"Unknown event type: {event_type}")
            features[f"event={event_type}"] = 1
            if np.isfinite(iat):
                if iat <= 0: raise ValueError("Known IATs must be positive")
                features["log1p_iat_days"],features["iat_known"] = np.log1p(iat),1
        if variable is not None:
            if variable not in VARIABLES+("marital_status","employment","ehr_history"):
                raise ValueError(f"Unknown dynamic variable: {variable}")
            features[f"variable={variable}"] = 1
            missing = _missing(value)
            features["missing"] = float(missing)
            if variable in VARIABLES and not missing:
                mean,std = self.measurement_scales[variable]
                features["measurement_value_z"] = (float(value)-mean)/std
                features["value_known"] = 1
            elif variable in ("marital_status","employment"):
                for level,indicator in zip(CATEGORIES[variable]+("<unseen>",),_category(value,variable)):
                    features[f"{variable}={level}"] = indicator
            elif variable == "ehr_history":
                for i,v in enumerate(self.text(value)):
                    features[f"ehr_{i}"] = v
        return np.asarray(list(features.values()),dtype=np.float32)


def patient_split(dataset, fractions=(0.7,0.15,0.15), seed=42, counts=None):
    """Patient-disjoint IDs; no splitting of one person's rows across partitions."""
    ids = dataset.patients.patient_id.to_numpy()
    fractions = np.asarray(fractions,dtype=float)
    if len(fractions) != 3 or not np.isfinite(fractions).all() or (fractions <= 0).any() or not np.isclose(fractions.sum(),1):
        raise ValueError("Three positive fractions summing to one are required")
    if len(ids) < 3 or len(set(ids)) != len(ids):
        raise ValueError("At least three distinct patients are required")
    if counts is None:
        counts = np.floor(fractions*len(ids)).astype(int)
        counts[-1] = len(ids)-counts[:2].sum()
    else:
        if len(counts) != 3 or any(isinstance(c, bool) or not isinstance(c, (int, np.integer)) or c < 1 for c in counts) or sum(counts) != len(ids):
            raise ValueError("counts must contain three positive integers summing to cohort size")
        counts = np.asarray(counts, dtype=int)
    if (counts < 1).any():
        raise ValueError("Each partition needs at least one patient; increase cohort or fractions")
    shuffled = np.random.default_rng(seed).permutation(ids)
    a,b = counts[0],counts[0]+counts[1]
    return {"train":tuple(shuffled[:a]),"validation":tuple(shuffled[a:b]),"test":tuple(shuffled[b:])}


def fit_static_transform(dataset, train_ids, use_ehr=False, text_encoder=None):
    """Fit physiological scales on train patients only; age is always age/100.

    use_ehr=True loads pretrained MiniLM, or accepts an injected pretrained encoder.
    False explicitly excludes text features (dimension zero); no fake embeddings.
    Categorical vocabularies come from the declared schema, not held-out records.
    """
    train_ids = tuple(train_ids)
    patients = dataset.patients[dataset.patients.patient_id.isin(train_ids)]
    if not train_ids or len(set(train_ids)) != len(train_ids) or len(patients) != len(train_ids):
        raise ValueError("train_ids must be nonempty, unique and present")
    scales = {}
    for variable in VARIABLES:
        subset = dataset.measurements[(dataset.measurements.patient_id.isin(train_ids)) &
                                      (dataset.measurements.variable == variable)]
        if len(subset) and not subset.unit.eq(UNITS[variable]).all():
            raise ValueError(f"Expected {UNITS[variable]} for {variable}; convert units first")
        scales[variable] = _scale(pd.to_numeric(subset.value,errors="raise"))
    dimension = 0
    if use_ehr:
        if text_encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError('EHR input requires pip install -e ".[embeddings]"; or set use_ehr=False') from exc
            from .embeddings import DEFAULT_MODEL
            text_encoder = SentenceTransformer(DEFAULT_MODEL,device="cpu")
        dimension = int(text_encoder.get_sentence_embedding_dimension())
        if dimension <= 0: raise ValueError("Invalid embedding dimension")
    elif text_encoder is not None:
        raise ValueError("Set use_ehr=True when supplying a text encoder")
    return ClinicalPreprocessor(train_ids,scales,text_encoder,dimension)


def encode_dynamic_covariates(dataset, preprocessor, patient_id, query_time):
    """Latest strictly prior recorded statuses/note; entry snapshot at entry allowed.

    Unknown updates overwrite earlier known values; never backfill from the future.
    Return input vector, schema and staleness in log1p days.
    """
    patients = dataset.patients[dataset.patients.patient_id == patient_id]
    if len(patients) != 1: raise ValueError("Unknown patient")
    p = patients.iloc[0]
    t = pd.Timestamp(query_time)
    if not p.study_start_date <= t <= p.study_end_date: raise ValueError("Query outside observation window")
    records = dataset.dynamic_covariates[(dataset.dynamic_covariates.patient_id == patient_id) &
        (dataset.dynamic_covariates.recorded_at < t)]
    values,names = [],[]
    for variable in ("marital_status","employment","ehr_history"):
        value,date = p[variable],p.study_start_date
        available = records[records.variable == variable].sort_values("recorded_at",kind="stable")
        if len(available): value,date = available.iloc[-1]["value"],available.iloc[-1].recorded_at
        if variable == "ehr_history":
            values.extend(preprocessor.text(value));names.extend([f"ehr_{i}" for i in range(preprocessor.ehr_dimension)])
        else:
            values.extend(_category(value,variable));names.extend([f"{variable}={v}" for v in CATEGORIES[variable]+("<unseen>",)])
        values.extend([float(_missing(value)),np.log1p((t-date).total_seconds()/86400)])
        names.extend([f"{variable}_missing",f"{variable}_log1p_days_since_record"])
    return {"values":np.asarray(values,dtype=np.float32),"feature_names":names}


def build_observation_windows(dataset, patient_ids=None):
    """Likelihood metadata only; future censor intervals are not predictor features."""
    p = dataset.patients
    if patient_ids is not None: p = p[p.patient_id.isin(patient_ids)]
    result = p[["patient_id","study_start_date","study_end_date","start_study_age","censor_age"]].copy()
    result["window_days"] = (result.study_end_date-result.study_start_date).dt.total_seconds()/86400
    result["right_censor_days"] = p.right_censor_days.to_numpy()
    result["zero_events"] = p.n_observed_events.to_numpy() == 0
    result["pre_entry_history_unknown"] = True
    return result.reset_index(drop=True)


def prepare_continuous_targets(dataset, preprocessor, patient_ids=None):
    """Tidy irregular multi-output targets; retain raw values, units, ages and dates.

    Missing targets get a false target mask and zero normalized placeholder.
    No interpolation or filling of unmeasured times/variables is performed.
    """
    targets = dataset.measurements.copy()
    if patient_ids is not None: targets = targets[targets.patient_id.isin(patient_ids)].copy()
    if not targets.variable.isin(VARIABLES).all(): raise ValueError("Unknown physiological variable")
    targets["variable_id"] = targets.variable.map({v:i for i,v in enumerate(VARIABLES)})
    targets["target_mask"] = pd.to_numeric(targets.value,errors="raise").map(np.isfinite)
    normalized = []
    for row in targets.itertuples():
        if row.unit != UNITS[row.variable]: raise ValueError("Measurement unit mismatch")
        mean,std = preprocessor.measurement_scales[row.variable]
        normalized.append((row.value-mean)/std if np.isfinite(row.value) else 0.0)
    targets["target_z"] = np.asarray(normalized,dtype=np.float32)
    targets["query_age_scaled"] = pd.to_numeric(targets.measurement_age,errors="raise") / 100.0
    return targets.reset_index(drop=True)


def build_causal_batches(dataset, preprocessor, queries):
    """Padded unified event/measurement/update tokens for explicit query timestamps.

    queries is a DataFrame with patient_id, query_time. Histories use dates < query,
    so all simultaneous target-panel values/events are excluded. Baseline input is
    the entry snapshot; updates are separate dynamic context and history tokens.
    No oracle table, end-of-study age, total counts or terminal gaps enter features.
    """
    queries = pd.DataFrame(queries)
    if not {"patient_id","query_time"} <= set(queries) or queries.empty:
        raise ValueError("Nonempty queries with patient_id and query_time required")
    targets = prepare_continuous_targets(dataset,preprocessor)
    histories,dates,static,dynamic,ages,query_dates = [],[],[],[],[],[]
    patients = dataset.patients.set_index("patient_id",drop=False)
    if not patients.index.is_unique: raise ValueError("Duplicate patient IDs")
    for query in queries.itertuples():
        if query.patient_id not in patients.index: raise ValueError("Unknown patient")
        p = patients.loc[query.patient_id]; t = pd.Timestamp(query.query_time)
        if not p.study_start_date <= t <= p.study_end_date: raise ValueError("Query outside observation window")
        records = []
        events = dataset.events[(dataset.events.patient_id == p.patient_id) & (dataset.events.event_date < t)]
        for row in events.itertuples():
            if row.event_date < p.study_start_date: raise ValueError("Observed event before entry")
            records.append((row.event_date,preprocessor.token("event",row.event_date,p,event_type=row.event_type,iat=row.iat_days)))
        measures = targets[(targets.patient_id == p.patient_id) & (targets.measurement_date < t)]
        for row in measures.itertuples():
            if row.measurement_date < p.study_start_date: raise ValueError("Observed measurement before entry")
            records.append((row.measurement_date,preprocessor.token("measurement",row.measurement_date,p,variable=row.variable,value=row.value if row.target_mask else None)))
        updates = dataset.dynamic_covariates[(dataset.dynamic_covariates.patient_id == p.patient_id) &
            (dataset.dynamic_covariates.recorded_at < t) & (dataset.dynamic_covariates.source != "baseline")]
        for row in updates.itertuples():
            if row.recorded_at < p.study_start_date: raise ValueError("Observed update before entry")
            records.append((row.recorded_at,preprocessor.token("covariate",row.recorded_at,p,variable=row.variable,value=row.value)))
        records.sort(key=lambda item:item[0])
        histories.append([v for _,v in records]);dates.append([d for d,_ in records])
        static.append(preprocessor.static_row(p))
        context = encode_dynamic_covariates(dataset,preprocessor,p.patient_id,t)
        dynamic.append(context["values"])
        ages.append(preprocessor.age_scaled((t-p.date_of_birth).total_seconds()/(86400*YEAR_DAYS)))
        query_dates.append(t)
    longest = max(1,max(map(len,histories)))
    tokens = np.zeros((len(histories),longest,len(preprocessor.token_feature_names)),dtype=np.float32)
    mask = np.zeros(tokens.shape[:2],dtype=bool)
    timestamps = np.full(tokens.shape[:2],np.datetime64("NaT"),dtype="datetime64[ns]")
    for i,history in enumerate(histories):
        if history:
            tokens[i,:len(history)] = history;mask[i,:len(history)] = True
            timestamps[i,:len(history)] = np.asarray(dates[i],dtype="datetime64[ns]")
    return {"patient_ids":queries.patient_id.to_numpy(),"query_times":np.asarray(query_dates,dtype="datetime64[ns]"),
            "static":np.asarray(static,dtype=np.float32),"dynamic":np.asarray(dynamic,dtype=np.float32),
            "query_age_scaled":np.asarray(ages,dtype=np.float32),
            "query_elapsed_days":np.asarray([(pd.Timestamp(q.query_time)-patients.loc[q.patient_id].study_start_date).total_seconds()/86400
                                             for q in queries.itertuples()],dtype=np.float32),"history":tokens,"history_mask":mask,
            "history_times":timestamps,"lengths":mask.sum(axis=1),
            "static_feature_names":preprocessor.static_feature_names,
            "dynamic_feature_names":context["feature_names"],"history_feature_names":preprocessor.token_feature_names,
            "ehr_enabled":bool(preprocessor.ehr_dimension)}


def prepare_event_targets(dataset, patient_ids=None):
    """Raw marked-event targets, preserving unknown first IATs and entry delays.

    No density head should treat the first delay as a full known IAT. Right
    censoring/zero-event contributions come from build_observation_windows.
    """
    events = dataset.events.copy()
    if patient_ids is not None: events = events[events.patient_id.isin(patient_ids)].copy()
    if not events.event_type.isin(EVENT_TYPES).all(): raise ValueError("Unknown event type")
    events["mark_id"] = events.event_type.map({v:i for i,v in enumerate(EVENT_TYPES)})
    events["iat_known"] = np.isfinite(pd.to_numeric(events.iat_days,errors="raise").astype(float))
    if (events.loc[events.iat_known,"iat_days"] <= 0).any(): raise ValueError("Known IATs must be positive")
    return events.reset_index(drop=True)
