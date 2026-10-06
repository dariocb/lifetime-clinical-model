"""Illustrative psychiatric hospital data, not a fitted clinical simulator.

Age is measured at study end; counts refer to OBSERVED timelines. Hidden
pre-study events are held in an oracle-only table, never in baseline EHR text.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd

MARITAL = ("single", "separated", "widowed", "unknown/missing", "partnered", "married", "divorced")
EMPLOYMENT = ("employed", "unemployed", "unknown/missing")
SEX = ("male", "female")
EVENT_TYPES = ("outpatient visit", "emergency visit", "inpatient visit")
YEAR_DAYS = 365.2425
STATIC_VARIABLES = ("date_of_birth", "sex")
DYNAMIC_VARIABLES = ("age", "marital_status", "employment", "ehr_history",
                     "systolic_bp", "glucose", "creatinine")

@dataclass(frozen=True)
class SyntheticConfig:
    scenario: str = "simple"  # simple, dynamic_events, structured_physiology, flexible_observations, combined
    event_rate_per_year: float = 1.5
    trajectory_strength: float = 1.0
    covariate_strength: float = 1.0
    history_strength: float = 1.0
    residual_strength: float = 1.0
    n_samples: int = 500
    mean_events: float = 12.0
    count_distribution: str = "negative_binomial"  # or poisson
    count_dispersion: float = 3.0
    seed: int = 42
    study_end: str = "2018-10-31"
    age_range: tuple[float, float] = (18.0, 95.0)
    age_means: tuple[float, ...] = (24.0, 50.0, 75.0)
    age_stds: tuple[float, ...] = (4.0, 14.5, 2.5)
    age_weights: tuple[float, ...] = (0.35, 0.35, 0.30)
    # Optional piecewise uniform density; len(weights) == len(edges)-1.
    age_bin_edges: tuple[float, ...] | None = None
    age_bin_weights: tuple[float, ...] | None = None
    followup_years: tuple[float, float] = (1.0, 15.0)
    earliest_study_age: float = 16.0
    pre_study_event_rate: float = 0.6  # expected events per hidden year
    patient_rate_sigma: float = 0.5
    timeline_beta: tuple[float, float] = (1.0, 1.0)
    event_type_weights: tuple[float, ...] = (0.70, 0.20, 0.10)
    marital_weights: tuple[float, ...] = (0.30, 0.08, 0.10, 0.06, 0.14, 0.22, 0.10)
    employment_weights: tuple[float, ...] = (0.50, 0.43, 0.07)
    sex_weights: tuple[float, ...] = (0.48, 0.52)
    ehr_missing_probability: float = 0.1
    ehr_field_missing_probability: float = 0.35
    include_measurements: bool = True
    measurement_probability: float = 0.5  # probability of a panel at each visit
    measurement_schedule: str = "event_linked"  # event_linked, independent, mixed
    independent_measurement_rate: float = 2.0  # expected panels per observed year

    include_dynamic_covariates: bool = True
    marital_update_rate: float = 0.2  # recorded updates per observed year
    employment_update_rate: float = 0.5
    ehr_update_rate: float = 1.0

    def __post_init__(self):
        if self.scenario not in ("simple", "dynamic_events", "structured_physiology", "flexible_observations", "combined"):
            raise ValueError("Unknown synthetic scenario")
        for name in ("event_rate_per_year", "trajectory_strength", "covariate_strength", "history_strength", "residual_strength"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not isinstance(self.n_samples, int) or isinstance(self.n_samples, bool) or self.n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        for name in ("mean_events", "pre_study_event_rate", "patient_rate_sigma", "independent_measurement_rate", "marital_update_rate", "employment_update_rate", "ehr_update_rate"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.measurement_schedule not in ("event_linked", "independent", "mixed"):
            raise ValueError("measurement_schedule must be event_linked, independent, or mixed")
        if self.count_distribution not in ("poisson", "negative_binomial"):
            raise ValueError("Unknown count_distribution")
        if not np.isfinite(self.count_dispersion) or self.count_dispersion <= 0:
            raise ValueError("count_dispersion must be positive")
        if not all(np.isfinite(self.timeline_beta)) or min(self.timeline_beta) <= 0:
            raise ValueError("timeline_beta parameters must be positive")
        lo, hi = self.age_range
        f_lo, f_hi = self.followup_years
        if not all(np.isfinite([lo,hi,f_lo,f_hi,self.earliest_study_age])) or not (0 <= self.earliest_study_age < lo < hi <= 120) or not (0 < f_lo <= f_hi):
            raise ValueError("Invalid age/followup limits; youngest age must exceed earliest_study_age")
        for name in ("event_type_weights", "marital_weights", "employment_weights", "sex_weights"):
            _probabilities(getattr(self,name), len({"event_type_weights":EVENT_TYPES,"marital_weights":MARITAL,"employment_weights":EMPLOYMENT,"sex_weights":SEX}[name]))
        for name in ("ehr_missing_probability", "ehr_field_missing_probability", "measurement_probability"):
            if not 0 <= getattr(self,name) <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        if (self.age_bin_edges is None) != (self.age_bin_weights is None):
            raise ValueError("Provide both age_bin_edges and age_bin_weights")
        if self.age_bin_edges is not None:
            edges = np.asarray(self.age_bin_edges)
            if len(edges) < 2 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0) or edges[0] < lo or edges[-1] > hi:
                raise ValueError("Age bin edges must increase inside age_range")
            _probabilities(self.age_bin_weights,len(edges)-1)
        else:
            _probabilities(self.age_weights,len(self.age_means))
            if len(self.age_stds) != len(self.age_means) or not np.all(np.isfinite(self.age_means)) or not np.all(np.isfinite(self.age_stds)) or min(self.age_stds) <= 0 or not all(lo <= m <= hi for m in self.age_means):
                raise ValueError("Invalid age mixture components")
        if pd.isna(pd.Timestamp(self.study_end)) or pd.Timestamp(self.study_end).tzinfo is not None:
            raise ValueError("study_end must be a valid timezone-naive date")

def _probabilities(weights, size):
    w = np.asarray(weights, dtype=float)
    if len(w) != size or size < 1 or not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Weights must be finite, nonnegative, nonempty, and have the expected length")
    return w / w.sum()

@dataclass
class SyntheticDataset:
    patients: pd.DataFrame
    events: pd.DataFrame
    pre_study_events: pd.DataFrame  # oracle truth, excluded from training
    measurements: pd.DataFrame
    config: SyntheticConfig
    dynamic_covariates: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(
        columns=["patient_id", "recorded_at", "recorded_age", "variable", "value", "is_missing", "source"]))
    ehr_embeddings: np.ndarray | None = None
    embedding_metadata: dict = field(default_factory=dict)
    oracle_trajectories: pd.DataFrame = field(default_factory=pd.DataFrame)

    def save(self, directory):
        """Write tidy CSV tables plus provenance; embedding rows match patients."""
        out = Path(directory); out.mkdir(parents=True, exist_ok=True)
        for name in ("patients", "events", "pre_study_events", "measurements", "dynamic_covariates"):
            getattr(self,name).to_csv(out / f"{name}.csv", index=False)
        if not self.oracle_trajectories.empty:
            self.oracle_trajectories.to_csv(out / "oracle_trajectories.csv", index=False)
        metadata = {"config": asdict(self.config), "synthetic": True,
                    "age_reference": "study_end", "embedding": self.embedding_metadata,
                    "static_variables": STATIC_VARIABLES, "dynamic_variables": DYNAMIC_VARIABLES,
                    "patient_covariate_reference": "study entry, except age/censor metadata"}
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if self.ehr_embeddings is not None:
            np.save(out / "ehr_embeddings.npy", self.ehr_embeddings)
            (out / "embedding_patient_ids.json").write_text(json.dumps(self.patients.patient_id.tolist()),encoding="utf-8")


def _sample_ages(rng, config):
    n = config.n_samples
    if config.age_bin_edges is not None:
        edges = np.asarray(config.age_bin_edges)
        bins = rng.choice(len(edges)-1, n, p=_probabilities(config.age_bin_weights,len(edges)-1))
        return rng.uniform(edges[bins], edges[bins+1])
    k = rng.choice(len(config.age_means), n, p=_probabilities(config.age_weights,len(config.age_means)))
    means, stds = np.asarray(config.age_means)[k], np.asarray(config.age_stds)[k]
    ages = rng.normal(means,stds)
    lo,hi = config.age_range
    outside = (ages < lo) | (ages > hi)
    # Rejection rather than clipping: no artificial pile-up at boundaries.
    while outside.any():
        ages[outside] = rng.normal(means[outside],stds[outside])
        outside = (ages < lo) | (ages > hi)
    return ages


def _baseline_note(rng, config):
    if rng.random() < config.ehr_missing_probability:
        return None
    clauses = [rng.choice(["Baseline psychiatric assessment: persistent anxiety reported.",
                           "Baseline psychiatric assessment: low mood and disturbed sleep reported.",
                           "Baseline psychiatric assessment: mood stable at intake.",
                           "Baseline psychiatric assessment: concentration difficulties reported."])]
    for choices in (["General physical examination: no acute findings documented.",
                     "General physical examination: mild fatigue; further examination not recorded."],
                    ["Medication history: antidepressant reported; dose not recorded.",
                     "Medication history: antipsychotic reported; adherence unknown.",
                     "Medication history: no regular medication reported."],
                    ["Prior care history is incompletely documented.",
                     "Family and social history only partially available."]):
        if rng.random() >= config.ehr_field_missing_probability:
            clauses.append(rng.choice(choices))
    return " ".join(clauses)


def generate_dataset(config=None):
    """Generate observed count-conditioned timelines and separate hidden history.

    mean_events controls expected observed count per patient, not lifetime count.
    Rates have lognormal heterogeneity and NB overdispersion (or Poisson).
    Given counts, event positions follow a scaled Beta on each observation window.
    Thus this is not a mechanistic GP/TPP simulator or a fitted clinical model.
    """
    config = config or SyntheticConfig()
    if config.scenario != "simple":
        from .structured_synthetic import generate_structured_dataset
        return generate_structured_dataset(config)
    rng = np.random.default_rng(config.seed)
    # Separate measurement randomness: changing schedules leaves patients/events identical.
    measurement_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 1]))
    end = pd.Timestamp(config.study_end).normalize()
    ages = _sample_ages(rng,config)
    patients,events,hidden,measurements = [],[],[],[]
    event_columns = ["patient_id", "event_id", "event_date", "event_age", "event_type", "iat_days", "iat_from_entry_days"]
    measurement_columns = ["patient_id", "event_id", "measurement_date", "measurement_age", "variable", "value", "unit", "measurement_source"]
    for i,age in enumerate(ages):
        pid = f"P{i:06d}"
        dob = end - pd.Timedelta(days=int(np.floor(age*YEAR_DAYS)))
        age = (end-dob).total_seconds()/(86400*YEAR_DAYS)
        available = age-config.earliest_study_age
        followup = min(float(rng.uniform(*config.followup_years)), available)
        start = end-pd.Timedelta(days=followup*YEAR_DAYS)
        start_age = (start-dob).total_seconds()/(86400*YEAR_DAYS)
        rate = config.mean_events * rng.lognormal(-config.patient_rate_sigma**2/2,config.patient_rate_sigma)
        if config.count_distribution == "poisson":
            count = int(rng.poisson(rate))
        else:
            r = config.count_dispersion
            count = int(rng.negative_binomial(r,r/(r+rate)))
        note = _baseline_note(rng,config)
        positions = np.sort(rng.beta(*config.timeline_beta,size=count))
        dates = [start + pd.Timedelta(days=float(p*followup*YEAR_DAYS)) for p in positions]
        for j,date in enumerate(dates):
            etype = rng.choice(EVENT_TYPES,p=_probabilities(config.event_type_weights,len(EVENT_TYPES)))
            eage = (date-dob).total_seconds()/(86400*YEAR_DAYS)
            eid = f"{pid}_E{j:05d}"
            # First IAT unknown because the preceding event is unobserved.
            iat = np.nan if j == 0 else (date-dates[j-1]).total_seconds()/86400
            events.append([pid,eid,date,eage,etype,iat,(date-start).total_seconds()/86400])
            if (config.include_measurements and config.measurement_schedule in ("event_linked", "mixed")
                    and measurement_rng.random() < config.measurement_probability):
                measurements.extend(_measurement_panel(measurement_rng,pid,eid,date,dob,"event_linked"))
        if config.include_measurements and config.measurement_schedule in ("independent", "mixed"):
            panel_count = measurement_rng.poisson(config.independent_measurement_rate*followup)
            for position in np.sort(measurement_rng.uniform(0,1,panel_count)):
                date = start + pd.Timedelta(days=float(position*followup*YEAR_DAYS))
                measurements.extend(_measurement_panel(measurement_rng,pid,None,date,dob,"independent"))
        hidden_start = dob + pd.Timedelta(days=config.earliest_study_age*YEAR_DAYS)
        hidden_years = max(0,(start-hidden_start).total_seconds()/(86400*YEAR_DAYS))
        hidden_n = int(rng.poisson(config.pre_study_event_rate*hidden_years))
        hidden_dates = sorted(hidden_start + pd.Timedelta(days=float(p*hidden_years*YEAR_DAYS)) for p in rng.uniform(0,1,hidden_n))
        for j,date in enumerate(hidden_dates):
            hidden.append([pid,f"{pid}_H{j:05d}",date,(date-dob).total_seconds()/(86400*YEAR_DAYS),
                           rng.choice(EVENT_TYPES,p=_probabilities(config.event_type_weights,len(EVENT_TYPES))),
                           np.nan if j == 0 else (date-hidden_dates[j-1]).total_seconds()/86400,np.nan])
        censor_days = (end-(dates[-1] if dates else start)).total_seconds()/86400
        patients.append(dict(patient_id=pid,date_of_birth=dob,age=age,sex=rng.choice(SEX,p=_probabilities(config.sex_weights,2)),
             marital_status=rng.choice(MARITAL,p=_probabilities(config.marital_weights,len(MARITAL))),
             employment=rng.choice(EMPLOYMENT,p=_probabilities(config.employment_weights,3)),
             ehr_history=note,ehr_missing=note is None,ehr_recorded_at=start,
             study_start_date=start,start_study_age=start_age,study_end_date=end,
             censor_age=age,followup_years=followup,n_observed_events=count,right_censor_days=censor_days))
    dataset = SyntheticDataset(pd.DataFrame(patients),pd.DataFrame(events,columns=event_columns),
                            pd.DataFrame(hidden,columns=event_columns),pd.DataFrame(measurements,columns=measurement_columns)
                            .sort_values(["patient_id","measurement_date","variable"]).reset_index(drop=True),config)
    if config.include_dynamic_covariates:
        dataset.dynamic_covariates = _generate_dynamic_covariates(dataset)
    return dataset


def _measurement_panel(rng, patient_id, event_id, date, dob, source):
    """Toy three-variable panel at any timestamp; no visit required."""
    age = (date-dob).total_seconds()/(86400*YEAR_DAYS)
    shock = rng.normal(0,1)
    values = [("systolic_bp",120+0.35*(age-40)+6*shock+rng.normal(0,7),"mmHg"),
              ("glucose",90+0.12*(age-40)+4*shock+rng.normal(0,8),"mg/dL"),
              ("creatinine",0.85+0.002*(age-40)+0.03*shock+rng.normal(0,0.08),"mg/dL")]
    return [[patient_id,event_id,date,age,variable,max(0.01,value),unit,source]
            for variable,value,unit in values]


def _generate_dynamic_covariates(dataset):
    """Independent recorded updates, not visit events or clinically valid transitions.

    Baseline rows match patients. Repeated values are allowed; unknown/missing
    means the value was not documented, not a known real-world state change.
    """
    config = dataset.config
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 2]))
    rows = []
    for patient in dataset.patients.itertuples():
        for variable, rate in [("marital_status",config.marital_update_rate),
                               ("employment",config.employment_update_rate),
                               ("ehr_history",config.ehr_update_rate)]:
            baseline = getattr(patient,variable)
            rows.append([patient.patient_id,patient.study_start_date,patient.start_study_age,
                         variable,baseline,baseline is None or baseline == "unknown/missing","baseline"])
            count = rng.poisson(rate*patient.followup_years)
            for position in np.sort(rng.uniform(0,1,count)):
                date = patient.study_start_date + pd.Timedelta(days=float(position*patient.followup_years*YEAR_DAYS))
                if variable == "marital_status":
                    value = rng.choice(MARITAL,p=_probabilities(config.marital_weights,len(MARITAL)))
                elif variable == "employment":
                    value = rng.choice(EMPLOYMENT,p=_probabilities(config.employment_weights,len(EMPLOYMENT)))
                else:
                    value = _baseline_note(rng,config)
                    if value is not None:
                        value = value.replace("Baseline psychiatric assessment:","Psychiatric history update:")
                rows.append([patient.patient_id,date,
                    (date-patient.date_of_birth).total_seconds()/(86400*YEAR_DAYS),variable,value,
                    value is None or value == "unknown/missing","update"])
    return pd.DataFrame(rows,columns=["patient_id","recorded_at","recorded_age","variable","value","is_missing","source"])\
        .sort_values(["patient_id","recorded_at","variable"]).reset_index(drop=True)
