# Synthetic clinical dataset

This dataset represents observed patient histories, not real clinical records.
`synthetic/` contains 1,300 patients; `toy/` contains 130 patients sampled
deterministically from the richer combined scenario in `generated_synthetic/`. Both use the same schema.

## Tables and variables

| File | Contents |
|---|---|
| `patients.csv` | One row per patient: birth date, baseline demographics/context and observation-window metadata |
| `events.csv` | Observed visit times, ages, event-type marks and waiting-time metadata |
| `measurements.csv` | Irregular recorded physiological values, with time, age, variable and unit |
| `dynamic_covariates.csv` | Timestamped baseline records and later context updates |
| `pre_study_events.csv` | Hidden simulated events before entry; oracle-only, excluded from observed model inputs |
| `metadata.json` | Generation settings and provenance |
| `splits.json` | Disjoint training, validation and test patient IDs |

Physiology comprises **systolic blood pressure** (`systolic_bp`, mmHg),
**glucose** (`glucose`, mg/dL) and **creatinine** (`creatinine`, mg/dL).
Measurement rows identify their source and may have an associated `event_id`;
measurement times need not coincide with visit times or form a regular grid.

## Baseline and changing context

The patient table stores baseline sex, marital status, employment and EHR text
(`ehr_history`), plus missing-text flags. Date of birth and entry age define the
age clock. Marital status, employment and EHR information can have later
timestamped updates in `dynamic_covariates.csv`; a baseline field being fixed
in a model does not mean it cannot change in life.

TPP-A holds its baseline vector fixed. TPP-B/C use the latest available
physiology and changing context at visit origins, including missingness and
staleness. Their current interval parameters stay fixed until the next visit.
The GPs use a fixed causal prefix context for their function prior and condition
on available measurements. The lifetime prototype shares a causal encoder
between its visit branch and physiology branch, with its assumptions documented
in the notebook.

`age`/`censor_age`, study end, follow-up duration, total event count and terminal
censoring gaps describe the observation window. They are **not future predictor
inputs**. Model ages are commonly scaled by 100; IATs are measured in days.
Plots convert time to patient age or years as labelled.

## Timelines and event marks

Each patient has a date of birth, a study-entry date and a study-end date.
Observed visits lie within that window. Marks are **outpatient visit**,
**emergency visit** and **inpatient visit** (IDs 0, 1 and 2 respectively).
`event_age` places visits on the patient's age timeline. The first delay from
entry is distinct from a full inter-visit interval whose preceding event is
known. No recorded event after the final visit and before study end contributes
a right-censoring survival term, not an invented event.

Pre-entry history is unknown to the observed-data models. The hidden
`pre_study_events.csv` table must not be used as training history. Measurements
and context updates have their own timestamps; only information available by
the prediction origin enters a causal history. Missing physiological targets
are excluded from value likelihoods; carried-forward context values are not
scored as additional measurements. Masks and time since the last record expose
missingness and staleness without filling a regular training grid.

## Cohort fitting and evaluation

The full split contains **1,000 training / 200 validation / 100 test** patients.
The toy split contains **100 / 20 / 10**. Every eligible training patient contributes
to fitting. Some model-specific windows require complete physiology or enough
measurement panels, so eligible counts can be smaller; notebooks print them.
Validation selects checkpoints and test patients remain outside fitting.
Preprocessing scales are learned only from the training split.

Plots of one patient's held-out future illustrate predictions from a
cohort-fitted model; they are not per-patient model training. A plot's artificial
conditioning/held-out cutoff is distinct from the actual end-of-follow-up
censoring time.

The generator is `lifetime_clinical/data/synthetic.py`; the supplied generation
configuration is `configs/synthetic_data.json`. See the [main README](../README.md)
for generation commands.

## Controlled generation scenarios

The default CLI configuration now selects `combined`; the full `synthetic/` corpus remains unchanged. `toy/` now uses a 130-patient
subset of the combined scenario. New output is written to `data/generated_synthetic/`.
Use `configs/synthetic_data_simple.json` to reproduce the original simple scenario.

| `scenario` | Mechanisms |
|---|---|
| `simple` | Original count-conditioned visits and linear age/Gaussian panels |
| `dynamic_events` | Sequential mark-dependent waiting times driven by persistent context, previous marks and available measurements |
| `structured_physiology` | Patient-specific baselines/slopes, smooth shared and private oscillations, persistent covariate effects |
| `flexible_observations` | Centred skew/tail residual components with context-dependent scale |
| `combined` | All three mechanisms together |

Rich scenarios use `event_rate_per_year`, rather than `mean_events`,
`count_distribution`, `count_dispersion` or `timeline_beta` (simple-only options).
A patient-specific rate multiplier adds heterogeneity. At each visit origin,
context and previous observed physiology determine mark probabilities and a
mark-specific Weibull waiting time. Interval parameters remain fixed until the
next visit: intervening updates inform the following interval. Longer exposure
produces more events on average; follow-up ends administratively.

Employment and marital records follow persistent states, changing to a different
category at each update. Missing/unknown remains an observed documentation state.
EHR text stays fixed in rich scenarios; `ehr_update_rate` applies only to simple.
These associations are invented benchmark mechanisms, not clinical claims.

Physiology combines the age trend, patient intercept/slope, three smooth shared
factors with cross-feature loadings and feature-private fluctuations. Covariate
effects are piecewise constant, so transitions can introduce jumps. Residuals
are independent between panels, with Gaussian, centred skew and finite-variance
Student-t components. Positive clipping can change their mean near zero.
`trajectory_strength`, `covariate_strength`, `history_strength` and
`residual_strength` control ablations; zero removes the corresponding addition.
Measurement scheduling has a separate random stream; changing it can change
subsequent visits in dynamic scenarios because observed measurements drive risk.

`oracle_trajectories.csv` records noiseless physiology on a twice-monthly grid
solely for simulation evaluation. Like hidden pre-entry events, it must never
enter training inputs. Model loaders ignore this extra table. A new dataset
requires new disjoint patient splits before fitting; existing notebooks continue
to use `data/synthetic/` or `data/toy/` until explicitly switched. Model superiority
is an evaluation question, not guaranteed by these scenarios.

Prepare or refresh the rich toy subset with `python scripts/prepare_toy_data.py`.
The existing full split defines patient memberships; existing toy IDs are retained
before adding patients, without selecting by model performance. Fast mode trains
for five epochs.
