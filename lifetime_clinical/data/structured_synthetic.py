"""Controlled synthetic benchmarks; mechanisms are illustrative, not clinical laws."""
from dataclasses import replace
import numpy as np
import pandas as pd
from .synthetic import (generate_dataset, YEAR_DAYS, EVENT_TYPES, MARITAL, EMPLOYMENT,
                        _probabilities)

VARIABLES = ('systolic_bp', 'glucose', 'creatinine')
UNITS = ('mmHg', 'mg/dL', 'mg/dL')
SCALES = np.array([10., 12., .12])


def generate_structured_dataset(config):
    # Preserve baseline population, hidden history and table contracts.
    dataset = generate_dataset(replace(config, scenario='simple', include_measurements=False,
                                      include_dynamic_covariates=False))
    dataset.config = config
    events, measurements, updates, oracle = [], [], [], []
    dynamic = config.scenario in ('dynamic_events', 'combined')
    structured = config.scenario in ('structured_physiology', 'combined')
    flexible = config.scenario in ('flexible_observations', 'combined')
    for i, patient in enumerate(dataset.patients.itertuples()):
        # Mechanism streams are independent; measurement schedule cannot change
        # baseline population, trajectories or covariate transitions.
        latent_rng, cov_rng, event_rng, obs_rng = [np.random.default_rng(
            np.random.SeedSequence([config.seed, 10 + j, i])) for j in range(4)]
        start, end, dob = patient.study_start_date, patient.study_end_date, patient.date_of_birth
        duration = patient.followup_years
        baseline = np.array([120., 90., .85])
        age_slopes = np.array([.35, .12, .002])
        intercept = latent_rng.normal(0, .7, 3)
        slope = latent_rng.normal(0, .08, 3)
        phases = latent_rng.uniform(0, 2*np.pi, 3)
        periods = latent_rng.uniform(2., 6., 3)
        private_phases = latent_rng.uniform(0, 2*np.pi, 3)
        cov_records = {v: [(0., getattr(patient, v))] for v in
                       ('employment', 'marital_status', 'ehr_history')}
        for variable, rate, choices, weights in (
                ('employment', config.employment_update_rate, EMPLOYMENT, config.employment_weights),
                ('marital_status', config.marital_update_rate, MARITAL, config.marital_weights)):
            if config.include_dynamic_covariates:
                times = np.sort(cov_rng.uniform(0, duration, cov_rng.poisson(rate*duration)))
                previous = cov_records[variable][0][1]
                for t in times:
                    # A persistent state with genuine transitions rather than iid redraws.
                    probabilities = _probabilities(weights, len(choices)).copy()
                    probabilities[np.array(choices) == previous] = 0
                    value = cov_rng.choice(choices, p=probabilities/probabilities.sum())
                    cov_records[variable].append((float(t), value))
                    previous = value
            if config.include_dynamic_covariates:
                for t, value in cov_records[variable]:
                    date = start + pd.Timedelta(days=t*YEAR_DAYS)
                    updates.append([patient.patient_id, date, patient.start_study_age+t,
                                    variable, value, value == 'unknown/missing',
                                    'baseline' if t == 0 else 'update'])
        if config.include_dynamic_covariates:
            # Keep text fixed: random text updates would add no learnable signal.
            updates.append([patient.patient_id, start, patient.start_study_age, 'ehr_history',
                            patient.ehr_history, patient.ehr_history is None, 'baseline'])

        def covariate(t):
            employment = next(value for time, value in reversed(cov_records['employment']) if time <= t)
            marital = next(value for time, value in reversed(cov_records['marital_status']) if time <= t)
            return float(employment == 'unemployed') + .4*float(marital in ('separated', 'divorced'))

        def physiology(t):
            age = patient.start_study_age+t
            mean = baseline + age_slopes*(age-40)
            if structured:
                # Shared smooth factors plus private fluctuations at arbitrary times.
                shared = np.sin(2*np.pi*t/periods + phases)
                loadings = np.array([[.8, .25, 0], [.55, -.3, .25], [.3, .15, .55]])
                private = .25*np.sin(2*np.pi*t/(periods*.7)+private_phases)
                mean = mean + config.trajectory_strength*SCALES*(intercept+slope*t+loadings@shared+private)
                mean = mean + config.covariate_strength*SCALES*.35*covariate(t)
            return mean

        def panel(t, eid, source):
            mean = physiology(t)
            noise = obs_rng.normal(size=3)*np.array([.5, .55, .45])
            if flexible:
                # Centred skew-normal component plus finite-variance t tails.
                skew = np.abs(obs_rng.normal(size=3))-np.sqrt(2/np.pi)
                heavy = obs_rng.standard_t(5, size=3)/np.sqrt(5/3)
                rich_noise = ((.25*noise+.35*skew+.35*heavy)*
                              (1+.6*covariate(t)+.3*(1+np.sin(t))))
                weight = config.residual_strength/(1+config.residual_strength)
                noise = (1-weight)*noise+weight*rich_noise
            value = np.maximum(.01, mean+SCALES*noise)
            date = start + pd.Timedelta(days=float(t*YEAR_DAYS))
            for d, variable in enumerate(VARIABLES):
                measurements.append([patient.patient_id, eid, date, patient.start_study_age+t,
                                     variable, value[d], UNITS[d], source])
            return value

        # Independent panels are generated before visits, but only past values
        # can drive an interval. Scheduled future panels never enter its context.
        independent = []
        if config.include_measurements and config.measurement_schedule in ('independent', 'mixed'):
            times = np.sort(obs_rng.uniform(0, duration, obs_rng.poisson(config.independent_measurement_rate*duration)))
            independent = [(float(t), panel(float(t), None, 'independent')) for t in times]
        latest = None
        independent_index = 0
        t, previous_mark, visit_count = 0., None, 0
        frailty = event_rng.lognormal(-config.patient_rate_sigma**2/2, config.patient_rate_sigma)
        probabilities = _probabilities(config.event_type_weights, 3)
        while config.event_rate_per_year > 0:
            while independent_index < len(independent) and independent[independent_index][0] <= t:
                pt, pv = independent[independent_index]
                if latest is None or pt > latest[0]:
                    latest = (pt, pv)
                independent_index += 1
            risk = covariate(t)*config.covariate_strength if dynamic else 0.
            if dynamic and latest is not None:
                age_mean = baseline + age_slopes*(patient.start_study_age+latest[0]-40)
                risk += .3*float(np.mean((latest[1]-age_mean)/SCALES))
            history = config.history_strength*(previous_mark is not None and previous_mark != 0) if dynamic else 0.
            logits = np.log(np.where(probabilities > 0, probabilities, 1.)) + np.array([-.3, .5, .8])*(risk+.5*history)
            mark_prob = np.exp(logits-logits.max())* (probabilities > 0)
            mark_prob /= mark_prob.sum()
            k = int(event_rng.choice(3, p=mark_prob))
            # Unit-mean Weibull waiting factors have distinct shapes/scales by mark.
            shapes = np.array([1.25, .85, 1.6]) if dynamic else np.ones(3)
            mark_scales = np.array([1.15, .5, .75]) if dynamic else np.ones(3)
            from math import gamma
            factor = event_rng.weibull(shapes[k])/gamma(1+1/shapes[k])
            rate = config.event_rate_per_year*frailty*np.exp(np.clip(.35*risk+.3*history, -1.5, 1.5))
            delay = max(1e-6, float(factor*mark_scales[k]/rate))
            next_t = t+delay
            if next_t >= duration:
                break  # administrative censoring, no invented final event
            date = start + pd.Timedelta(days=next_t*YEAR_DAYS)
            eid = f'{patient.patient_id}_E{visit_count:05d}'
            events.append([patient.patient_id, eid, date, patient.start_study_age+next_t,
                           EVENT_TYPES[k], np.nan if visit_count == 0 else delay*YEAR_DAYS,
                           next_t*YEAR_DAYS])
            if config.include_measurements and config.measurement_schedule in ('event_linked', 'mixed') and obs_rng.random() < config.measurement_probability:
                latest = (next_t, panel(next_t, eid, 'event_linked'))
            t, previous_mark, visit_count = next_t, k, visit_count+1
        dataset.patients.loc[i, 'n_observed_events'] = visit_count
        dataset.patients.loc[i, 'right_censor_days'] = (duration-t)*YEAR_DAYS
        for time in np.linspace(0, duration, max(2, int(duration*24)+1)):
            for d, variable in enumerate(VARIABLES):
                oracle.append([patient.patient_id, start+pd.Timedelta(days=float(time*YEAR_DAYS)),
                               patient.start_study_age+time, variable, physiology(time)[d]])
    dataset.events = pd.DataFrame(events, columns=dataset.events.columns)
    dataset.measurements = pd.DataFrame(measurements, columns=dataset.measurements.columns).sort_values(
        ['patient_id', 'measurement_date', 'variable']).reset_index(drop=True)
    dataset.dynamic_covariates = pd.DataFrame(updates, columns=dataset.dynamic_covariates.columns).sort_values(
        ['patient_id', 'recorded_at', 'variable']).reset_index(drop=True)
    dataset.oracle_trajectories = pd.DataFrame(oracle, columns=[
        'patient_id', 'date', 'age', 'variable', 'latent_value'])
    return dataset
