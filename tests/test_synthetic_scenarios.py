import numpy as np
import pandas as pd
from dataclasses import replace
from lifetime_clinical.data.synthetic import SyntheticConfig, generate_dataset


def test_rich_reproducible_censored_and_oracle_separate(tmp_path):
    config = SyntheticConfig(n_samples=8, scenario='combined', seed=71,
                             measurement_schedule='mixed')
    first, second = generate_dataset(config), generate_dataset(config)
    for name in ('patients', 'events', 'measurements', 'dynamic_covariates', 'oracle_trajectories'):
        pd.testing.assert_frame_equal(getattr(first, name), getattr(second, name))
    for patient in first.patients.itertuples():
        events = first.events[first.events.patient_id == patient.patient_id]
        assert events.event_date.between(patient.study_start_date, patient.study_end_date, inclusive='left').all()
        assert len(events) == patient.n_observed_events
        last = events.event_date.iloc[-1] if len(events) else patient.study_start_date
        assert np.isclose(patient.right_censor_days, (patient.study_end_date-last).total_seconds()/86400)
        if len(events):
            assert pd.isna(events.iat_days.iloc[0])
            assert np.allclose(events.iat_days.iloc[1:], events.event_date.diff().dt.total_seconds().iloc[1:]/86400)
    assert np.isfinite(first.measurements.value).all()
    assert first.oracle_trajectories.groupby(['patient_id', 'variable']).latent_value.std().gt(0).all()
    first.save(tmp_path)
    assert (tmp_path/'oracle_trajectories.csv').exists()


def test_scenario_ablation_and_measurement_streams():
    config = SyntheticConfig(n_samples=6, scenario='structured_physiology', seed=91)
    observed = generate_dataset(config)
    unobserved = generate_dataset(replace(config, include_measurements=False))
    pd.testing.assert_frame_equal(observed.events, unobserved.events)
    pd.testing.assert_frame_equal(observed.oracle_trajectories, unobserved.oracle_trajectories)
    flat = generate_dataset(replace(config, trajectory_strength=0, covariate_strength=0))
    for (_, variable), group in flat.oracle_trajectories.groupby(['patient_id', 'variable']):
        slope = {'systolic_bp': .35, 'glucose': .12, 'creatinine': .002}[variable]
        assert np.allclose(np.diff(group.latent_value), slope*np.diff(group.age))
    simple = generate_dataset(SyntheticConfig(n_samples=3))
    assert simple.oracle_trajectories.empty
