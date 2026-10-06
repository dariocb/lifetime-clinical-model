from pathlib import Path
import numpy as np

from lifetime_clinical.tutorials.experiments import (
    load_toy, tpp_data, gp_data, fit_tpp_a, fit_tpp_b, fit_tpp_c,
    fit_gp_a, fit_gp_b, fit_gp_c, representative_tpp_sequence,
    representative_gp_block,
)
from lifetime_clinical.data.tutorial import physiology_feature_indices

ROOT = Path(__file__).resolve().parents[1]


def test_data_contract_and_toy_split():
    dataset, splits, pre = load_toy(ROOT / "data" / "toy")
    assert len(dataset.patients) == 130
    assert {k: len(v) for k, v in splits.items()} == {"train": 100, "validation": 20, "test": 10}
    assert set(dataset.measurements.variable.unique()) == {"systolic_bp", "glucose", "creatinine"}
    assert len(physiology_feature_indices(pre)) == 3


def test_tpp_models_fit_and_predict():
    dataset, splits, pre = load_toy(ROOT / "data" / "toy")
    static = tpp_data(dataset, splits, pre, dynamic=False)
    dynamic = tpp_data(dataset, splits, pre, dynamic=True)
    models = [fit_tpp_a(static, epochs=1), fit_tpp_b(dynamic, pre, epochs=1), fit_tpp_c(dynamic, pre, epochs=1)]
    seqs = [representative_tpp_sequence(static["test"], 6), representative_tpp_sequence(dynamic["test"], 6), representative_tpp_sequence(dynamic["test"], 6)]
    for model, seq in zip(models, seqs):
        if getattr(model, "input_mode", "static") == "dynamic":
            j = len(seq.contexts) - 1
            history = {"features": seq.contexts[:j], "iats": np.diff(np.r_[0., seq.times])[:j]}
            inputs = seq.contexts[-1]
        else:
            history = list(np.diff(np.r_[0., seq.times]))
            inputs = seq.static
        pred = model.next_event_distribution(inputs, np.array([30., 180., 365.]), history=history)
        assert np.isfinite(pred["survival"]).all()
        assert np.all((pred["survival"] >= 0) & (pred["survival"] <= 1))


def test_gp_models_fit_and_predict():
    dataset, splits, pre = load_toy(ROOT / "data" / "toy")
    blocks = gp_data(dataset, splits, pre, .25)
    conditioning, heldout = representative_gp_block(blocks["test"], 4)
    age = np.linspace(conditioning.ages.min(), heldout.ages.max(), 12)
    variables = np.zeros(len(age), dtype=int)
    for model in [fit_gp_a(blocks, 1), fit_gp_b(blocks, 1), fit_gp_c(blocks, 1)]:
        forecast = model.forecast(conditioning, age, variables, retrospective=True)
        samples = forecast.sample(16, seed=1)
        assert samples.shape == (16, len(age))
        assert np.isfinite(samples).all()


def test_notebook_layout():
    expected = {
        "notebooks/introduction/01_temporal_point_processes.ipynb",
        "notebooks/introduction/02_gaussian_processes.ipynb",
        "notebooks/introduction/03_transformers.ipynb",
        "notebooks/introduction/04_normalizing_flows.ipynb",
        "notebooks/data/01_synthetic_clinical_data.ipynb",
        "notebooks/models/00_baselines.ipynb",
        "notebooks/models/01_tpp_models.ipynb",
        "notebooks/models/02_gp_models_1_2.ipynb",
        "notebooks/models/03_gp_model_3.ipynb",
        "notebooks/lifetime_model/01_lifetime_model_formulation.ipynb",
    }
    found = {p.relative_to(ROOT).as_posix() for p in (ROOT / "notebooks").rglob("*.ipynb")}
    assert found == expected
