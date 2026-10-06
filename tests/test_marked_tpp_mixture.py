import numpy as np
import pytest
import torch
from scipy.special import logsumexp
from scipy.stats import lognorm

from lifetime_clinical.data.baselines import TPPSequence
from lifetime_clinical.models.tpp.conditional_flow import ConditionalFlowTPP
from lifetime_clinical.models.tpp.joint_flow import JointFlowTPP
from lifetime_clinical.models.tpp.shared_private_flow import SharedPrivateFlowTPP

SHIFTS = np.array([-.7, 0., .9])
WEIGHTS = np.array([.6, .3, .1])
SIGMA = .6


def controlled_model(cls=ConditionalFlowTPP, n_marks=3):
    kwargs = dict(input_dim=5, hidden_dim=8, n_marks=n_marks, flow_layers=1,
                  transformer_layers=1)
    if cls != ConditionalFlowTPP:
        kwargs.update(feature_indices=(0, 1, 2), input_mode="dynamic", feature_layers=1)
    if cls == SharedPrivateFlowTPP:
        kwargs.update(shared_dim=2)
    model = cls(**kwargs)
    # Exact mark-specific lognormal components make the expected answer independent
    # of implementation details and expose accidental timing/mark independence.
    with torch.no_grad():
        model.net[-1].weight.zero_()
        model.net[-1].bias.zero_()
        for k in range(n_marks):
            model.net[-1].bias[4*k] = SHIFTS[k]
            model.net[-1].bias[4*k+1] = np.arctanh(np.log(SIGMA)/.8)
        model.net[-1].bias[4*n_marks:] = torch.tensor(np.log(WEIGHTS[:n_marks]), dtype=torch.float64)
    return model


@pytest.mark.parametrize("cls", [ConditionalFlowTPP, JointFlowTPP, SharedPrivateFlowTPP])
def test_mixture_forecasts_and_censoring(cls):
    model = controlled_model(cls)
    inputs = np.zeros(5)
    history = () if cls == ConditionalFlowTPP else {}
    elapsed = 120.
    horizons = np.array([0., 20., 200., 700.])
    distributions = lognorm(s=SIGMA, scale=model.time_scale*np.exp(SHIFTS))
    origin_sf = distributions.sf(elapsed)
    sf = distributions.sf((elapsed+horizons)[:, None])
    pdf = distributions.pdf((elapsed+horizons)[:, None])
    norm = np.dot(WEIGHTS, origin_sf)
    pred = model.next_event_distribution(inputs, horizons, history, elapsed)
    np.testing.assert_allclose(pred["survival"], sf@WEIGHTS/norm, rtol=1e-11)
    np.testing.assert_allclose(pred["marked_density"], pdf*WEIGHTS/norm, rtol=1e-11)
    np.testing.assert_allclose(pred["density"], pred["marked_density"].sum(-1))
    np.testing.assert_allclose(pred["hazard"], pred["density"]/pred["survival"])
    np.testing.assert_allclose(pred["marked_hazard"].sum(-1), pred["hazard"])
    np.testing.assert_allclose(pred["mark_probabilities"], WEIGHTS*origin_sf/norm)
    np.testing.assert_allclose(pred["mark_probabilities_at_event"], pdf*WEIGHTS/(pdf@WEIGHTS)[:, None])
    assert not np.allclose(pred["mark_probabilities_at_event"][1], pred["mark_probabilities_at_event"][-1])

    seq = TPPSequence("patient", inputs, np.zeros((2, 5)),
                      np.array([100.]), np.array([1]), 500.)
    expected = np.log(WEIGHTS[1]) + distributions.logpdf(100.)[1]
    expected += logsumexp(np.log(WEIGHTS) + distributions.logsf(400.))
    torch.testing.assert_close(model.conditional_nll([seq]),
                               torch.tensor(-expected, dtype=torch.float64))
    if cls != ConditionalFlowTPP:
        feature_ll = model.feature_log_prob(inputs, {}).detach()
        feature_ll += model.feature_log_prob(inputs, {"features": seq.contexts[:1], "iats": np.array([100.])}).detach()
        torch.testing.assert_close(model.nll([seq]), model.conditional_nll([seq])-feature_ll)
    model.nll([seq]).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_joint_sampling_and_marginal_quantiles():
    model = controlled_model()
    inputs = np.zeros(5)
    waits, marks = model.sample_next_event(inputs, n_samples=15000, seed=4)
    np.testing.assert_allclose(np.bincount(marks, minlength=3)/len(marks), WEIGHTS, atol=.012)
    for k in range(3):
        assert abs(np.log(waits[marks == k]/model.time_scale).mean()-SHIFTS[k]) < .04
    elapsed = 250.
    waits, marks = model.sample_next_event(inputs, n_samples=15000, elapsed_days=elapsed, seed=5)
    pred = model.next_event_distribution(inputs, [200.], elapsed_days=elapsed)
    np.testing.assert_allclose(np.bincount(marks, minlength=3)/len(marks), pred["mark_probabilities"], atol=.012)
    assert abs(np.mean(waits>200.)-pred["survival"][0]) < .012
    probabilities = np.array([.01, .5, .99])
    quantiles = model.quantile(probabilities, inputs, elapsed_days=elapsed)
    np.testing.assert_allclose(model.next_event_distribution(inputs, quantiles, elapsed_days=elapsed)["cdf"],
                               probabilities, atol=1e-10)


def test_unmarked_case_and_empty_window():
    model = controlled_model(n_marks=1)
    inputs = np.zeros(5)
    empty = TPPSequence("empty", inputs, np.zeros((1, 5)),
                        np.array([]), np.array([], dtype=int), 500.)
    expected = lognorm(s=SIGMA, scale=model.time_scale*np.exp(SHIFTS[0]))
    np.testing.assert_allclose(model.score([empty]), -expected.logsf(500.))
    np.testing.assert_allclose(model.iat_survival([0., 100.], inputs).detach().numpy(), expected.sf([0., 100.]))
    np.testing.assert_allclose(model.quantile([.1, .9], inputs), expected.ppf([.1, .9]))
    pred = model.next_event_distribution(inputs, [0., 100.])
    assert np.isfinite(pred["mark_probabilities_at_event"]).all()
    np.testing.assert_allclose(pred["density"], expected.pdf([0., 100.]))
