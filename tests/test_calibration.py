from mpc_execution.calibration import brier, log_loss, reliability_table
import numpy as np
import pytest
import math

LOG2 = math.log(2)
CASES = [
    # name,                         y,            p,              expected (brier, log_loss)
    ("perfect",                     [1, 0, 1, 0], [1, 0, 1, 0],                0.0,   0.0),
    ("coin",                        [1,0],        [0.5, 0.5],                  0.25,  LOG2),
    ("constant_predictor",          [1, 1, 0, 0], [0.5, 0.5, 0.5, 0.5],        0.25,  LOG2),
    ("fractional_outcome",          [0.5],        [0.5],                       0.0,   LOG2),
]

@pytest.mark.parametrize("name, y, p, want_brier, want_log_loss", CASES, ids = [c[0] for c in CASES])
def test_scores(name, y, p, want_brier, want_log_loss):
    assert brier(y, p) == pytest.approx(want_brier, abs=1e-9)
    assert log_loss(y, p) == pytest.approx(want_log_loss, abs=1e-9)

def simulate(power, n=20_000, seed=0):
    """p uniform; y Bernoulli(p ** power): power 1 is a calibrated model, power 2 is not."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(size=n)
    y = (rng.uniform(size=n) < p ** power).astype(float)
    return y, p

def test_reliability_of_a_calibrated_model():
    y, p = simulate(power=1)
    t = reliability_table(y, p, n_bins=10)
    assert list(t.columns) == ["predicted", "observed", "sem", "count"]
    assert (t["count"] == 2_000).all()
    assert t.predicted.is_monotonic_increasing
    assert t.predicted.iloc[0] < 0.1 and t.predicted.iloc[-1] > 0.9      # bins really are bins of p
    assert ((t.predicted - t.observed).abs() < 4 * t["sem"]).all()

def test_reliability_shows_miscalibration():
    y, p = simulate(power=2)
    middle = reliability_table(y, p, n_bins=10).iloc[3:7]
    assert (middle.observed < middle.predicted - 0.1).all()

def test_log_loss_punishes_a_confident_wrong_prediction():
    assert log_loss([1], [0]) == pytest.approx(-math.log(1e-12))