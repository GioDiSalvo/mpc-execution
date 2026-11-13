import numpy as np
import pytest
from mpc_execution.book import Side
from mpc_execution.simulator import Simulator, SimParams
from mpc_execution.fills import ConstantFill, candidate_prices, collect_fill_samples

N_LEVELS, QTY, DT, N_TRIALS, N_TRIALS_STATS = 3, 25, 5.0, 20, 300

def test_candidate_prices_works_correctly():
    sim = Simulator(SimParams(), seed=0)
    state = sim.snapshot()
    b, a = state.best_bid, state.best_ask
    assert candidate_prices(state, Side.BUY, 3).tolist() == [b, b - 1, b - 2]
    assert candidate_prices(state, Side.SELL, 3).tolist() == [a, a + 1, a + 2]
    assert candidate_prices(state, Side.BUY, 3).dtype.kind == "i"

def test_constant_fill_is_actually_constant():
    target_pi = np.array([0.5, 0.2])
    const_fill = ConstantFill(target_pi)
    for i in range(N_TRIALS):
            sim = Simulator(SimParams(), seed=i)
            state = sim.snapshot()
            side = Side.BUY if i%2==0 else Side.SELL
            pi = const_fill.probs(state, side, QTY)
            assert np.array_equal(pi, target_pi)
    with pytest.raises(ValueError):
         ConstantFill(np.array([1.2]))

def test_collect_fill_samples_has_right_entries():
    df = collect_fill_samples(SimParams(), N_LEVELS, QTY, DT, N_TRIALS, seed=0)
    assert len(df) == 60
    assert list(df.columns) == ["trial", "side", "level", "price", "filled_frac",
                              "spread", "imbalance",
                              "depth_at_level", "shares_ahead",]
    assert df.filled_frac.between(0,1).all()
    first = df.groupby("trial").side.first()
    assert df.groupby("trial").side.nunique().eq(1).all()
    assert first.tolist() == ["BUY" if t % 2 == 0 else "SELL" for t in range(N_TRIALS)]
    wide = df.pivot(index="trial", columns="level", values="filled_frac")
    assert (wide.diff(axis=1).fillna(0) <= 0).all().all()
    df1 = collect_fill_samples(SimParams(), N_LEVELS, QTY, DT, N_TRIALS, seed=0)
    assert df.equals(df1)

def test_fill_probability_is_highest_at_the_touch():
     df = collect_fill_samples(SimParams(), N_LEVELS, QTY, DT, N_TRIALS_STATS, seed=0)
     pi = df.groupby("level").filled_frac.mean()
     assert pi.iloc[0] > pi.iloc[-1] + .1
     assert 0 < pi.iloc[0] < 1

def test_constant_fill_is_non_increasing():
     samples = collect_fill_samples(SimParams(), N_LEVELS, QTY, DT, N_TRIALS, seed=0)
     model = ConstantFill.from_samples(samples)
     expected = samples.groupby("level").filled_frac.mean().to_numpy()
     assert model.pi.shape == (N_LEVELS,)
     assert np.allclose(model.pi, expected)
     assert np.all(np.diff(model.pi) <= 0)