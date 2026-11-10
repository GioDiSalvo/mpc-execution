import numpy as np
import pytest
from mpc_execution.book import Side
from mpc_execution.simulator import Simulator, SimParams
from mpc_execution.schedule import twap
from mpc_execution.policies import Crossing, OrderSpec
from mpc_execution.experiment import run_order, run_many
from mpc_execution.metrics import summarize

Q, N, DT = 200, 4, 5.0

class FarLimit:
    """Posts one limit order 50 ticks away from the touch: never fills."""
    def decide(self, state, side, q, target, remaining):
        price = state.best_bid - 50 if side == Side.BUY else state.best_ask + 50
        return [OrderSpec("limit", target - q, price)]

@pytest.fixture
def sim():
    s = Simulator(SimParams(), seed=0)
    s.step(50.0)
    return s

def test_crossing_follows_schedule_exactly(sim):
    r = run_order(sim, Crossing(), Side.BUY, Q, twap(Q, N), DT)
    assert np.array_equal(r.schedule, r.filled_path)
    assert r.filled == Q and all(e.side == Side.BUY and not e.passive for e in r.executions)

def test_record_is_consistent(sim):
    t0 = sim.t
    r = run_order(sim, Crossing(), Side.SELL, Q, twap(Q, N), DT)
    assert len(r.mids) == len(r.filled_path) == N + 1
    assert r.filled_path[0] == 0 and r.filled == r.filled_path[-1]
    assert sim.t == pytest.approx(t0 + N * DT)
    assert all(t0 <= t <= sim.t for t, _, _ in r.market_trades)
    assert all (t0 <= ex.t <= sim.t for ex in r.executions)

def test_unfilled_limit_orders_do_not_survive_the_order(sim):
    r = run_order(sim, FarLimit(), Side.BUY, Q, twap(Q, N), DT)
    assert r.filled == 0 and sim.my_orders == set()
    summarize(r)                                    # nan metrics, no exception

def test_run_many_alternates_sides_and_is_reproducible():
    kw = dict(params=SimParams(), policy_factory=Crossing, dt=DT,
              schedule=twap(Q, N), n_orders=4, seed0=7)
    a, b = run_many(**kw), run_many(**kw)
    assert [r.side for r in a] == [Side.BUY, Side.SELL, Side.BUY, Side.SELL]
    assert [summarize(r) for r in a] == [summarize(r) for r in b]