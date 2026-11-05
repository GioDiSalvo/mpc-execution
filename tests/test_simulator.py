from mpc_execution.simulator import Simulator, SimParams
from mpc_execution.book import Side
import numpy as np
import pytest

def test_seeded_book_is_consistent():
    sim = Simulator(SimParams(), seed = 0)
    b = sim.book
    b.check_invariants()
    assert b.best_bid() == SimParams().mid0 - 1
    assert b.best_ask() == SimParams().mid0 + 1
    assert b.spread() == 2
    assert len(b.bids) == SimParams.K - 1 and len(b.asks) == SimParams.K - 1

def test_seeded_depth_decreases_with_distance():
    sim = Simulator(SimParams(), seed = 0)
    b = sim.book
    depths_ask = [sum(o.qty for o in b.asks[p]) for p in sorted(b.asks)]
    assert all(x >= y for x, y in zip(depths_ask, depths_ask[1:]))
    depths_bid = [sum(o.qty for o in b.bids[p]) for p in sorted(b.bids)]
    assert all(x <= y for x, y in zip(depths_bid, depths_bid[1:]))

def test_same_seed_same_book():
    a = Simulator(SimParams(), seed = 7).book
    b = Simulator(SimParams(), seed = 7).book
    assert repr(a) == repr(b)

def test_params_are_overridable():
    p = SimParams(K=3, mu=0.5)
    assert p.K == 3 and p.mu == 0.5 and p.alpha == 1.5

def test_rates_have_the_right_shape_and_values():
    p = SimParams()
    sim = Simulator(p, seed = 0)
    b = sim.book
    rates = sim._rates()
    K = p.K
    assert len(rates) == 4 * p.K + 2
    assert np.all(rates >= 0) and rates.sum() > 0
    for d in range(1, K + 1):
        assert rates[d - 1] == rates[d - 1 + K] == p.lam(d)
        if d == 1:
            assert rates[2 * K] == 0 and rates[3 * K] == 0     # inside the spread: empty
            continue
        qty_bids = sum(o.qty for o in b.bids[p.mid0 + 1 - d])
        qty_asks = sum(o.qty for o in b.asks[p.mid0 - 1 + d])
        assert rates[d + 2 * K - 1] == pytest.approx(p.theta * qty_bids)
        assert rates[d + 3 * K - 1] == pytest.approx(p.theta * qty_asks)
    assert rates[-1] == rates[-2] == p.mu

def test_cancellation_rate_is_zero_on_an_empty_level():
    p = SimParams()
    sim = Simulator(p, seed=0)
    b = sim.book
    oid = b.bids[p.mid0 - 2][0].id        # the single seeded order at distance 3 (D9: from best ask)
    b.cancel(oid)
    rates = sim._rates()
    assert rates[2 * p.K + 2] == 0        # index of "cancel bid, d = 3"
    assert rates[2 * p.K + 1] > 0         # d = 2 untouched

def test_single_event_keeps_book_consistent():
    sim = Simulator(SimParams(), seed=0)
    tau = sim._next_event()
    assert tau > 0 and sim.t == tau
    sim.book.check_invariants()

def test_many_events_keep_book_populated():
    sim = Simulator(SimParams(), seed=1)
    for _ in range(10_000):
        sim._next_event()
        sim.book.check_invariants()
        assert sim.book.best_bid() is not None and sim.book.best_ask() is not None
        assert sim.book.spread() >= 1

def test_mid_moves():
    sim = Simulator(SimParams(), seed=3)
    mids = set()
    for _ in range(20_000):
        sim._next_event()
        mids.add(sim.book.mid())
    assert len(mids) > 1

def test_same_seed_same_trajectory():
    sim1, sim2 = Simulator(SimParams(), seed=7), Simulator(SimParams(), seed=7)
    for _ in range(1000):
        sim1._next_event()
        sim2._next_event()
        assert repr(sim1.book) == repr(sim2.book) and sim1.t == sim2.t

def test_apply_event_touches_the_right_side():
    p = SimParams()
    K = p.K

    sim = Simulator(p, seed=0)
    before = sim._depth(Side.BUY)
    sim._apply_event(0)                       # limit buy, d = 1: inside the spread
    assert sim._depth(Side.BUY) > before
    assert sim.book.best_bid() == p.mid0      # D9: best_ask - 1

    sim = Simulator(p, seed=0)
    before = sim._depth(Side.SELL)
    sim._apply_event(K)                       # limit sell, d = 1
    assert sim._depth(Side.SELL) > before

    sim = Simulator(p, seed=0)
    before = sim._depth(Side.BUY)
    sim._apply_event(2 * K + 1)               # cancel bid, d = 2: the seeded best bid
    assert sim._depth(Side.BUY) < before
    assert sim._depth(Side.SELL) == Simulator(p, seed=0)._depth(Side.SELL)

    sim = Simulator(p, seed=0)
    before = sim._depth(Side.SELL)
    sim._apply_event(3 * K + 1)               # cancel ask, d = 2
    assert sim._depth(Side.SELL) < before

    sim = Simulator(p, seed=0)
    before = sim._depth(Side.SELL)
    sim._apply_event(4 * K)                   # market buy consumes asks
    assert sim._depth(Side.SELL) < before

    sim = Simulator(p, seed=0)
    before = sim._depth(Side.BUY)
    sim._apply_event(4 * K + 1)               # market sell consumes bids
    assert sim._depth(Side.BUY) < before