from mpc_execution.simulator import Simulator, SimParams
from mpc_execution.book import Side
import numpy as np
import pytest

def test_seeded_book_is_consistent():
    p = SimParams()
    sim = Simulator(p, seed = 0)
    b = sim.book
    b.check_invariants()
    assert b.best_bid() == p.mid0 - 1
    assert b.best_ask() == p.mid0 + 1
    assert b.spread() == 2
    assert len(b.bids) == p.K - 1 and len(b.asks) == p.K - 1

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
    assert p.K == 3 and p.mu == 0.5 and p.alpha == 1.2

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
        assert rates[d + 2 * K - 1] == pytest.approx(p.theta * len(b.bids[p.mid0 + 1 - d]))
        assert rates[d + 3 * K - 1] == pytest.approx(p.theta * len(b.asks[p.mid0 - 1 + d]))
    assert rates[-1] == rates[-2] == p.mu

def test_cancellation_rate_is_zero_on_an_empty_level():
    p = SimParams()
    sim = Simulator(p, seed=0)
    b = sim.book
    for o in list(b.bids[p.mid0 - 2]):    # the seeded orders at distance 3 (D9: from best ask)
        b.cancel(o.id)
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

def test_default_book_is_alive():
    """The default parameters must give a book whose mid keeps moving after the seed is forgotten."""
    sim = Simulator(SimParams(), seed=3)
    for _ in range(5_000):     # burn-in: spread shrinks to tick and seed is forgotten
        sim._next_event()
    changes, prev = 0, sim.book.mid()
    for _ in range(20_000):
        sim._next_event()
        if sim.book.mid() != prev:
            changes += 1
            prev = sim.book.mid()
    assert changes > 50          # observed ≈ 450 with the chosen defaults

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

def test_step_advances_the_clock_exactly():
    sim = Simulator(SimParams(), seed=0)
    t0 = sim.t
    sim.step(2.5)
    assert sim.t == pytest.approx(t0 + 2.5)
    sim.book.check_invariants()

def test_market_order_executes_immediately():
    sim = Simulator(SimParams(), seed=0)
    ask = sim.book.best_ask()
    residual, execs = sim.place_market(Side.BUY, 10)
    assert sum(q for _, _, q in sim.trades) == 10
    assert residual == 0
    assert sum(e.qty for e in execs) == 10
    assert all(e.side is Side.BUY and not e.passive and e.price >= ask for e in execs)
    assert sim.my_orders == set()

def test_resting_bid_at_the_best_gets_filled_by_the_flow():
    sim = Simulator(SimParams(), seed=0)
    price = sim.book.best_bid()
    oid, execs = sim.place_limit(Side.BUY, price, 20)
    assert oid in sim.my_orders and execs == []
    for _ in range(500):
        execs += sim.step(1.0)
        if oid not in sim.my_orders:
            break
    assert oid not in sim.my_orders, "never filled"
    assert sum(e.qty for e in execs) == 20
    assert all(e.passive and e.side is Side.BUY and e.price == price for e in execs)

def test_resting_bid_far_from_the_best_is_rarely_filled():
    filled = 0
    for seed in range(5):
        sim = Simulator(SimParams(), seed=seed)
        sim.place_limit(Side.BUY, sim.book.best_bid() - 8, 20)
        execs = []
        for _ in range(100):
            execs += sim.step(1.0)
        filled += sum(e.qty for e in execs)
    assert filled < 0.2 * 5 * 20

def test_flow_never_cancels_my_orders():
    sim = Simulator(SimParams(), seed=1)
    oid, _ = sim.place_limit(Side.SELL, sim.book.best_ask() + 2, 30)
    execs = []
    for _ in range(300):
        execs += sim.step(1.0)
    still_resting = oid in sim.my_orders
    executed = sum(e.qty for e in execs)
    assert (still_resting and executed < 30) or (not still_resting and executed == 30)

def test_cancel_mine_removes_the_order():
    sim = Simulator(SimParams(), seed=0)
    oid, _ = sim.place_limit(Side.BUY, sim.book.best_bid() - 1, 15)
    assert sim.cancel_mine(oid) == 15
    assert oid not in sim.my_orders and oid not in sim.book.orders
    with pytest.raises(KeyError):
        sim.cancel_mine(oid)

def test_snapshot_is_consistent_with_the_book():
    sim = Simulator(SimParams(), seed=0)
    sim.step(5.0)
    oid, _ = sim.place_limit(Side.BUY, sim.book.best_bid(), 7)
    s = sim.snapshot()
    b = sim.book
    assert s.best_bid == b.best_bid() and s.best_ask == b.best_ask()
    assert s.bid_depth[0] == sum(o.qty for o in b.bids[s.best_bid])
    assert s.ask_depth[0] == sum(o.qty for o in b.asks[s.best_ask])
    assert -1 <= s.imbalance <= 1
    side, price, qty, ahead = s.my_orders[oid]
    assert side is Side.BUY and price == s.best_bid and qty == 7 and ahead == b.queue_ahead(oid)

def test_market_order_is_recorded_as_trades():
    sim = Simulator(SimParams(), seed=0)
    n0 = len(sim.trades)
    residual, _ = sim.place_market(Side.BUY, 30)
    assert sum(q for _, _, q in sim.trades[n0:]) == 30 - residual