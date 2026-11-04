from mpc_execution.simulator import Simulator, SimParams

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