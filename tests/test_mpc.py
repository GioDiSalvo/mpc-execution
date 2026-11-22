from mpc_execution.book import Side
from mpc_execution.simulator import Simulator, SimParams
from mpc_execution.mpc import MPCParams, MPC, nested_cov, candidate_costs, solve_qp, _round_to_shares
from mpc_execution.fills import ConstantFill, LogisticFill, candidate_prices, collect_fill_samples
from mpc_execution.policies import OrderSpec, ThresholdRule
from mpc_execution.experiment import run_order
from mpc_execution.schedule import twap
import numpy as np
import pytest

PI, C = np.array([1.0, 0.5]), np.array([0.5, -0.5])
TOL = 1e-3
FILL = ConstantFill(np.array([0.6, 0.1, 0.03]))
CASES = [
    # name,          param overrides,            call overrides,                             (u0, u1)
    ("a_wide_tubes", {},                         {},                                         (0, 100)),
    ("b_upper_tube", dict(rho_up=10),            {},                                         (0, 35)),
    ("c_lower_tube", dict(rho_up=0, rho_low=30), dict(gap=100),                              (70, 30)),
    ("d_last_step",  {},                         dict(gap=40, remaining=40, last_step=True), (40, 0)),
    ("e_variance",   dict(beta=25),              {},                                         (20, 10)),
    ("f_ahead",      dict(rho_up=5),             dict(gap=-10),                              (0, 0)),
    ("g_kappa",      {},                         dict(kappa=40),                             (5, 40)),
]

@pytest.fixture
def state():
    sim = Simulator(SimParams(), seed=0)
    sim.step(50.0)
    return sim.snapshot()

@pytest.fixture(scope="module")
def logistic():
    samples = collect_fill_samples(SimParams(), n_levels=3, qty=25, dt=5.0, n_trials=500)
    return LogisticFill.from_samples(samples)

@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_candidate_costs(state, side):
    n_levels = SimParams().K
    limit_prices = candidate_prices(state, side, n_levels)
    costs = candidate_costs(state, side, limit_prices) # n_levels + 1
    expected_costs = np.array([0.5] + [-0.5 - i / state.spread for i in range(n_levels)])
    assert np.allclose(costs, expected_costs)

def params(**over):
    base = dict(n_levels=1, gamma=0.02, xi=0.5, rho_up=1e6, rho_low=1e6, beta=None)
    return MPCParams(**(base | over))

@pytest.mark.parametrize("name,over,call,expected", CASES, ids=[c[0] for c in CASES])
def test_solve_qp(name, over, call, expected):
    args = dict(gap=25, remaining=1e6, last_step=False) | call
    u = solve_qp(PI, C, p=params(**over), **args)
    assert u == pytest.approx(expected, abs=TOL)

def test_infeasible_problem_raises():
    with pytest.raises(RuntimeError):
        solve_qp(PI, C, gap=400, remaining=1e6, last_step=False, p=params(rho_low=50), kappa=250)

def test_nested_cov():
    pi = np.array([1, 0.6, 0.1])
    S = nested_cov(pi)
    assert np.all(S[0] == 0) and np.all(S[:, 0] == 0)
    assert S.diagonal() == pytest.approx(pi * (1 - pi))
    assert S[1, 2] == pytest.approx(0.1 - 0.06)

def test_round_to_shares():
    assert np.array_equal(_round_to_shares(np.array([9.6, 0.4]), 10), [10, 0])            # nothing to fix
    assert np.array_equal(_round_to_shares(np.array([8.6, 0.7, 0.7, 0.0]), 10), [9, 0, 1, 0])   # 11 -> 10
    assert np.array_equal(_round_to_shares(np.array([1.5, 1.5, 1.5, 1.5]), 6), [2, 0, 2, 2])   # 8 -> 6

@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_mpc_orders_are_well_formed(state, side):
    orders = MPC(FILL).decide(state, side, q=100, target=125, remaining=400)
    assert orders                                                           # behind schedule: it acts                                                      
    assert all(isinstance(o.qty, np.int64) and o.qty > 0 for o in orders)
    assert sum(o.qty for o in orders) <= 25 + MPCParams().rho_up            # upper tube
    assert sum(o.kind == "market" for o in orders) <= 1
    allowed = set(candidate_prices(state, side, 3).tolist())
    assert all(o.price in allowed for o in orders if o.kind == "limit")

def test_mpc_last_step_sends_the_whole_gap_to_market(state):
    assert MPC(FILL).decide(state, Side.BUY, q=475, target=500, remaining=25) == [OrderSpec("market", 25)]

def test_mpc_never_sends_more_than_remaining(state):
    wide = MPCParams(rho_up=1e6, rho_low=1e6)
    orders = MPC(FILL, wide).decide(state, Side.SELL, q=490, target=495, remaining=10)
    assert sum(o.qty for o in orders) <= 10

@pytest.mark.parametrize("target", [495, 500, 505])
def test_mpc_does_nothing_when_the_order_is_complete(state, target):
    assert MPC(FILL).decide(state, Side.BUY, q=500, target=target, remaining=0) == []

def test_mpc_falls_back_to_crossing_when_the_qp_is_infeasible(state):
    policy = MPC(FILL, MPCParams(kappa_frac=0.1))
    order = policy.decide(state, Side.BUY, q=0, target=400, remaining=500)
    # Q = q + remaining = 500, u[t] <= kappa * Q = 50, u[0] >= gap - rho_low = 400 - 50 = 350 ---> QP unfeasible ---> Runtime Error
    assert order == [OrderSpec("market", 400)] and policy.n_fallbacks == 1

def test_mpc_completes_an_order_with_passive_fills():
    """End to end: book, order flow, schedule, fill model, QP and execution loop on one 500-share order."""
    sim = Simulator(SimParams(), seed=0)
    sim.step(50.0)
    policy = MPC(FILL)
    r = run_order(sim, policy, side=Side.BUY, Q=500, schedule=twap(500, 20), dt=5.0)
    assert r.filled == 500 and sim.my_orders == set()
    assert policy.n_fallbacks == 0
    assert any(e.passive for e in r.executions)

class Compare:
    """Runs policy `a` and records, at every step, whether policy `b` would have sent the same orders."""
    def __init__(self, a, b):
        self.a, self.b, self.same = a, b, []
    def decide(self, state, side, q, target, remaining):
        orders = self.a.decide(state, side, q, target, remaining)
        self.same.append(orders == self.b.decide(state, side, q, target, remaining))
        return orders

@pytest.mark.parametrize("seed, side", [(0, Side.BUY), (1, Side.SELL), (2, Side.BUY)])
def test_mpc_with_constant_fills_is_the_threshold_rule(seed, side):
    sim = Simulator(SimParams(), seed=seed)
    sim.step(50.0)
    p = MPCParams()
    both = Compare(MPC(FILL, p), ThresholdRule(tau=int(p.rho_low), ahead=int(p.rho_up)))
    run_order(sim, both, side, 500, twap(500, 20), 5.0)
    assert all(both.same)

def test_mpc_completes_an_order_with_logistic_fills(logistic):
    sim = Simulator(SimParams(), seed=0)
    sim.step(50.0)
    policy = MPC(logistic)
    r = run_order(sim, policy, side=Side.BUY, Q=500, schedule=twap(500, 20), dt=5.0)
    assert r.filled == 500 and sim.my_orders == set()
    assert policy.n_fallbacks == 0
    assert any(e.passive for e in r.executions)

@pytest.mark.parametrize("seed, side", [(0, Side.BUY), (1, Side.SELL), (2, Side.BUY)])
def test_mpc_with_logistic_fills_departs_from_the_threshold_rule(logistic, seed, side):
    sim = Simulator(SimParams(), seed=seed)
    sim.step(50.0)
    p = MPCParams()
    both = Compare(MPC(logistic, p), ThresholdRule(tau=int(p.rho_low), ahead=int(p.rho_up)))
    run_order(sim, both, side, 500, twap(500, 20), 5.0)
    assert not all(both.same)