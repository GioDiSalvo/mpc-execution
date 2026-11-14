from mpc_execution.book import Side
from mpc_execution.simulator import Simulator, SimParams
from mpc_execution.mpc import MPCParams, nested_cov, candidate_costs, solve_qp
from mpc_execution.fills import candidate_prices
import numpy as np
import pytest

PI, C = np.array([1.0, 0.5]), np.array([0.5, -0.5])
TOL = 1e-3

@pytest.fixture
def state():
    sim = Simulator(SimParams(), seed=0)
    sim.step(50.0)
    return sim.snapshot()

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