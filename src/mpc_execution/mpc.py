from dataclasses import dataclass
import numpy as np
import cvxpy as cp
from mpc_execution.book import Side
from mpc_execution.simulator import BookState

@dataclass(frozen=True)
class MPCParams:
    n_levels: int = 3                # limit candidates: our touch and the ticks behind it (D20)
    gamma: float = 0.02              # schedule-deviation penalty, spreads per share^2
    xi: float = 0.5                  # rollout: cost per share of finishing with market orders, in spreads
    rho_up: float = 25.0             # upper tube, shares
    rho_low: float = 50.0            # lower tube, shares
    beta: float | None = None        # variance budget on the step's executed quantity, shares^2; None = off
    kappa_frac: float | None = 0.5   # cap on any single child order, as a fraction of Q; None = off

def nested_cov(pi: np.ndarray) -> np.ndarray:
    """Sigma_ij = min(pi_i, pi_j) - pi_i pi_j (paper; verified in notebook 03)."""
    return np.minimum.outer(pi, pi) - np.outer(pi, pi)

def candidate_costs(state: BookState, side: Side, limit_prices: np.ndarray) -> np.ndarray:
    """Cost per share in spreads, phi * (p - mid) / spread. Returns [0.5, c_1, ..., c_n]:
    the market order pays half a spread, a limit at our touch earns half a spread."""
    phi = side.value
    spread, mid = state.spread, state.mid
    limit = phi * (limit_prices - mid) / spread
    return np.concatenate(([0.5], limit))

def solve_qp(pi: np.ndarray, c: np.ndarray, gap: float, remaining: float,
             last_step: bool, p: MPCParams, kappa: float = np.inf) -> np.ndarray:
    """Continuous solution u of the one-step QP, shape (1 + n_levels,), u[0] = market.
    pi[0] must be 1, gap = s[t+1] - q[t].
    Raises RuntimeError if the solver does not return an optimal status."""
    assert pi[0] == 1 and len(pi) == len(c)
    u = cp.Variable(len(pi), nonneg = True)
    objective = cp.Minimize((pi * (c - p.xi)) @ u + p.gamma * cp.square(pi @ u - gap))
    rho_low = 0 if last_step else p.rho_low
    constraints = [u.sum() <= max(0, min(gap + p.rho_up, remaining)), u[0] >= gap - rho_low]
    if p.beta is not None:
        constraints.append(cp.quad_form(u, cp.psd_wrap(nested_cov(pi))) <= p.beta)
    if np.isfinite(kappa):
        constraints.append(u <= kappa)
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.SolverError as e:
        raise RuntimeError(f"QP solver failed: {e}") from e
    if prob.status != cp.OPTIMAL:
        raise RuntimeError(f"QP not solved: status {prob.status} (not optimal)")
    return np.maximum(u.value, 0.0)