from typing import Protocol
import numpy as np
import pandas as pd
from mpc_execution.book import Side
from mpc_execution.simulator import BookState, Simulator, SimParams

def candidate_prices(state: BookState, side: Side, n_levels: int) -> np.ndarray:
    """Level j = j ticks behind our own touch (D20). Integer array of length n_levels."""
    phi = side.value
    best = state.best_bid if side is Side.BUY else state.best_ask
    return best - phi * np.arange(n_levels)

class FillModel(Protocol):
    def probs(self, state: BookState, side: Side, qty: int) -> np.ndarray:
        """pi[j] in [0, 1]: expected executed fraction within one step of `qty` shares
        posted now at candidate level j. Shape (n_levels,)."""

class ConstantFill:
    """The paper's model: one number per level, independent of the book state."""
    def __init__(self, pi):            # np.asarray(float); ValueError if any value outside [0, 1]
        if np.any((pi<0) | (pi>1)):
            raise ValueError(f"Fill probabilities must lie in [0,1], got {pi}")
        self.pi = pi
    
    def probs(self, state, side, qty):
        return self.pi

    @classmethod
    def from_samples(cls, samples: pd.DataFrame) -> "ConstantFill":   # mean filled_frac by level
        return cls(samples.groupby("level").filled_frac.mean().to_numpy())

def collect_fill_samples(params: SimParams, n_levels: int, qty: int, dt: float,
                         n_trials: int, seed: int = 0, warmup: float = 50.0) -> pd.DataFrame:
    """One row per (trial, level). Columns:
    trial, side (name), level, price, filled_frac,
    spread, imbalance, depth_at_level (shares ahead at that price before posting, ours excluded),
    shares_ahead (cumulative depth of our own side from the touch down to that level, before posting:
    the shares with price or time priority over our order; from `shares_ahead_by_level`)."""
    sim = Simulator(params, seed)
    sim.step(warmup)
    rows = []
    for i in range(n_trials):
        side = Side.BUY if i%2==0 else Side.SELL
        state = sim.snapshot() # state before posting my orders
        spread, imbalance = state.spread, state.imbalance
        depth_at_level = state.bid_depth if side is Side.BUY else state.ask_depth
        assert n_levels <= len(depth_at_level), "levels required are more than the ones available, i.e., n_levels > len(depth_at_level)"
        prices = [int(p) for p in candidate_prices(state, side, n_levels)]

        for price in prices: # n_levels
            oid, ex = sim.place_limit(side, int(price), qty)
            assert oid is not None and len(ex) == 0

        ex = sim.step(dt)

        shares_ahead = shares_ahead_by_level(state, side, n_levels)
        for j, price in enumerate(prices):
            executed = sum(e.qty for e in ex if e.price == price)
            rows.append({"trial": i, "side": side.name, "level": j, "price": price,
                         "filled_frac": executed / qty,
                         "spread": spread, "imbalance": imbalance,
                         "depth_at_level": int(depth_at_level[j]),
                         "shares_ahead": shares_ahead[j]}
            )
        sim.cancel_all_mine()
        assert sim.my_orders == set()
    return pd.DataFrame(rows)

def shares_ahead_by_level(state: BookState, side: Side, n_levels: int) -> np.ndarray:
    """Shares with price or time priority over an order posted now at each candidate level:
    cumulative depth of our own side from the touch down to that level. Shape (n_levels,)."""
    depth_at_level = state.bid_depth if side is Side.BUY else state.ask_depth
    return np.array([int(sum(depth_at_level[:j+1])) for j in range(n_levels)])

def design_matrix(shares_ahead, level, spread) -> np.ndarray:
    """One row per candidate order. Columns: 1, log(1 + shares_ahead), level, 1[spread > 1]."""
    assert len(shares_ahead) == len(level) == len(spread)
    shares_ahead, level, spread = np.asarray(shares_ahead, dtype=float), np.asarray(level, dtype=float), np.asarray(spread, dtype=float)
    n = len(shares_ahead)
    ones = np.ones(n)
    return np.array([ones, np.log(1 + shares_ahead), level, (spread > 1).astype(int)]).T

def sigmoid(z):
    """1 / (1 + exp(-z)), with z clipped to [-35, 35]."""
    z = np.clip(z, -35, 35)
    return 1 / (1 + np.exp(-z))

def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-6, max_iter: int = 100, tol: float = 1e-10) -> np.ndarray:
    """Coefficients of the (fractional) logit, by Newton's method.
    Minimises -sum(y log p + (1 - y) log(1 - p)) + (l2 / 2) |beta|^2 with p = sigmoid(X beta).
    Raises RuntimeError if it has not converged after max_iter steps."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = sigmoid(X @ beta)
        g = X.T @ (p - y) + l2 * beta                      # gradient
        w = (p * (1 - p)) [:, None]
        H = X.T @ (X * w) + l2 * np.eye(X.shape[1])        # Hessian
        step = np.linalg.solve(H, g)
        if np.max(np.abs(step)) < tol:
            break
        beta -= step
    else:
        raise RuntimeError(f"fit_logistic did not converge in {max_iter} iterations")
    return beta

class LogisticFill:
    """State-dependent fill model: pi_j = sigmoid(beta . x_j), x_j the features of candidate level j in the current book."""
    def __init__(self, beta, n_levels: int):
        self.beta = np.asarray(beta, dtype=float)
        self.n_levels = n_levels

    def probs(self, state: BookState, side: Side, qty: int) -> np.ndarray:     # shape (n_levels,)
        shares_ahead = shares_ahead_by_level(state, side, self.n_levels)
        levels =  np.arange(self.n_levels)
        spread = np.full(self.n_levels, state.spread)
        X = design_matrix(shares_ahead, levels, spread)
        pi = sigmoid(X @ self.beta)
        return pi

    @classmethod
    def from_samples(cls, samples: pd.DataFrame, l2: float = 1e-6) -> "LogisticFill":
        X = design_matrix(samples.shares_ahead, samples.level, samples.spread) # the ingredients
        y = samples.filled_frac.to_numpy()                                     # what happened
        beta = fit_logistic(X, y, l2)                                          # learn the coefficients
        n_levels = samples.level.max() + 1
        return cls(beta, n_levels)
        