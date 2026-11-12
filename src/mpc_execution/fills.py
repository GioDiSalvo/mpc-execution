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
    shares_ahead (from snapshot().my_orders right after posting; number of shares at better price or earlier at same price)."""
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

        for j, price in enumerate(prices):
            executed = sum(e.qty for e in ex if e.price == price)
            rows.append({"trial": i, "side": side.name, "level": j, "price": price,
                         "filled_frac": executed / qty,
                         "spread": spread, "imbalance": imbalance,
                         "depth_at_level": int(depth_at_level[j]),
                         "shares_ahead": int(sum(depth_at_level[:j+1]))}
            )
        sim.cancel_all_mine()
        assert sim.my_orders == set()
    return pd.DataFrame(rows)