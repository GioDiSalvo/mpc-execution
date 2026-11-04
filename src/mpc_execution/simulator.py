from dataclasses import dataclass
import numpy as np
from mpc_execution.book import LimitOrderBook, Side

@dataclass
class SimParams:
    """
    Rates of the Cont-Stoikov-Talreja order flow.
    Time unit is arbitrary: only ratios matter.
    See docs/notes.pdf, Section 2.
    """
    K: int = 10           # price levels per side beyond the best (distances 1 ... K)
    lambda0: float = 2.0    # limit-order arrival rate at distance 1
    alpha: float = 1.5      # power-law decay of arrivals with distance
    theta: float = 0.02     # cancellation rate per share
    mu: float = 1.0         # market-order rate per side (neutral regime)
    qty_mean: float = 50.0  # mean order size
    mid0: int = 10_000    # initial mid price, in ticks
    floor: int = 100      # regenerate liquidity if a side's depth drops below this

    def lam(self, d):
        'Gets arrays and scalars as input'
        return self.lambda0 / d ** self.alpha

class Simulator:
    def __init__(self, params: SimParams, seed: int):
        self.p = params
        self.rng = np.random.default_rng(seed)
        self.book = LimitOrderBook()
        self.t = 0.0        # it's the simulator's clock
        self._seed_book()   # initialization of the book

    def _seed_book(self):
        for d in range(2, self.p.K + 1):
            qty = max(1, round(self.p.lam(d) * self.p.qty_mean / self.p.theta))
            self.book.add_limit(Side.BUY, self.p.mid0 + 1 - d, qty)
            self.book.add_limit(Side.SELL, self.p.mid0 - 1 + d, qty)