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

    def _level_price(self, side, d):
        ref = self.book.best_ask() if side is Side.BUY else self.book.best_bid()
        assert ref is not None, "opposite side empty: floor should have prevented this"
        return ref - d if side is Side.BUY else ref + d

    def _level_qty(self, side, d):
        book_side = self.book.bids if side is Side.BUY else self.book.asks
        price = self._level_price(side, d)
        queue = book_side.get(price)
        return sum(o.qty for o in queue) if queue else 0

    def _rates(self):
        """
        Returns a 4K+2-dim array, describes the occurance rate of an event at distance d from the best of the opposite side:
        - buy limit orders (lam(d), for d = 1,...,K);
        - sell limit orders (lam(d), for d = 1,...,K);
        - buy cancellation (theta(d) * Q_d, for d = 1,...,K where Q_d is the queue size at level d for the buy);
        - sell cancellation (theta(d) * Q_d, for d = 1,...,K where Q_d is the queue size at level d for the sell);
        - the two final entries are buy market orders pressure vs sell market order pressure.
        """
        ds = np.arange(1, self.p.K + 1)
        lam = self.p.lam(ds)
        cancel_bid = self.p.theta * np.array([self._level_qty(Side.BUY, d) for d in ds])
        cancel_ask = self.p.theta * np.array([self._level_qty(Side.SELL, d) for d in ds])
        return np.concatenate([lam, lam, cancel_bid, cancel_ask, [self.p.mu], [self.p.mu]])