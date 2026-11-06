from dataclasses import dataclass
import numpy as np
from mpc_execution.book import LimitOrderBook, Side
from enum import IntEnum

class Block(IntEnum):
    LIMIT_BUY = 0
    LIMIT_SELL = 1
    CANCEL_BID = 2
    CANCEL_ASK = 3

@dataclass
class SimParams:
    """
    Rates of the Cont-Stoikov-Talreja order flow.
    Time unit is arbitrary: only ratios matter.
    See docs/notes.pdf, Section 2.
    """
    K: int = 10           # price levels per side beyond the best (distances 1 ... K)
    lambda0: float = 2.0    # limit-order arrival rate at distance 1
    alpha: float = 1.2      # power-law decay of arrivals with distance
    theta: float = 0.2     # cancellation rate per resting order
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
        self.n_floor_events: int = 0 # counter for empty book
        self.MARKET_BUY = 4 * self.p.K
        self.MARKET_SELL = 4 * self.p.K + 1
        self._seed_book()   # initialization of the book

    def _seed_book(self):
        for d in range(2, self.p.K + 1):
            n = max(1, round(self.p.lam(d) / self.p.theta))
            qty = round(self.p.qty_mean)
            for _ in range(n):
                self.book.add_limit(Side.BUY, self.p.mid0 + 1 - d, qty)
                self.book.add_limit(Side.SELL, self.p.mid0 - 1 + d, qty)

    def _level_price(self, side, d):
        'Price at distance d from the opposite best quote.'
        if side is Side.BUY:
            ref, opposite, sign = self.book.best_ask(), Side.SELL, -1
        else:
            ref, opposite, sign = self.book.best_bid(), Side.BUY, 1
        assert ref is not None, f"{opposite.name} side empty: floor should have prevented this"
        return ref + sign * d

    def _level_qty(self, side, d):
        """Tot qty in all resting orders at distance d that the flow may cancel (ours included)."""
        book_side = self.book.bids if side is Side.BUY else self.book.asks
        price = self._level_price(side, d)
        queue = book_side.get(price)
        return sum(o.qty for o in queue) if queue else 0

    def _level_count(self, side, d):
        """Number of resting orders at distance d that the flow may cancel (ours excluded)."""
        book_side = self.book.bids if side is Side.BUY else self.book.asks
        price = self._level_price(side, d)
        queue = book_side.get(price)
        return sum(1 for o in queue if not o.is_mine) if queue else 0

    def _rates(self):
        """
        Returns a 4K+2-dim array, describes the occurance rate of an event at distance d from the best of the opposite side:
        - buy limit orders (lam(d), for d = 1,...,K);
        - sell limit orders (lam(d), for d = 1,...,K);
        - buy cancellation (theta(d) * Q_d, for d = 1,...,K where Q_d is the queue size -number of orders- at level d for the buy, ours excluded);
        - sell cancellation (theta(d) * Q_d, for d = 1,...,K where Q_d is the queue size -number of orders- at level d for the sell, ours excluded);
        - the two final entries are buy market orders pressure vs sell market order pressure.
        """
        ds = np.arange(1, self.p.K + 1)
        lam = self.p.lam(ds)
        cancel_bid = self.p.theta * np.array([self._level_count(Side.BUY, d) for d in ds])
        cancel_ask = self.p.theta * np.array([self._level_count(Side.SELL, d) for d in ds])
        return np.concatenate([lam, lam, cancel_bid, cancel_ask, [self.p.mu], [self.p.mu]])

    def _draw_qty(self):
        return self.rng.geometric(1 / self.p.qty_mean)

    def _depth(self, side):
        'Returns the total quantity of shares in the given side'
        book_side = self.book.bids if side is Side.BUY else self.book.asks
        return sum(o.qty for queue in book_side.values() for o in queue)

    def _apply_event(self, e):
        K = self.p.K
        if e >= 4 * K:
            side = Side.BUY if e == self.MARKET_BUY else Side.SELL
            qty = self._draw_qty()
            self.book.add_market(side, qty)
        else:
            block, d = divmod(e, K) # d ranges in 0,...,K-1
            block = Block(block)
            if block in (Block.LIMIT_BUY, Block.LIMIT_SELL):
                qty = self._draw_qty()
                side = Side.BUY if block is Block.LIMIT_BUY else Side.SELL
                self.book.add_limit(side, self._level_price(side, d + 1), qty)
            else:
                side = Side.BUY if block is Block.CANCEL_BID else Side.SELL
                book_side = self.book.bids if block is Block.CANCEL_BID else self.book.asks
                price = self._level_price(side, d + 1)
                queue = book_side.get(price)
                candidates = [o for o in queue if not o.is_mine] if queue else []
                if candidates:
                    victim = self.rng.choice(len(candidates))
                    self.book.cancel(candidates[victim].id)

    def _apply_floor(self):
        K = self.p.K
        assert self.book.best_bid() is not None or self.book.best_ask() is not None, "both sides of the book are empty"
        for side in (Side.BUY, Side.SELL):
            if self._depth(side) < self.p.floor:
                price = self._level_price(side, K + 1)
                self.book.add_limit(side, price, self.p.floor)
                self.n_floor_events += 1

    def _next_event(self):
        """Draw the next event (Gillespie), apply it to the book, advance the clock.
        Returns the waiting time tau."""
        rates = self._rates()
        Lambda = rates.sum()
        tau = self.rng.exponential(1 / Lambda) # time to next event
        probabilities = rates / Lambda
        e = self.rng.choice(len(rates), p=probabilities) # index of next event
        self._apply_event(e)
        self._apply_floor()
        self.t += tau       # move simulator's clock
        return tau