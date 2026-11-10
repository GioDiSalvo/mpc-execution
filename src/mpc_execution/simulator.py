from dataclasses import dataclass
import numpy as np
from mpc_execution.book import LimitOrderBook, Side, Fill
from enum import IntEnum

class Block(IntEnum):
    LIMIT_BUY = 0
    LIMIT_SELL = 1
    CANCEL_BID = 2
    CANCEL_ASK = 3

@dataclass(frozen=True)
class Execution:
    """One execution of OURS, in either direction (D16)."""
    t: float
    side: Side        # our side: BUY if we bought
    price: int        # price actually traded (the resting order's price, D7)
    qty: int
    passive: bool     # True: our resting order was hit by the flow. False: we crossed the spread.

@dataclass
class BookState:
    """What a policy observes at a decision time (D15)."""
    t: float
    mid: float
    spread: int
    best_bid: int
    best_ask: int
    bid_depth: np.ndarray    # shape (K,): shares at best_bid, best_bid-1, ..., best_bid-(K-1)
    ask_depth: np.ndarray    # shape (K,): shares at best_ask, best_ask+1, ..., best_ask+(K-1),
    imbalance: float         # (sum(bid_depth[:3]) - sum(ask_depth[:3])) / (sum of both), in [-1, 1]
    my_orders: dict[int, tuple[Side, int, int, int]]   # id -> (side, price, remaining qty, queue_ahead)

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
        self.my_orders: set[int] = set()          # ids of our resting orders
        self._executions: list[Execution] = []    # passive executions since the last step()
        self.trades: list[tuple[float, int, int]] = [] # (t, price, qty)

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

    def _record_passive_fills(self, fills: list[Fill], aggressor: Side) -> None:
        """Fills of OUR resting orders hit by a flow order on side `aggressor`."""
        for f in fills:
            if f.is_mine:
                self._executions.append(Execution(self.t, Side(-aggressor.value), f.price, f.qty, passive=True))

    def _record_trades(self, fills: list[Fill]) -> None:
        for f in fills:
            self.trades.append((self.t, f.price, f.qty))

    def _apply_event(self, e):
        K = self.p.K
        if e >= 4 * K:
            side = Side.BUY if e == self.MARKET_BUY else Side.SELL
            qty = self._draw_qty()
            _, fills = self.book.add_market(side, qty)
            self._record_passive_fills(fills, side)
            self._record_trades(fills)
        else:
            block, d = divmod(e, K) # d ranges in 0,...,K-1
            block = Block(block)
            if block in (Block.LIMIT_BUY, Block.LIMIT_SELL):
                qty = self._draw_qty()
                side = Side.BUY if block is Block.LIMIT_BUY else Side.SELL
                _, fills = self.book.add_limit(side, self._level_price(side, d + 1), qty)
                self._record_passive_fills(fills, side)
                self._record_trades(fills)
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

    def _next_event(self, t_max: float | None = None) -> float | None:
        """Draw the next event (Gillespie), apply it to the book, advance the clock.
        Returns the waiting time tau."""
        rates = self._rates()
        Lambda = rates.sum()
        tau = self.rng.exponential(1 / Lambda) # time to next event
        if t_max is not None and self.t + tau > t_max:
            self.t = t_max
            return None
        else:
            probabilities = rates / Lambda
            e = self.rng.choice(len(rates), p=probabilities) # index of next event
            self._apply_event(e)
            self._apply_floor()
            self.t += tau       # move simulator's clock
            return tau

    def place_limit(self, side: Side, price: int, qty: int) -> tuple[int | None, list[Execution]]:
        """Send our limit order. Returns (resting order id or None, executions of the marketable part)."""
        order_id, fills = self.book.add_limit(side, price, qty, is_mine=True)
        if order_id is not None:
            self.my_orders.add(order_id)
        self._record_trades(fills)
        executions = []
        for f in fills:
            executions.append(Execution(self.t, side, f.price, f.qty, passive=False))
        return order_id, executions

    def place_market(self, side: Side, qty: int) -> tuple[int, list[Execution]]:
        """Send our market order. Returns (unfilled residual, executions)."""
        unfilled_residual_qty, fills = self.book.add_market(side, qty)
        self._record_trades(fills)
        executions = []
        for f in fills:
            executions.append(Execution(self.t, side, f.price, f.qty, passive=False))
        return unfilled_residual_qty, executions

    def cancel_mine(self, order_id: int) -> int:
        """Withdraw one of our resting orders. Returns the cancelled qty. KeyError if not ours."""
        if order_id not in self.my_orders:
            raise KeyError(f"order {order_id} not in self.my_orders")
        cancelled_qty = self.book.cancel(order_id)
        self.my_orders.remove(order_id)
        return cancelled_qty

    def cancel_all_mine(self) -> int:
        """Withdraw all our resting orders. Returns the total cancelled qty."""
        tot_cancelled_qty = 0
        for o in list(self.my_orders):
            tot_cancelled_qty += self.cancel_mine(o)
        return tot_cancelled_qty

    def step(self, dt: float) -> list[Execution]:
        """Let the flow run for dt. Returns the passive executions of our orders in the interval."""
        t_end = self.t + dt
        while self._next_event(t_max=t_end) is not None:
            pass
        executions, self._executions = self._executions, []
        self.my_orders &= self.book.orders.keys()       # orders fully executed have left the book
        return executions

    def snapshot(self) -> BookState:
        """What a policy observes now (D15): depth from the own-side best, our resting orders with queue position."""
        K = self.p.K
        assert K >= 3,"imbalance uses 3 levels, so K must be >= {3}, but is currently K={K}."
        b = self.book
        best_bid, best_ask = b.best_bid(), b.best_ask()
        assert best_bid is not None and best_ask is not None, "floor should keep both sides populated"

        bid_depth = np.zeros(K, dtype=int)
        ask_depth = np.zeros(K, dtype=int)
        for k in range(K):
            q_bid = b.bids.get(best_bid - k)
            q_ask = b.asks.get(best_ask + k)
            bid_depth[k] = sum(o.qty for o in q_bid) if q_bid else 0
            ask_depth[k] = sum(o.qty for o in q_ask) if q_ask else 0

        top_bid, top_ask = bid_depth[:3].sum(), ask_depth[:3].sum()
        imbalance = (top_bid - top_ask) / (top_bid + top_ask)

        my_orders = {}
        for oid in self.my_orders:
            o = b.orders[oid]
            my_orders[oid] = (o.side, o.price, o.qty, b.queue_ahead(oid))

        return BookState(
            t=self.t, mid=b.mid(), spread=b.spread(), best_bid=best_bid, best_ask=best_ask,
            bid_depth=bid_depth, ask_depth=ask_depth, imbalance=imbalance, my_orders=my_orders,
        )