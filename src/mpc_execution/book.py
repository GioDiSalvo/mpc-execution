from dataclasses import dataclass
from collections import deque
from enum import Enum

class Side(Enum):
    BUY = 1
    SELL = -1

@dataclass         # generates automatically __init__, __repr__, __eq__
class Order:
    id: int
    side: Side
    price: int     # ticks, see DESIGN.md D1
    qty: int       # remaining quantity
    is_mine: bool = False

@dataclass(frozen=True) # makes the class instance immutable
class Fill:
    order_id: int
    price: int
    qty: int
    is_mine: bool

class LimitOrderBook:
    def __init__(self):
        self.bids: dict[int, deque[Order]] = {}    # keys are prices in tick units, and val is the queue of order at that price
        self.asks: dict[int, deque[Order]] = {}
        self.orders: dict[int, Order] = {}         # keys are order_id, and val is the order itself
        self._next_id = 0

    def __repr__(self):
        def fmt_side(levels: dict[int, deque[Order]], prices: list[int]) -> str:
            if not prices:
                return "(empty)"
            return " | ".join(
                f"{p} x {sum(o.qty for o in levels[p])} ({len(levels[p])})" for p in prices
            )

        asks = sorted(self.asks)[:5][::-1]            # best ask last, close to bid
        bids = sorted(self.bids, reverse=True)[:5]    # best bid first
        header = f"LimitOrderBook(bid/ask {self.best_bid()}/{self.best_ask()}, spread {self.spread()})"
        return "\n".join([header, f"  asks: {fmt_side(self.asks, asks)}", f"  bids: {fmt_side(self.bids, bids)}"])

    def best_bid(self) -> int | None:
        if not self.bids:
            return None
        return max(self.bids)

    def best_ask(self) -> int | None:
        if not self.asks:
            return None
        return min(self.asks)

    def spread(self) -> int | None:
        bid, ask = self.best_bid(), self.best_ask()
        if bid is None or ask is None:
            return None
        return ask - bid

    def mid(self) -> float | None:
        bid, ask = self.best_bid(), self.best_ask()
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2

    def add_market(self, side, qty):
        """
        Execute a mrket order.
        Returns the number of non filled shares and the list of fills.
        Raises ValueError if qty <= 0.
        """
        if qty <= 0:
            raise ValueError("quantity must be positive")
        limit = float("inf") if side is Side.BUY else float("-inf")
        return self._match(side, limit, qty) 

    def add_limit(self, side, price, qty, is_mine = False):
        """
        Execute the marketable part against the opposite side, then rest the residual in the book.
        Returns (order_id, fills); order_id is None if nothing rested.
        """
        if qty <= 0:
            raise ValueError("quantity must be positive")

        residual, fills = self._match(side, price, qty)
        if residual == 0:
            return None, fills
        
        order_id = self._next_id
        self._next_id += 1
        order = Order(order_id, side, price, residual, is_mine)
        self.orders[order_id] = order
        book_side = self.bids if side is Side.BUY else self.asks
        book_side.setdefault(price, deque()).append(order)
        return order_id, fills

    def check_invariants(self):
        """
        General sanity check: no bid-ask crossing, no empty levels, no non-positive qty, no wrong order stored.
        """
        bid, ask = self.best_bid(), self.best_ask()
        assert bid is None or ask is None or bid < ask, f"crossed book: bid ({bid}) >= ask({ask})"
        for side in (self.bids, self.asks):
            for price, queue in side.items():
                assert len(queue) > 0, f"empty level at price {price}"
                assert all(o.qty > 0 for o in queue), f"non positive quantity at price {price}"
        assert all(o.id in self.orders for q in list(self.bids.values()) + list(self.asks.values()) for o in q)
        in_book = sum(len(q) for q in self.bids.values()) + sum(len(q) for q in self.asks.values())
        assert in_book == len(self.orders), f"index has {len(self.orders)} orders, book has {in_book}"

    def _remove(self, order):
        """
        Remove the given order from its queue in the book, deleting the queue itself if left empty.
        The order is removed from the record self.orders coherently.
        """
        price, side, o_id = order.price, order.side, order.id
        book_side = self.bids if side is Side.BUY else self.asks
        queue = book_side[price]
        queue.remove(order)
        if not queue:
            del book_side[price]
        del self.orders[o_id]

    def cancel(self, order_id):
        """
        Withdraw a resting order.
        Returns the cancelled quantity.
        Raises KeyError if the id is unknown.
        """
        if order_id not in self.orders:
            raise KeyError(f"unknown order id {order_id}")
        order = self.orders[order_id]
        self._remove(order)
        return order.qty

    def queue_ahead(self, order_id):
        """
        Returns the amount of shares related to orders at better price and to earlier orders at same price.
        Raises KeyError if the id is unknown.
        """
        if order_id not in self.orders:
            raise KeyError
        else:
            tot_qty = 0
            order = self.orders[order_id]
            price, side = order.price, order.side

            if side is Side.BUY:
                book_side = self.bids
                prices = [p for p in book_side if p > price]
            else:
                book_side = self.asks
                prices = [p for p in book_side if p < price]

            ahead_same_level = 0
            for o in book_side[price]:
                if o.id == order_id:
                    break
                ahead_same_level += o.qty

            tot_qty += ahead_same_level
            for p in prices:
                tot_qty += sum(o.qty for o in book_side[p])
            return tot_qty

    def _match(self, side, limit_price, qty):
        """
        Execute qty against the opposite side, level by level from the best, while the level price is within limit_price.
        Fills occur at the resting order’s price.
        Fully executed resting orders are removed.
        Returns (residual, fills).
        """
        fills = []
        opposite = self.asks if side is Side.BUY else self.bids

        while qty > 0:
            best = self.best_ask() if side is Side.BUY else self.best_bid()
            if best is None:
                break
            if (side is Side.BUY and best > limit_price) or (side is Side.SELL and best < limit_price):
                break
            queue = opposite[best]
            while queue and qty > 0:
                order = queue[0]
                executed_qty = min(qty, order.qty)
                fills.append(Fill(order.id, best, executed_qty, order.is_mine))
                qty -= executed_qty
                order.qty -= executed_qty
                if order.qty == 0:
                    self._remove(order)
        return qty, fills