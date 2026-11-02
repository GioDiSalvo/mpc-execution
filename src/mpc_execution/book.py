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

    def add_limit(self, side, price, qty, is_mine = False):
        if qty <= 0:
            raise ValueError("quantity must be positive")
        
        if side is Side.BUY:
            opposite_best = self.best_ask()
            crosses = opposite_best is not None and price >= opposite_best
            book_side = self.bids
        else:
            opposite_best = self.best_bid()
            crosses = opposite_best is not None and price <= opposite_best
            book_side = self.asks

        if crosses:
            raise NotImplementedError("crossing orders: cycle 4")

        order_id = self._next_id
        self._next_id += 1
        order = Order(order_id, side, price, qty, is_mine)
        self.orders[order_id] = order
        book_side.setdefault(price, deque()).append(order)
        return order_id, []

    def check_invariants(self):
        bid, ask = self.best_bid(), self.best_ask()
        assert bid is None or ask is None or bid < ask, f"crossed book: bid ({bid}) >= ask({ask})"
        for side in (self.bids, self.asks):
            for price, queue in side.items():
                assert len(queue) > 0, f"empty level at price {price}"
                assert all(o.qty > 0 for o in queue), f"non positive quantity at price {price}"
        assert all(o.id in self.orders for q in list(self.bids.values()) + list(self.asks.values()) for o in q)
        in_book = sum(len(q) for q in self.bids.values()) + sum(len(q) for q in self.asks.values())
        assert in_book == len(self.orders), f"index has {len(self.orders)} orders, book has {in_book}"

    def cancel(self, order_id):
        if order_id not in self.orders:
            raise KeyError
        else:
            order = self.orders.pop(order_id)
            price, side, qty = order.price, order.side, order.qty
            book_side = self.bids if side is Side.BUY else self.asks
            queue = book_side[price]
            queue.remove(order)
            if not queue:
                del book_side[price]
            return qty

    def queue_ahead(self, order_id):
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