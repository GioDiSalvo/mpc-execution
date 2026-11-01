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
    