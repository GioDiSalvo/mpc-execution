from dataclasses import dataclass
from typing import Literal, Protocol
from mpc_execution.book import Side
from mpc_execution.simulator import BookState

@dataclass(frozen=True)
class OrderSpec:
    kind: Literal["market", "limit"]
    qty: int
    price: int | None = None     # required for limit

class Policy(Protocol):
    def decide(self, state: BookState, side: Side, q: int, target: int, remaining: int) -> list[OrderSpec]:
        """q: executed so far; target: s_{k+1}; remaining: Q - q."""

class Crossing:
    """Send the scheduled slice as a market order at every step (the paper's baseline)."""
    def decide(self, state, side, q, target, remaining):
        gap = min(target - q, remaining)
        return [OrderSpec("market", gap)] if gap > 0 else []

@dataclass(frozen=True)
class ThresholdRule:
    """Rule-based schedule follower in the spirit of Cartea & Jaimungal (2015): passive while within
    tolerance, market orders to catch up. No fill model, no optimisation (D24)."""
    tau: int = 50      # tolerated delay behind the schedule, shares; the excess goes to market
    ahead: int = 0     # shares it may post beyond the schedule target

    def decide(self, state: BookState, side, q, target, remaining):
        if remaining <= 0:
            return []
        gap = target - q
        if gap == remaining:                    # last step (D22): complete with a market order
            return [OrderSpec("market", gap)]
        market = max(0, gap - self.tau)
        limit = max(0, min(gap + self.ahead, remaining) - market)
        orders = []
        if market > 0:
            orders.append(OrderSpec("market", market))
        if limit > 0:
            price = state.best_bid if side is Side.BUY else state.best_ask
            orders.append(OrderSpec("limit", limit, price))
        return orders

class Traced:
    def __init__(self, policy):
        self.policy, self.log = policy, []
    def decide(self, state, side, q, target, remaining):
        orders = self.policy.decide(state, side, q, target, remaining)
        self.log.append((q, target, state.spread, [(o.kind, o.qty, o.price) for o in orders]))
        return orders        