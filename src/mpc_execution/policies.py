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

class Traced:
    def __init__(self, policy):
        self.policy, self.log = policy, []
    def decide(self, state, side, q, target, remaining):
        orders = self.policy.decide(state, side, q, target, remaining)
        self.log.append((q, target, state.spread, [(o.kind, o.qty, o.price) for o in orders]))
        return orders