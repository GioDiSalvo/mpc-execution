## D1 — Prices are integer ticks
**Decision.** All prices inside the book are integers (number of ticks). Conversion to currency happens outside.
**Why.** Avoids floating-point comparisons in the matching logic; makes equality of price levels exact.
**Consequence.** The simulator and the candidate-order generator must produce tick prices; metrics convert back.

## D2 — Mid price is the only non-integer quantity in the book
**Decision.** `mid()` returns a float (`best_bid + spread / 2`), e.g. 101.5 for bid 100 / ask 103. All other prices stay integer ticks (D1).
**Why.** The mid is a reference level, not a tradeable price: nothing is ever placed or filled at it. Forcing it to an integer would bias every cost measured against it by up to half a tick.
**Consequence.** Costs expressed "relative to mid" (the paper's `c_t`) are floats in tick units. Any code that needs a price to *send* an order (candidate generation, mid-IOC) must round explicitly and record the rounding rule.

## D3 — Invariants are enforced by construction, checked only in tests
**Decision.** `best_bid < best_ask`, no empty levels, index = book. Never checked inside `add_*`; verified by `check_invariants()` in tests, including a randomized one.
**Why.** The matching logic cannot produce a crossed book; a runtime check would be dead code in the hot path.
**Consequence.** Any new mutating method must be covered by the randomized test.

## D4 — `queue.remove(order)` is O(queue length)
**Decision.** Accepted for now.
**Why.** Queues are short in the simulator we plan; a level dict or lazy deletion can replace it later without changing the interface.
**Consequence.** Revisit if profiling shows cancellations dominating.

## D5 — `_remove` vs `cancel`
**Decision.** `_remove(order)` is the mechanical removal, shared by `cancel` and `_match`. `cancel(order_id)` is the public withdrawal, with id lookup.
**Why.** One place knows how to take an order out of the book.

## D6 — Return values and errors
**Decision.** `add_limit` returns `(None, fills)` when nothing rests. `cancel` and `queue_ahead` raise `KeyError` on unknown ids. `qty <= 0` raises `ValueError`.
**Why.** Silent defaults (returning 0) hide caller bugs in the simulator.

## D7 — Fills occur at the resting order's price
**Decision.** An incoming order executes at the price of each resting order it hits, never at its own limit.
**Why.** Standard exchange rule (price-time priority); it is what the paper's cost model assumes.

## D8 — Market orders carry an unbounded limit
**Decision.** `add_market` calls `_match` with `+inf` for buys and `-inf` for sells.
**Why.** One matching routine for both order types. The residual of a market order with insufficient liquidity is returned, not resting.