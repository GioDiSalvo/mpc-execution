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

## D9 — Limit-order distances are measured from the opposite best quote
**Decision.** A new buy limit order at distance `d` is placed at `best_ask - d`; a new sell at `best_bid + d`, for `d = 1..K`. Cancellation rates use the same indexing. (Cont–Stoikov–Talreja convention.)
**Why.** With `d < spread` the order lands inside the spread and improves the best quote, so the spread can close on its own after a level is emptied. Measuring from the same-side best would only ever add depth behind the best, and the spread could widen but never narrow.
**Consequence.** `_level_price(side, d)` is the single place that implements this arithmetic; both `_rates` and `_next_event` use it. The number of levels `K` must exceed the spread for every distance to be a valid price. The seed populates distances 2..K so that the seeded levels coincide with those _rates observes; distance 1 (inside the initial spread) starts empty.”

## D10 — Order sizes are geometric with mean `qty_mean`
**Decision.** Every simulated limit and market order has size drawn from `Geometric(1 / qty_mean)`, with support {1, 2, …} and mean exactly `qty_mean`. One draw per order, none for cancellations.
**Why.** Simplest discrete law with the right support and a one-parameter mean. Its tail is exponential, so no huge orders; a heavier tail can be swapped in later at a single call site (`_draw_qty`).
**Consequence.** Rates are expressed per order, not per share, except cancellations (D11). The equilibrium depth used by the seed is `lambda(d) * qty_mean / theta`.

## D11 — Cancellations remove one whole order, chosen in proportion to its size, never ours
**Decision.** When a cancellation event is drawn at distance `d`, one resting order at that level is selected with probability proportional to its quantity and cancelled entirely. Orders with `is_mine = True` are never candidates. If no candidate exists, the event has no effect but the clock still advances.
**Why.** The CST rate `theta * Q_d` models one clock per share; selecting an order by size preserves that total rate while matching real markets, where cancellations are of whole orders. Our own orders are cancelled only by our policy.
**Consequence.** A level consisting only of our orders is never cancelled by the flow; our orders at a level are therefore a slight stabiliser of that level.

## D12 — A liquidity floor prevents a side from emptying
**Decision.** After every event, each side whose total depth is below `floor` receives one order of size `floor` at distance `K + 1` from the opposite best. The number of interventions is counted in `n_floor_events`.
**Why.** The CST model has finitely many levels and can empty a side by a run of market orders and cancellations; an empty side leaves the mid undefined and the algorithm cannot run. In real markets there is always liquidity at some price.
**Consequence.** `n_floor_events` is a diagnostic: a well-parameterised simulation triggers the floor rarely. Both sides below threshold with one of them empty is unreachable by design and guarded by an assertion.