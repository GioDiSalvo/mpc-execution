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

## D13 — Default order-flow parameters
**Decision.** `lambda0 = 2, alpha = 1.2, theta = 0.2, mu = 1.0, qty_mean = 50, K = 10, floor = 100`.
**Why.** Chosen in `notebooks/01_book_sanity.ipynb` from a grid over `theta`, `mu`, `alpha`. Target: a liquid stock. With these values (3 seeds, 50k events): spread = 1 tick 95 % of the time; 2.2–2.5 mid changes per 100 events; ≈ 275 shares at the best, ≈ 33 at distance 10; floor triggered 1–4 times per run. `theta = 0.02` (CST-like per-share value reinterpreted per order) gives a frozen book; `mu ≥ 2` empties it.
**Consequence.** `SimParams()` is this configuration everywhere (tests, experiments). Time unit: ≈ 20 events per unit of simulated time.

## D14 — `step(dt)` ends exactly at `t + dt`
**Decision.** `step(dt)` runs Gillespie events until the next drawn waiting time would overshoot `t + dt`; that event is discarded and the clock is set to `t + dt`. Implemented as `_next_event(t_max)`, which returns `None` when it truncates.
**Why.** Decision times must be equally spaced for the execution algorithm. Discarding the overshooting draw is exact, not an approximation: the exponential is memoryless, so the residual time to the next event, conditional on none having occurred by `t + dt`, is again exponential with the same rate.
**Consequence.** No event ever straddles two steps. The floor (D12) is applied after each applied event, never after a truncation.

## D15 — Snapshot depth is indexed from the own-side best
**Decision.** `BookState.bid_depth[k]` is the total quantity (ours included) at `best_bid - k`, `ask_depth[k]` at `best_ask + k`, for `k = 0..K-1`; `imbalance` uses the first three levels of each side. Rates keep the D9 convention (distance from the opposite best).
**Why.** Two different consumers. The flow model needs distances measured as in Cont–Stoikov–Talreja; a policy or a fill model sees the book as a trader does, from its own best quote. Keeping both explicit avoids off-by-one errors between the two.
**Consequence.** `_level_qty` / `_level_count` serve rates and diagnostics; `snapshot()` serves policies. `K >= 3` is required by the imbalance.

## D16 — Executions are reported from our point of view
**Decision.** `Execution(t, side, price, qty, passive)` records our side, the price actually traded (the resting order's price, D7) and whether we were passive (our resting order hit by the flow) or aggressive (we crossed the spread). Aggressive executions are returned immediately by `place_limit` / `place_market`; passive ones are buffered and returned by the next `step()`.
**Why.** A book `Fill` describes the resting order (its id and owner), which is the wrong point of view for our accounting: for an aggressive order the resting order is not ours. Metrics need our side and the time, which a `Fill` does not carry.
**Consequence.** The flow never cancels our orders (D11); only `cancel_mine` / `cancel_all_mine` do. `my_orders` is re-synchronised with the book at the end of every `step`, because fully executed orders leave the book.

## D17 — Our orders live exactly one decision step
**Decision.** At each decision time the policy sends its child orders; after `step(dt)` every order of ours still resting is cancelled (`cancel_all_mine`) before the next decision. Nothing of ours carries over between steps.
**Why.** This is the paper's setting (§4): the QP decides fresh quantities at each step, and a fill probability is "the probability that an order posted now is filled before the next decision". Letting orders persist would make queue position a state variable and the fill probability a function of the order's age, which neither the paper nor our fill models handle.
**Consequence.** Fill probabilities are per-step quantities. Queue position is lost at every step: an order re-posted at the same price joins the back of the queue, so the simulated passive fill rate is a lower bound on what a persistent order would get. Market orders are unaffected.

## D18 — One fresh simulator per parent order, common random numbers across policies
**Decision.** `run_many` creates a new `Simulator(params, seed=seed0 + i)` for order `i`, lets it run for `warmup` time units before the order arrives, and alternates the side (BUY for even `i`, SELL for odd `i`). The schedule is the same for all orders.
**Why.** A fresh simulator makes orders independent samples, so means and standard errors over orders are meaningful. The same seeds across policies give common random numbers: two policies face the same order flow until their own orders perturb it, which cuts the variance of the *difference* between them. Alternating sides cancels any drift of the mid in the average arrival slippage, so a policy cannot look good because the price happened to move its way. The warm-up removes the artificial seeded state (closed spread, exact equilibrium depths, see D9 and D13).
**Consequence.** Comparisons between policies are paired by order index. Results depend on `seed0` only through sampling noise, and reproducibility is exact for a given `(params, seed0, n_orders)`. Costs across orders are not comparable between sides without the sign convention `phi = side.value` in the metrics.

## D19 — The simulated flow has no intraday profile, so VWAP coincides with TWAP
**Decision.** All rates in `SimParams` are constant in time. The expected market volume is therefore linear in time, and `from_volume_profile` with the simulator's own volume reduces to `twap`.
**Why.** Intraday seasonality (the U-shaped volume curve) is a separate modelling choice with its own parameters; it is not needed to study fill probabilities and schedule following.
**Consequence.** Experiments use TWAP and Almgren–Chriss as schedules. `from_volume_profile` is implemented and tested so that a time-varying `mu(t)` can be added later without touching the execution code. The paper's TWAP/VWAP distinction cannot be reproduced here.

## D20 — Candidate limit prices are our own touch and the levels behind it; pi is an expected executed fraction
**Decision.** At a decision time the policy may post limit orders at `n_levels` candidate prices: level j is j ticks behind our own best (BUY: best_bid − j; SELL: best_ask + j), j = 0..n_levels−1. A market order is the extra candidate with pi = 1. The fill probability of level j is defined as the expected fraction of an order of `qty` shares posted now at level j that executes before the next decision (one step, D17).
**Why.** The paper's pi is the probability of a binary fill of a fixed-size order; in a simulator with partial fills the natural analogue, and the quantity the QP's expectation actually needs, is the executed fraction. Posting inside the spread is excluded for now (the spread is one tick 95% of the time).
**Consequence.** pi depends on `qty` and `dt`; both are fixed per experiment. The paper's nested-fill covariance min(pi_i, pi_j) − pi_i pi_j assumes binary fills; with fractions it is an upper bound, measured in notebook 03.

## D21 — Fill samples are collected by posting all candidate levels at once, in one long simulation
**Decision.** `collect_fill_samples` runs a single simulator, and at each trial posts one order of `qty` at every candidate level simultaneously, steps `dt`, records the executed fraction per level and cancels the rest (D17). Trials alternate sides.
**Why.** This is exactly what MPC does, so the nested structure between levels (level j can only execute after level j−1 is gone, by price priority) is measured as the policy will experience it. One long simulation gives trials that are weakly dependent but costs no warm-up per trial.
**Consequence.** Our resting orders (n_levels × qty shares) perturb the book slightly; trials are not independent, so standard errors are approximate. Within a trial the executed fraction is non-increasing in j, exactly.