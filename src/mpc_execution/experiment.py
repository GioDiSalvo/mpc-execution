from mpc_execution.book import Side
from mpc_execution.simulator import Simulator, SimParams
from mpc_execution.metrics import OrderResult
from mpc_execution.policies import Policy
import numpy as np

def run_order(sim: Simulator, policy: Policy, side: Side, Q: int, schedule: np.ndarray, dt: float) -> OrderResult:
    """Execute one parent order of Q shares over len(schedule)-1 steps of length dt. Our orders live one step (D17)."""
    n = len(schedule) - 1
    q = 0
    executions, filled_path, mids = [], [0], [sim.snapshot().mid]
    trades_before = len(sim.trades)
    for k in range(n):
        state = sim.snapshot()
        for o in policy.decide(state, side, q, int(schedule[k + 1]), Q - q):
            if o.kind == "market":
                _, ex = sim.place_market(side, o.qty)
            else:
                assert o.price is not None
                _, ex = sim.place_limit(side, o.price, o.qty)
            executions += ex
        executions += sim.step(dt)
        sim.cancel_all_mine()                      # D17: unfilled limit orders die at the end of the step
        q = sum(e.qty for e in executions)
        filled_path.append(q)
        mids.append(state.mid)            # t_n
    return OrderResult(side, Q, schedule, filled_path=np.array(filled_path), mids=np.array(mids),
                       executions=executions, market_trades=sim.trades[trades_before:])

def run_many(params: SimParams, policy_factory, dt: float,
             schedule: np.ndarray, n_orders: int, seed0: int = 0, warmup: float = 50.0) -> list[OrderResult]:
    """One fresh simulator per order (seed = seed0 + i), warmed up for `warmup` time units,
    sides alternating BUY/SELL. Returns one row per order with summarize(result)."""
    results = []
    Q = int(schedule[-1])
    for i in range(n_orders):
        sim = Simulator(params, seed=seed0 + i)
        sim.step(warmup)
        side = Side.BUY if i%2==0 else Side.SELL
        results.append(run_order(sim, policy_factory(), side, Q, schedule, dt))
    return results