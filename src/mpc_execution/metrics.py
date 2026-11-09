from dataclasses import dataclass
import numpy as np
from mpc_execution.book import Side
from mpc_execution.simulator import Execution

@dataclass(frozen=True)
class OrderResult:
    """Record of one parent order, from which all metrics are computed.
    n = len(schedule) - 1 is the number of decision steps; index k = 0..n
    refers to the decision times t_0 (arrival) < t_1 < ... < t_n (end).
    """
    side: Side
    Q: int                                        # parent order size, shares
    schedule: np.ndarray                          # s_k, (n+1,) cumulative target, s_0 = 0, s_n = Q
    filled_path: np.ndarray                       # q_k, (n+1,) cumulative shares filled by t_k, q_0 = 0 (aka positions)
    mids: np.ndarray                              # (n+1,) mid at t_k; mids[0] is the arrival price
    executions: list[Execution]                   # our fills, in time order, aggressive and passive
    market_trades: list[tuple[float, int, int]]   # all market trades (t, price, qty) in [t_0, t_n], ours included

    @property
    def n_steps(self) -> int:
        return len(self.schedule) - 1

    @property
    def filled(self) -> int:
        'sum of execution qty'
        return sum(e.qty for e in self.executions)

    @property
    def arrival_mid(self) -> float:
        'mid at k = 0'
        return float(self.mids[0])

def fwap(r: OrderResult) -> float:
    'quantity-weighted mean of our execution prices'
    if not r.executions:
        return float("nan")    
    return sum(e.price * e.qty for e in r.executions) / r.filled

def market_vwap(r: OrderResult) -> float:
    'quantity-weighted mean of all market trades'
    if not r.market_trades:
        return float("nan")
    return sum(p * q for _, p, q in r.market_trades) / sum(q for _, _, q in r.market_trades)

def swap(r: OrderResult) -> float:
    'sum_k (s_{k+1} - s_k) * mid_k / Q   (§3.4: executing the slice at the mid of each decision time)'
    return float(np.diff(r.schedule) @ r.mids[:-1]) / r.Q

def arrival_slippage_bps(r: OrderResult) -> float:
    '1e4 * phi * (fwap - arrival_mid) / arrival_mid,  phi = side.value'
    f = fwap(r)
    mid = r.arrival_mid
    phi = r.side.value
    return 1e4 * phi * (f - mid) / mid

def vwap_slippage_bps(r: OrderResult) -> float:
    '1e4 * phi * (fwap - vwap) / vwap,  phi = side.value'
    f = fwap(r)
    mv = market_vwap(r)
    phi = r.side.value
    return 1e4 * phi * (f - mv) / mv

def schedule_shortfall_bps(r: OrderResult) -> float:
    '1e4 * phi * (fwap - swap) / swap,  phi = side.value'
    f = fwap(r)
    s = swap(r)
    phi = r.side.value
    return 1e4 * phi * (f - s) / s

def completion(r: OrderResult) -> float:
    'filled / Q'
    return r.filled / r.Q

def deviation(r: OrderResult) -> np.ndarray:
    '# q_k - s_k'
    return r.filled_path - r.schedule

def summarize(r: OrderResult) -> dict:
    summary = {
        "side": r.side.name, "Q": r.Q, "filled": r.filled, "completion": completion(r),
        "fwap": fwap(r), "market_vwap": market_vwap(r), "swap": swap(r),
        "arrival_slippage_bps": arrival_slippage_bps(r),
        "vwap_slippage_bps": vwap_slippage_bps(r),
        "schedule_shortfall_bps": schedule_shortfall_bps(r),
        "max_abs_deviation": int(np.max(np.abs(deviation(r)))),
        "final_deviation": int(deviation(r)[-1])
    }
    return summary