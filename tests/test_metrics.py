from mpc_execution.simulator import Execution
from mpc_execution.metrics import OrderResult, fwap, market_vwap, swap, arrival_slippage_bps, vwap_slippage_bps, schedule_shortfall_bps, completion, summarize, deviation
from mpc_execution.book import Side
import numpy as np
import math
import pytest

def make(side=Side.BUY, executions=None, filled_path=(0, 40, 90)):
    if executions is None:
        executions = [
            Execution(1.0, side, 10001, 30, False),
            Execution(3.0, side, 10000, 10, True),
            Execution(7.0, side, 10003, 50, False)
            ]

    return OrderResult(
        side=side, Q=100,
        schedule=np.array([0, 50, 100]),
        filled_path=np.array(filled_path),
        mids=np.array([10_000.0, 10_001.0,10_002.5]),
        executions=executions,
        market_trades=[(0.5, 10000, 30), (1.0, 10001, 30), (3.0, 10000, 10),
                       (6.0, 10006, 30), (7.0, 10003, 50)],
        )

def test_prices():
    r = make()
    assert r.filled == 90
    assert fwap(r) == 10002.0
    assert market_vwap(r) == pytest.approx(10002.4)
    assert swap(r) == 10000.5

def test_slippages_buy_positive_is_cost():
    r = make()
    assert arrival_slippage_bps(r) == pytest.approx(2.0)
    assert vwap_slippage_bps(r) == pytest.approx(-0.3999, abs=1e-3)
    assert schedule_shortfall_bps(r) == pytest.approx(1.4999, abs=1e-3)

def test_sell_flips_sign():
    assert arrival_slippage_bps(make(Side.SELL)) == pytest.approx(-2.0)

def test_completion_and_deviation():
    r = make()
    assert completion(r) == 0.9
    assert np.array_equal(deviation(r), [0, -10, -10])
    s = summarize(r)
    assert s["max_abs_deviation"] == 10 and s["final_deviation"] == -10

def test_no_fills_gives_nan_not_error():
    r = make(executions=[], filled_path=(0, 0, 0))
    assert math.isnan(fwap(r)) and math.isnan(arrival_slippage_bps(r))
    assert completion(r) == 0
    assert math.isnan(summarize(r)["schedule_shortfall_bps"])