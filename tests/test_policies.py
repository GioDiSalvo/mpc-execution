import pytest
from mpc_execution.book import Side
from mpc_execution.simulator import Simulator, SimParams
from mpc_execution.policies import Crossing, OrderSpec

@pytest.fixture
def state():
    return Simulator(SimParams(), seed=0).snapshot()

@pytest.mark.parametrize("q", [50, 150])
def test_behind_schedule_sends_the_gap_as_one_market_order(state, q):
    specs = Crossing().decide(state, Side.BUY, q, target=250, remaining=400)
    assert specs == [OrderSpec("market", 250 - q)]

@pytest.mark.parametrize("q", [250, 300])
def test_on_or_ahead_of_schedule_sends_nothing(state, q):
    assert Crossing().decide(state, Side.BUY, q=q, target=250, remaining=500 - q) == []

def test_never_sends_more_than_remaining(state):
    specs = Crossing().decide(state, Side.SELL, q=100, target=250, remaining=120)
    assert specs == [OrderSpec("market", 120)]
