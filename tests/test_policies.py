import pytest
from mpc_execution.book import Side
from mpc_execution.simulator import Simulator, SimParams
from mpc_execution.policies import Crossing, OrderSpec, ThresholdRule, Traced

CASES = [
    # name,                  ahead, q,   target, remaining, expected (kind, qty)
    ("within_tolerance",     0,     100, 125,    400,       [("limit", 25)]),
    ("beyond_tolerance",     0,     100, 175,    400,       [("market", 25), ("limit", 50)]),
    ("on_schedule_no_ahead", 0,     100, 100,    400,       []),
    ("on_schedule_ahead",    25,    100, 100,    400,       [("limit", 25)]),
    ("last_step",            25,    475, 500,    25,        [("market", 25)]),
    ("capped_by_remaining",  25,    490, 495,    10,        [("limit", 10)]),
]

@pytest.mark.parametrize("name,ahead,q,target,remaining,expected", CASES, ids=[c[0] for c in CASES])
def test_threshold_rule(state, name, ahead, q, target, remaining, expected):
    orders = ThresholdRule(tau=50, ahead=ahead).decide(state, Side.BUY, q, target, remaining)
    want = [OrderSpec(kind, qty, state.best_bid if kind == "limit" else None) for kind, qty in expected]
    assert orders == want

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

def test_traced_records_decisions_without_changing_them(state):
    traced = Traced(Crossing())
    assert traced.decide(state, Side.BUY, 0, 25, 500) == Crossing().decide(state, Side.BUY, 0, 25, 500)
    assert len(traced.log) == 1

@pytest.mark.parametrize("case, ahead, q, target, remaining, expected", CASES, ids=[c[0] for c in CASES])
def test_threshold_rule(state, case, ahead, q, target, remaining, expected):
    orders = ThresholdRule(tau=50, ahead=ahead).decide(state, Side.BUY, q, target, remaining)
    wanted = [OrderSpec(kind, gap, state.best_bid if kind == "limit" else None) for kind, gap in expected]
    assert orders == wanted

@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_threshold_rule_posts_at_its_own_touch(state, side):
    touch = state.best_bid if side == Side.BUY else state.best_ask
    orders = ThresholdRule(tau=50, ahead=0).decide(state, side, 100, 125, 400)
    assert orders == [OrderSpec("limit", 25, touch)]