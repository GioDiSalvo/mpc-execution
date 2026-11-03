from mpc_execution.book import LimitOrderBook, Side
import pytest
import random

def test_empty_book_has_no_prices():
    b = LimitOrderBook()
    assert b.best_bid() is None
    assert b.best_ask() is None
    assert b.mid() is None
    assert b.spread() is None

def test_passive_limit_orders_set_price():
   b = LimitOrderBook()
   b.add_limit(Side.BUY, 98, 50)
   b.add_limit(Side.SELL, 101, 30)
   assert b.best_ask() == 101
   assert b.best_bid() == 98
   assert b.spread() == 3
   assert b.mid() == 99.5

def test_same_price_orders_in_queue_in_arrival_order():
    b = LimitOrderBook()
    id1, _ = b.add_limit(Side.BUY, 100, 30)
    id2, _ = b.add_limit(Side.BUY, 100, 20)
    queue = b.bids[100]
    assert [o.id for o in queue] == [id1, id2]

def test_zero_quantity_is_rejected():
    b = LimitOrderBook()
    with pytest.raises(ValueError):
        b.add_limit(Side.BUY, 99, 0)

def test_order_does_not_cross_when_opposite_side_is_empty():
    b = LimitOrderBook()
    b.add_limit(Side.SELL, 1000, 20)
    assert b.best_bid() is None
    assert b.best_ask() == 1000
    assert b.mid() is None

def test_book_never_crosses_under_random_operations():
    rng = random.Random(0)
    b = LimitOrderBook()
    live_ids = []
    for _ in range(5000):
        side = rng.choice([Side.BUY, Side.SELL])
        r = rng.random()
        if r < 0.7:
            o_id, _ = b.add_limit(side, rng.randint(90, 110), rng.randint(1, 100))
            if o_id is not None:
                live_ids.append(o_id)
        elif r < 0.85:
            b.add_market(side, rng.randint(1, 40))
        elif live_ids:
            o_id = live_ids.pop(rng.randrange(len(live_ids)))
            if o_id in b.orders:
                b.cancel(o_id)
        b.check_invariants()
    assert len(b.orders) > 0, f"book ended empty: the test exercised nothing"

def test_queue_ahead_counts_better_prices_and_earlier_orders_at_same_price():
    b = LimitOrderBook()
    id_top, _ = b.add_limit(Side.BUY, 99, 10)
    id_mid, _ = b.add_limit(Side.BUY, 98, 22)
    id_last, _ = b.add_limit(Side.BUY, 98, 37)
    assert b.queue_ahead(id_last) == 10 + 22
    assert b.queue_ahead(id_mid) == 10
    assert b.queue_ahead(id_top) == 0

    assert b.cancel(id_mid) == 22
    assert b.queue_ahead(id_last) == 10
    b.check_invariants()

def test_cancelling_last_order_at_a_level_removes_the_level():
    b = LimitOrderBook()
    o_id, _ = b.add_limit(Side.SELL, 104, 14)
    b.add_limit(Side.SELL, 105, 11)
    assert b.cancel(o_id) == 14
    assert 104 not in b.asks
    assert b.best_ask() == 105
    b.check_invariants()

def test_cancel_or_queue_ahead_of_non_existing_order_raises_error():
    b = LimitOrderBook()
    id1, _ = b.add_limit(Side.BUY, 99, 10)
    with pytest.raises(KeyError):
        b.cancel(id1 + 1)
    id2, _ = b.add_limit(Side.SELL, 102, 14)
    with pytest.raises(KeyError):
        b.queue_ahead(id2 + 1)
    b.check_invariants()

def test_marketable_order_fills_at_resting_price():
    b = LimitOrderBook()
    b.add_limit(Side.SELL, 102, 50)
    _, fills = b.add_limit(Side.BUY, 105, 20)
    assert b.best_bid() is None
    assert len(fills) == 1
    fill = fills[0]
    assert fill.price == 102
    assert fill.qty == 20
    assert b.asks[102][0].qty == 30
    b.check_invariants()

def test_market_order_sweeps_levels_in_price_order():
    b = LimitOrderBook()
    b.add_limit(Side.SELL, 101, 10)
    b.add_limit(Side.SELL, 103, 20)
    b.add_limit(Side.SELL, 102, 30)
    _, fills = b.add_market(Side.BUY, 45)
    assert len(fills) == 3
    total = sum(f.qty for f in fills)
    avg = sum(f.qty * f.price for f in fills) / total
    assert total == 45
    assert avg == pytest.approx((101 * 10 + 102 * 30 + 103 * 5) / 45)
    assert fills[0].price == 101 and fills[0].qty == 10
    assert fills[1].price == 102 and fills[1].qty == 30
    assert fills[2].price == 103 and fills[2].qty == 5
    assert b.asks[103][0].price == b.best_ask() == 103 and b.asks[103][0].qty == 15
    assert b.mid() is None
    b.check_invariants()

def test_fills_respect_time_priority_at_same_price():
    b = LimitOrderBook()
    id_first, _ = b.add_limit(Side.SELL, 102, 30)
    id_second, _ = b.add_limit(Side.SELL, 102, 30)
    qty, fills = b.add_market(Side.BUY, 40)
    assert id_first not in b.orders
    assert fills[0].order_id == id_first
    assert fills[1].order_id == id_second
    assert qty == 0
    assert b.asks[102][0].qty == 20
    assert b.asks[102][0].id == id_second
    b.check_invariants()

def test_limit_order_stops_at_its_limit_and_rests_residual():
    b = LimitOrderBook()
    b.add_limit(Side.SELL, 101, 30)
    b.add_limit(Side.SELL, 103, 50)
    _, fills = b.add_limit(Side.BUY, 102, 85)
    assert b.best_bid() == 102 and b.best_ask() == 103
    assert b.bids[102][0].qty == 55
    assert b.asks[103][0].qty == 50
    assert len(fills) == 1
    assert fills[0].price == 101 and fills[0].qty == 30
    assert 101 not in b.asks                 # level 101 gone
    assert b.best_bid() < b.best_ask()
    b.check_invariants()

def test_market_order_with_empty_opposite_side_executes_nothing():
    b = LimitOrderBook()
    qty, fills = b.add_market(Side.BUY, 10)
    assert qty == 10
    assert len(fills) == 0
    b.check_invariants()

def test_sell_side_matching_mirrors_buy_side():
    b = LimitOrderBook()
    b.add_limit(Side.BUY, 99, 10)
    b.add_limit(Side.BUY, 95, 20)
    b.add_limit(Side.BUY, 98, 30)
    _, fills = b.add_market(Side.SELL, 45)
    assert len(fills) == 3
    total = sum(f.qty for f in fills)
    avg = sum(f.qty * f.price for f in fills) / total
    assert total == 45
    assert avg == pytest.approx((99 * 10 + 98 * 30 + 95 * 5) / 45)
    assert fills[0].price == 99 and fills[0].qty == 10
    assert fills[1].price == 98 and fills[1].qty == 30
    assert fills[2].price == 95 and fills[2].qty == 5
    assert b.bids[95][0].price == 95 and b.bids[95][0].qty == 15
    assert b.mid() is None and b.best_ask() is None 
    b.check_invariants()

def test_repr_shows_top_levels_and_handles_empty_book():
    b = LimitOrderBook()
    assert "empty" in repr(b)
    b.add_limit(Side.BUY, 100, 30)
    b.add_limit(Side.BUY, 100, 20)
    b.add_limit(Side.SELL, 102, 10)
    r = repr(b)
    assert "100 x 50 (2)" in r
    assert "102 x 10 (1)" in r
    assert "bid/ask 100/102" in r