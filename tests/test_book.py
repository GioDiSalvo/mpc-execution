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

def test_crossing_limit_not_yet_supported():
    b = LimitOrderBook()
    b.add_limit(Side.BUY, 99, 20)
    with pytest.raises(NotImplementedError):
        b.add_limit(Side.SELL, 99, 15)

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
    for _ in range(2000):
        side = rng.choice([Side.BUY, Side.SELL])
        price = rng.randint(90, 110)
        qty = rng.randint(1,100)
        try:
            b.add_limit(side, price, qty)
        except NotImplementedError:
            pass # until future changes: then remove try
        b.check_invariants()

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
