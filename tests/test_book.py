from mpc_execution.book import LimitOrderBook

def test_empty_book_has_no_prices():
    b = LimitOrderBook()
    assert b.best_bid() is None
    assert b.best_ask() is None
    assert b.mid() is None
    assert b.spread() is None