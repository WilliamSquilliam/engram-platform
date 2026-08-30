"""Pure pricing-model unit tests (no app/DB)."""
from app import metrics


def test_cartridge_cost_positive_and_flat():
    assert metrics.price_everyday() > 0


def test_rag_cost_decreases_with_volume():
    # Fixed vector-DB cost amortizes over more queries.
    assert metrics.rag_cost(1_000) > metrics.rag_cost(1_000_000)


def test_training_cost():
    assert round(metrics.training_cost(3600, 1.86), 2) == 1.86
    assert metrics.training_cost(None, 1.86) == 0.0


def test_breakeven():
    # alt 0.01, cart 0.001 -> saving 0.009/query; cost 1.86 -> ~207 queries.
    be = metrics.breakeven_queries(1.86, 0.01, 0.001)
    assert round(be) == round(1.86 / 0.009)
    # No saving (cart pricier than alt) -> None.
    assert metrics.breakeven_queries(1.86, 0.001, 0.01) is None
    # Infeasible alternative -> None.
    assert metrics.breakeven_queries(1.86, None, 0.001) is None
