"""Unit tests for individual cleaning functions on small hand-made frames."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.clean import (
    drop_duplicate_orders, drop_empty_orders, drop_impossible_dates, drop_orphans,
    fix_negative_quantities, fix_price_outliers, match_canonical, net_revenue, normalize_text,
)
from src.common import known_date_formats, parse_mixed_dates


def test_normalize_text_trims_and_collapses_whitespace() -> None:
    result = normalize_text(pd.Series(["  Dhaka ", "Cox's   Bazar", np.nan]))
    assert result.iloc[0] == "Dhaka"
    assert result.iloc[1] == "Cox's Bazar"
    assert pd.isna(result.iloc[2])


def test_match_canonical_handles_case_alias_fuzzy_and_unknown() -> None:
    canonical = ["Chattogram", "Gazipur", "Home & Kitchen", "Cox's Bazar"]
    aliases = {"Chittagong": "Chattogram"}
    values = pd.Series(["GAZIPUR", "Chittagong", "Gzipur", "Home and Kitchen", "Coxs Bazar", "Atlantis", np.nan])
    result = match_canonical(values, canonical, aliases, cutoff=0.8)
    assert result.iloc[:5].tolist() == ["Gazipur", "Chattogram", "Gazipur", "Home & Kitchen", "Cox's Bazar"]
    assert pd.isna(result.iloc[5])
    assert pd.isna(result.iloc[6])


def test_parse_mixed_dates_reads_every_known_format_without_swapping_day_and_month(cfg: dict) -> None:
    values = pd.Series([
        "2024-03-05 10:15:00", "05/03/2024 10:15", "05-Mar-2024 10:15:00", "2024/03/05 10:15:00", "2024-02-30 10:15:00",
    ])
    parsed = parse_mixed_dates(values, known_date_formats(cfg))
    assert (parsed.iloc[:4].dt.date == pd.Timestamp("2024-03-05").date()).all()
    assert pd.isna(parsed.iloc[4])


def test_drop_duplicate_orders_removes_exact_copies_and_reused_ids() -> None:
    orders = pd.DataFrame({"order_id": [1, 1, 2, 2, 3], "customer_id": [10, 10, 20, 21, 30]})
    result = drop_duplicate_orders(orders)
    assert result["order_id"].tolist() == [1, 2, 3]


def test_drop_impossible_dates_keeps_only_the_trading_window() -> None:
    orders = pd.DataFrame({"order_date": pd.to_datetime(
        ["2023-01-01 00:00:00", "2025-12-31 23:59:59", "2031-05-01 00:00:00", None]
    )})
    result = drop_impossible_dates(orders, pd.Timestamp("2023-01-01"), pd.Timestamp("2025-12-31"))
    assert len(result) == 2


def test_drop_orphans_keeps_rows_with_known_parent() -> None:
    child = pd.DataFrame({"customer_id": [1, 2, 999]})
    assert drop_orphans(child, "customer_id", pd.Series([1, 2]))["customer_id"].tolist() == [1, 2]


def test_fix_negative_quantities_takes_absolute_value() -> None:
    items = pd.DataFrame({"quantity": [2, -3, 1]})
    assert fix_negative_quantities(items)["quantity"].tolist() == [2, 3, 1]


def test_fix_price_outliers_replaces_only_extreme_ratios() -> None:
    products = pd.DataFrame({"product_id": [1, 2, 3], "list_price": [100.0, 200.0, 50.0]})
    items = pd.DataFrame({"product_id": [1, 2, 3], "unit_price": [10000.0, 210.0, 2.0]})
    result = fix_price_outliers(items, products, ratio=10)
    assert result["unit_price"].tolist() == [100.0, 210.0, 50.0]


def test_drop_empty_orders_removes_orders_without_lines() -> None:
    orders = pd.DataFrame({"order_id": [1, 2]})
    items = pd.DataFrame({"order_id": [1, 1]})
    assert drop_empty_orders(orders, items)["order_id"].tolist() == [1]


def test_net_revenue_counts_delivered_orders_after_discount() -> None:
    orders = pd.DataFrame({
        "order_id": [1, 2], "order_status": ["delivered", "cancelled"], "discount_pct": [0.10, 0.0],
    })
    items = pd.DataFrame({"order_id": [1, 1, 2], "quantity": [2, 1, 5], "unit_price": [100.0, 50.0, 999.0]})
    assert net_revenue(orders, items) == 225.0
