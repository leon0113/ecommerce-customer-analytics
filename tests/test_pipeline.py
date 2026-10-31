"""End-to-end checks of the cleaned data against the generator's ground-truth issue log."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.clean import net_revenue
from tests.conftest import issue_keys


def test_order_row_counts_match_ground_truth(raw: dict, tables: dict, issues: pd.DataFrame) -> None:
    raw_order_ids = set(raw["orders"]["order_id"])
    dropped = issue_keys(issues, "orders", "impossible_date") | issue_keys(issues, "orders", "orphan_fk")
    orphan_lines = issue_keys(issues, "order_items", "orphan_fk")
    items = raw["order_items"]
    remaining_lines = items[~items["order_item_id"].isin(orphan_lines)]
    emptied = set(items["order_id"]) - set(remaining_lines["order_id"])

    assert set(tables["orders"]["order_id"]) == raw_order_ids - dropped - emptied
    assert len(raw["orders"]) - len(tables["orders"]) == len(issue_keys(issues, "orders", "duplicate_row")) + len(
        dropped | emptied
    )


def test_order_ids_are_unique(tables: dict) -> None:
    assert tables["orders"]["order_id"].is_unique
    assert tables["payments"]["order_id"].is_unique


def test_every_foreign_key_resolves(tables: dict) -> None:
    pairs = [
        ("orders", "customer_id", "customers"),
        ("order_items", "order_id", "orders"),
        ("order_items", "product_id", "products"),
        ("products", "category_id", "categories"),
        ("payments", "order_id", "orders"),
        ("returns", "order_item_id", "order_items"),
    ]
    for child, column, parent in pairs:
        assert tables[child][column].isin(tables[parent][column]).all(), f"{child}.{column}"


def test_dates_parsed_and_orders_in_window(tables: dict, cfg: dict) -> None:
    for table, column in [("customers", "signup_date"), ("orders", "order_date"),
                          ("payments", "payment_date"), ("returns", "return_date")]:
        assert tables[table][column].notna().all(), f"{table}.{column}"
    order_dates = tables["orders"]["order_date"]
    assert order_dates.min() >= pd.Timestamp(cfg["dates"]["start"])
    assert order_dates.max() < pd.Timestamp(cfg["dates"]["end"]) + pd.Timedelta(days=1)


def test_text_columns_hold_only_canonical_values(tables: dict, cfg: dict) -> None:
    divisions = cfg["geography"]["divisions"]
    districts = {d for division in divisions.values() for d in division["districts"]}
    unknown = cfg["cleaning"]["unknown_label"]
    assert set(tables["customers"]["division"]) <= set(divisions)
    assert set(tables["customers"]["district"]) <= districts | {unknown}
    assert set(tables["orders"]["delivery_district"]) <= districts | {unknown}
    assert set(tables["payments"]["payment_method"]) <= set(cfg["payments"]["methods"])


def test_null_rules(tables: dict, issues: pd.DataFrame, cfg: dict) -> None:
    unknown = cfg["cleaning"]["unknown_label"]
    customers = tables["customers"]
    nullable = {"customers": {"phone"}}
    for name, table in tables.items():
        required = set(table.columns) - nullable.get(name, set())
        assert table[list(required)].notna().all().all(), name
    assert (customers["district"] == unknown).sum() == len(issue_keys(issues, "customers", "missing_value", "district"))
    assert (customers["gender"] == unknown).sum() == len(issue_keys(issues, "customers", "missing_value", "gender"))
    assert customers["phone"].dropna().str.fullmatch(r"01[3-9]\d{8}").all()


def test_injected_value_errors_are_restored(tables: dict, issues: pd.DataFrame) -> None:
    items = tables["order_items"].set_index("order_item_id")
    for issue_type, column in [("negative_quantity", "quantity"), ("price_outlier", "unit_price")]:
        logged = issues[(issues["table"] == "order_items") & (issues["issue_type"] == issue_type)]
        logged = logged[logged["row_key"].isin(items.index)]
        original = logged.set_index("row_key")["original_value"].astype(float)
        assert np.allclose(items.loc[original.index, column].abs(), original.abs())
    assert (items["quantity"] > 0).all()


def test_category_typos_restored(tables: dict, raw: dict, issues: pd.DataFrame) -> None:
    altered = issues[(issues["table"] == "products") & (issues["column"] == "category_name")]
    expected = altered.set_index("row_key")["original_value"]
    categories = tables["categories"].set_index("category_id")["category_name"]
    products = tables["products"].set_index("product_id")
    actual = products.loc[expected.index, "category_id"].map(categories)
    assert (actual == expected).all()


def test_cleaning_log_explains_every_revenue_change(raw: dict, tables: dict, cleaning_log: pd.DataFrame) -> None:
    """Raw revenue plus each logged step's revenue change equals clean revenue (the revenue bridge)."""
    revenue_steps = cleaning_log.dropna(subset=["revenue_before"])
    raw_revenue = net_revenue(raw["orders"], raw["order_items"])
    clean_revenue = net_revenue(tables["orders"], tables["order_items"])
    bridged = raw_revenue + (revenue_steps["revenue_after"] - revenue_steps["revenue_before"]).sum()
    assert np.isclose(revenue_steps["revenue_before"].iloc[0], raw_revenue, atol=0.01)
    assert np.isclose(bridged, clean_revenue, atol=0.01)


def test_clean_revenue_reconciles_with_payments(raw: dict, tables: dict, issues: pd.DataFrame) -> None:
    """Order totals rebuilt from cleaned lines equal what the payment system recorded.

    Payments were captured independently of the order lines, so this catches any cleaning
    step that changed revenue wrongly (a missed duplicate, outlier or sign error). Orders
    that lost an orphan line are excluded, since their payment still includes that line.
    """
    orders, items, payments = tables["orders"], tables["order_items"], tables["payments"]
    lines = items.merge(orders[["order_id", "discount_pct"]], on="order_id")
    lines["net"] = lines["quantity"] * lines["unit_price"] * (1 - lines["discount_pct"])
    order_totals = lines.groupby("order_id")["net"].sum()

    raw_items = raw["order_items"]
    orphan_lines = issue_keys(issues, "order_items", "orphan_fk")
    lost_a_line = raw_items.loc[raw_items["order_item_id"].isin(orphan_lines), "order_id"].unique()
    comparable = order_totals.index.difference(lost_a_line)
    paid = payments.set_index("order_id")["amount"]

    assert len(comparable) >= 0.99 * len(order_totals)
    assert np.allclose(order_totals.loc[comparable], paid.loc[comparable], atol=0.01)
    assert np.isclose(order_totals.loc[comparable].sum(), paid.loc[comparable].sum(), atol=0.01 * len(comparable))
