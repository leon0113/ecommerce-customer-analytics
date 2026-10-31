"""Clean the raw CSVs into analysis-ready tables.

Every step logs rows before and after. Steps on orders and order lines also record net
revenue before and after, so the gap between raw and clean revenue can be explained
step by step. Run with: python -m src.clean
"""

from __future__ import annotations

import difflib
import logging

import numpy as np
import pandas as pd

from src.common import (
    known_date_formats, load_config, load_tables, parse_mixed_dates, project_path, setup_logging,
)

logger = logging.getLogger(__name__)

PRODUCT_COLUMNS = ["product_id", "product_name", "category_id", "brand", "list_price"]


class CleaningLog:
    """One row per cleaning step: table, step, row counts and (for order steps) net revenue."""

    def __init__(self) -> None:
        self._rows: list[dict] = []

    def record(
        self,
        table: str,
        step: str,
        rows_before: int,
        rows_after: int,
        rows_changed: int,
        revenue_before: float | None = None,
        revenue_after: float | None = None,
    ) -> None:
        """Store a step and log it."""
        self._rows.append({
            "table": table, "step": step, "rows_before": rows_before, "rows_after": rows_after,
            "rows_changed": int(rows_changed), "revenue_before": revenue_before, "revenue_after": revenue_after,
        })
        logger.info("%-11s %-45s rows %5d -> %5d  changed %4d", table, step, rows_before, rows_after, rows_changed)

    def to_frame(self) -> pd.DataFrame:
        """The log as a table, for the data quality notebook."""
        return pd.DataFrame(self._rows)


def normalize_text(values: pd.Series) -> pd.Series:
    """Trim and collapse repeated whitespace; missing values stay missing."""
    return values.str.strip().str.replace(r"\s+", " ", regex=True)


def match_canonical(values: pd.Series, canonical: list[str], aliases: dict[str, str], cutoff: float) -> pd.Series:
    """Map messy names to canonical spellings: case-insensitive exact match, known alias, then fuzzy match.

    Fuzzy matching runs once per distinct value, not per row. Values with no match become NaN.
    """
    cleaned = normalize_text(values).str.lower().str.replace(" and ", " & ", regex=False)
    lookup = {name.lower(): name for name in canonical}
    lookup |= {alias.lower(): name for alias, name in aliases.items()}
    for value in cleaned.dropna().unique():
        if value not in lookup:
            close = difflib.get_close_matches(value, list(lookup), n=1, cutoff=cutoff)
            if close:
                lookup[value] = lookup[close[0]]
    return cleaned.map(lookup)


def count_changed(before: pd.Series, after: pd.Series) -> int:
    """Number of values that differ, treating two missing values as equal."""
    return int((before.astype("string").fillna("<NA>") != after.astype("string").fillna("<NA>")).sum())


def standardize_column(
    df: pd.DataFrame, column: str, canonical: list[str], cfg: dict, log: CleaningLog, table: str
) -> pd.DataFrame:
    """Replace a text column with canonical spellings and log how many values changed."""
    clean_cfg = cfg["cleaning"]
    out = df.copy()
    out[column] = match_canonical(df[column], canonical, clean_cfg["known_aliases"], clean_cfg["fuzzy_cutoff"])
    unmatched = int((df[column].notna() & out[column].isna()).sum())
    if unmatched:
        logger.warning("%s.%s: %d values matched no canonical name", table, column, unmatched)
    log.record(table, f"standardize {column}", len(df), len(out), count_changed(df[column], out[column]))
    return out


def fill_unknown(df: pd.DataFrame, column: str, label: str, log: CleaningLog, table: str) -> pd.DataFrame:
    """Fill missing values with an explicit label so they stay visible as their own group."""
    missing = int(df[column].isna().sum())
    out = df.assign(**{column: df[column].fillna(label)})
    log.record(table, f"fill missing {column} with '{label}'", len(df), len(out), missing)
    return out


def parse_date_column(df: pd.DataFrame, column: str, cfg: dict, log: CleaningLog, table: str) -> pd.DataFrame:
    """Parse a date column written in mixed formats; values in no known format become NaT."""
    formats = known_date_formats(cfg)
    non_canonical = pd.to_datetime(df[column], format=formats[0], errors="coerce").isna()
    out = df.assign(**{column: parse_mixed_dates(df[column], formats)})
    unparsed = int(out[column].isna().sum())
    log.record(table, f"parse {column} ({unparsed} unparseable)", len(df), len(out), int(non_canonical.sum()))
    return out


def drop_orphans(child: pd.DataFrame, column: str, parent_keys: pd.Series) -> pd.DataFrame:
    """Keep only rows whose foreign key exists in the parent table."""
    return child[child[column].isin(parent_keys)]


def drop_duplicate_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows, then any remaining rows that reuse an order_id."""
    return orders.drop_duplicates().drop_duplicates("order_id", keep="first")


def drop_impossible_dates(orders: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Drop orders with an unparseable date or one outside the business's trading window."""
    in_window = orders["order_date"].between(start, end + pd.Timedelta(days=1), inclusive="left")
    return orders[in_window]


def fix_negative_quantities(items: pd.DataFrame) -> pd.DataFrame:
    """Treat negative quantities as sign errors; returns are recorded in their own table."""
    return items.assign(quantity=items["quantity"].abs())


def fix_price_outliers(items: pd.DataFrame, products: pd.DataFrame, ratio: float) -> pd.DataFrame:
    """Replace unit prices more than `ratio` times above or below list price with the list price."""
    list_price = items["product_id"].map(products.set_index("product_id")["list_price"])
    price_ratio = items["unit_price"] / list_price
    is_outlier = (price_ratio > ratio) | (price_ratio < 1 / ratio)
    return items.assign(unit_price=items["unit_price"].where(~is_outlier, list_price))


def drop_empty_orders(orders: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Drop orders that have no order lines left."""
    return orders[orders["order_id"].isin(items["order_id"])]


def net_revenue(orders: pd.DataFrame, items: pd.DataFrame) -> float:
    """Net revenue of delivered orders: quantity x unit price x (1 - order discount)."""
    delivered = orders.loc[orders["order_status"] == "delivered", ["order_id", "discount_pct"]]
    lines = items.merge(delivered, on="order_id")
    return float((lines["quantity"] * lines["unit_price"] * (1 - lines["discount_pct"])).sum())


def log_order_step(
    log: CleaningLog,
    table: str,
    step: str,
    before: tuple[pd.DataFrame, pd.DataFrame],
    after: tuple[pd.DataFrame, pd.DataFrame],
    rows_changed: int | None = None,
) -> None:
    """Record a step on orders or order lines with row counts and net revenue before and after."""
    position = 0 if table == "orders" else 1
    rows_before, rows_after = len(before[position]), len(after[position])
    log.record(
        table, step, rows_before, rows_after,
        rows_before - rows_after if rows_changed is None else rows_changed,
        round(net_revenue(*before), 2), round(net_revenue(*after), 2),
    )


def clean_customers(raw: pd.DataFrame, cfg: dict, log: CleaningLog) -> pd.DataFrame:
    """Canonical division and district names, parsed signup dates, explicit Unknown labels."""
    divisions = cfg["geography"]["divisions"]
    districts = [d for division in divisions.values() for d in division["districts"]]
    unknown = cfg["cleaning"]["unknown_label"]

    df = standardize_column(raw, "division", list(divisions), cfg, log, "customers")
    df = standardize_column(df, "district", districts, cfg, log, "customers")
    df = parse_date_column(df, "signup_date", cfg, log, "customers")
    df = fill_unknown(df, "district", unknown, log, "customers")
    df = fill_unknown(df, "gender", unknown, log, "customers")
    log.record("customers", "keep missing phone as null (not used)", len(df), len(df), int(df["phone"].isna().sum()))
    return df


def clean_products(raw: pd.DataFrame, categories: pd.DataFrame, cfg: dict, log: CleaningLog) -> pd.DataFrame:
    """Map category names to category_id and label missing brands."""
    df = standardize_column(raw, "category_name", categories["category_name"].tolist(), cfg, log, "products")
    df["category_id"] = df["category_name"].map(categories.set_index("category_name")["category_id"]).astype("int64")
    df = fill_unknown(df, "brand", cfg["cleaning"]["unknown_label"], log, "products")
    return df[PRODUCT_COLUMNS]


def clean_orders_and_items(
    raw_orders: pd.DataFrame,
    raw_items: pd.DataFrame,
    customers: pd.DataFrame,
    products: pd.DataFrame,
    cfg: dict,
    log: CleaningLog,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean orders and order lines together, since dropping an order removes its lines too."""
    start, end = pd.Timestamp(cfg["dates"]["start"]), pd.Timestamp(cfg["dates"]["end"])
    orders, items = raw_orders, raw_items

    before = (orders, items)
    orders = drop_duplicate_orders(orders)
    log_order_step(log, "orders", "drop duplicate orders", before, (orders, items))

    orders = parse_date_column(orders, "order_date", cfg, log, "orders")

    before = (orders, items)
    orders = drop_impossible_dates(orders, start, end)
    log_order_step(log, "orders", "drop impossible order dates", before, (orders, items))

    before = (orders, items)
    orders = drop_orphans(orders, "customer_id", customers["customer_id"])
    log_order_step(log, "orders", "drop orders with unknown customer_id", before, (orders, items))

    before = (orders, items)
    items = drop_orphans(items, "order_id", orders["order_id"])
    log_order_step(log, "order_items", "drop lines of removed orders", before, (orders, items))

    before = (orders, items)
    items = fix_negative_quantities(items)
    changed = int((before[1]["quantity"] != items["quantity"]).sum())
    log_order_step(log, "order_items", "negative quantity -> absolute value", before, (orders, items), changed)

    before = (orders, items)
    items = fix_price_outliers(items, products, cfg["cleaning"]["price_outlier_ratio"])
    changed = int((before[1]["unit_price"] != items["unit_price"]).sum())
    log_order_step(log, "order_items", "price outlier -> list price", before, (orders, items), changed)

    before = (orders, items)
    items = drop_orphans(items, "product_id", products["product_id"])
    log_order_step(log, "order_items", "drop lines with unknown product_id", before, (orders, items))

    before = (orders, items)
    orders = drop_empty_orders(orders, items)
    log_order_step(log, "orders", "drop orders left without lines", before, (orders, items))

    districts = [d for division in cfg["geography"]["divisions"].values() for d in division["districts"]]
    orders = standardize_column(orders, "delivery_district", districts, cfg, log, "orders")
    home_district = orders["customer_id"].map(customers.set_index("customer_id")["district"])
    missing = int(orders["delivery_district"].isna().sum())
    orders = orders.assign(delivery_district=orders["delivery_district"].fillna(home_district))
    log.record("orders", "fill missing delivery_district from customer", len(orders), len(orders), missing)
    return orders.reset_index(drop=True), items.reset_index(drop=True)


def clean_payments(raw: pd.DataFrame, orders: pd.DataFrame, cfg: dict, log: CleaningLog) -> pd.DataFrame:
    """Keep payments of kept orders, canonical method names, parsed dates."""
    df = drop_orphans(raw, "order_id", orders["order_id"])
    log.record("payments", "drop payments of removed orders", len(raw), len(df), len(raw) - len(df))
    df = standardize_column(df, "payment_method", list(cfg["payments"]["methods"]), cfg, log, "payments")
    df = parse_date_column(df, "payment_date", cfg, log, "payments")

    status = df["order_id"].map(orders.set_index("order_id")["order_status"])
    inconsistent = int(((status == "cancelled") != (df["payment_status"] == "failed")).sum())
    log.record("payments", "check failed payment <-> cancelled order", len(df), len(df), inconsistent)
    return df.reset_index(drop=True)


def clean_returns(raw: pd.DataFrame, items: pd.DataFrame, orders: pd.DataFrame, cfg: dict, log: CleaningLog) -> pd.DataFrame:
    """Keep returns of kept order lines, parsed dates, and check none precede their order."""
    df = drop_orphans(raw, "order_item_id", items["order_item_id"])
    log.record("returns", "drop returns of removed lines", len(raw), len(df), len(raw) - len(df))
    df = parse_date_column(df, "return_date", cfg, log, "returns")

    order_id = df["order_item_id"].map(items.set_index("order_item_id")["order_id"])
    order_date = order_id.map(orders.set_index("order_id")["order_date"])
    log.record("returns", "check return_date >= order_date", len(df), len(df), int((df["return_date"] < order_date).sum()))
    return df.reset_index(drop=True)


def clean_all(raw: dict[str, pd.DataFrame], cfg: dict) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Run the full cleaning pipeline; return cleaned tables and the cleaning log."""
    log = CleaningLog()
    customers = clean_customers(raw["customers"], cfg, log)
    categories = raw["categories"]
    products = clean_products(raw["products"], categories, cfg, log)
    orders, items = clean_orders_and_items(raw["orders"], raw["order_items"], customers, products, cfg, log)
    payments = clean_payments(raw["payments"], orders, cfg, log)
    returns = clean_returns(raw["returns"], items, orders, cfg, log)
    tables = {
        "customers": customers, "categories": categories, "products": products, "orders": orders,
        "order_items": items, "payments": payments, "returns": returns,
    }
    return tables, log.to_frame()


def main() -> None:
    """Clean the raw tables and write them, plus the cleaning log, to the processed directory."""
    setup_logging()
    cfg = load_config()
    raw = load_tables(project_path(cfg["paths"]["raw_dir"]))
    tables, cleaning_log = clean_all(raw, cfg)

    out_dir = project_path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")
    cleaning_log.to_csv(out_dir / "cleaning_log.csv", index=False)

    raw_revenue = cleaning_log["revenue_before"].dropna().iloc[0]
    clean_revenue = cleaning_log["revenue_after"].dropna().iloc[-1]
    logger.info("Net revenue raw %.2f -> clean %.2f BDT (%+.2f)", raw_revenue, clean_revenue, clean_revenue - raw_revenue)
    logger.info("Wrote %d cleaned tables to %s", len(tables), cfg["paths"]["processed_dir"])


if __name__ == "__main__":
    main()
