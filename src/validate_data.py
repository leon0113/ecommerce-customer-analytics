"""Sanity report on the raw generated data: volumes, trends, Pareto share, returns and injected issues.

Rows with known structural issues (from the ground-truth log) are excluded and altered
category names are restored, so the report shows the business signal rather than the
noise. This is a generator check, not the cleaning pipeline.
Run with: python -m src.validate_data
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.common import known_date_formats, load_config, load_tables, parse_mixed_dates, project_path, setup_logging

logger = logging.getLogger(__name__)

ROW_LEVEL_ISSUES = ["impossible_date", "negative_quantity", "price_outlier", "orphan_fk"]
TABLE_KEYS = {"orders": "order_id", "order_items": "order_item_id"}


def exclude_known_issues(tables: dict[str, pd.DataFrame], issues: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Drop duplicate and structurally broken rows, and restore original category names."""
    tables = {name: df.copy() for name, df in tables.items()}
    tables["orders"] = tables["orders"].drop_duplicates()
    row_issues = issues[issues["issue_type"].isin(ROW_LEVEL_ISSUES)]
    for table, key in TABLE_KEYS.items():
        bad_keys = row_issues.loc[row_issues["table"] == table, "row_key"].astype("int64")
        tables[table] = tables[table][~tables[table][key].isin(bad_keys)]

    altered = issues[(issues["table"] == "products") & (issues["column"] == "category_name")]
    original = altered.set_index(altered["row_key"].astype("int64"))["original_value"]
    products = tables["products"]
    products["category_name"] = products["product_id"].map(original).fillna(products["category_name"])
    return tables


def build_delivered_lines(tables: dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    """One row per delivered order line with net revenue, order date, customer and category."""
    orders = tables["orders"].assign(
        order_date=parse_mixed_dates(tables["orders"]["order_date"], known_date_formats(cfg))
    )
    orders = orders[orders["order_status"] == "delivered"]
    lines = (
        tables["order_items"]
        .merge(orders[["order_id", "customer_id", "order_date", "discount_pct"]], on="order_id")
        .merge(tables["products"][["product_id", "category_name"]], on="product_id")
    )
    lines["revenue"] = lines["quantity"] * lines["unit_price"] * (1 - lines["discount_pct"])
    lines["returned"] = lines["order_item_id"].isin(tables["returns"]["order_item_id"])
    return lines


def report_volumes(raw: dict[str, pd.DataFrame], max_rows: int) -> None:
    """Raw row counts against the per-table cap."""
    counts = pd.Series({name: len(df) for name, df in raw.items()}, name="rows").to_frame()
    counts["within_cap"] = counts["rows"] <= max_rows
    logger.info("Raw row counts (cap %d):\n%s", max_rows, counts.to_string())


def report_revenue(lines: pd.DataFrame) -> None:
    """Revenue by year with YoY growth, and a month x year revenue grid."""
    by_year = lines.groupby(lines["order_date"].dt.year)["revenue"].sum().to_frame("revenue_bdt")
    by_year["yoy_growth_pct"] = by_year["revenue_bdt"].pct_change() * 100
    logger.info(
        "Order dates %s to %s", lines["order_date"].min().date(), lines["order_date"].max().date()
    )
    logger.info("Revenue by year (BDT):\n%s", by_year.round(1).to_string())
    grid = lines.pivot_table(
        index=lines["order_date"].dt.month, columns=lines["order_date"].dt.year,
        values="revenue", aggfunc="sum",
    )
    logger.info("Revenue by month (BDT, rows = month):\n%s", grid.round(0).to_string())


def report_pareto(lines: pd.DataFrame) -> None:
    """Share of revenue from the top 20% of purchasing customers."""
    per_customer = lines.groupby("customer_id")["revenue"].sum().sort_values(ascending=False)
    top_n = int(np.ceil(0.2 * len(per_customer)))
    share = per_customer.iloc[:top_n].sum() / per_customer.sum()
    logger.info("Top 20%% of %d buyers produce %.1f%% of revenue", len(per_customer), share * 100)


def report_timing(lines: pd.DataFrame) -> None:
    """Order share by weekday and the evening (18:00-23:59) share."""
    orders = lines.drop_duplicates("order_id")
    weekday = orders["order_date"].dt.day_name().value_counts(normalize=True).mul(100).round(1)
    evening = (orders["order_date"].dt.hour >= 18).mean() * 100
    logger.info("Order share by weekday (%%):\n%s", weekday.to_string())
    logger.info("Evening (18:00+) share of orders: %.1f%%", evening)


def report_returns(lines: pd.DataFrame, cfg: dict) -> None:
    """Observed line return rate per category next to the configured base rate."""
    observed = lines.groupby("category_name")["returned"].agg(lines_delivered="size", return_rate="mean")
    observed["configured_base_rate"] = pd.Series({k: v["return_rate"] for k, v in cfg["categories"].items()})
    observed["return_rate"] = observed["return_rate"].round(3)
    logger.info("Return rate by category:\n%s", observed.sort_values("return_rate", ascending=False).to_string())


def report_customer_types(lines: pd.DataFrame, truth: pd.DataFrame) -> None:
    """Orders, AOV and churn per hidden customer type, and churn per acquisition channel."""
    order_totals = lines.groupby(["order_id", "customer_id"])["revenue"].sum().reset_index()
    order_totals = order_totals.merge(truth[["customer_id", "customer_type"]], on="customer_id")
    by_type = order_totals.groupby("customer_type").agg(orders=("order_id", "size"), aov_bdt=("revenue", "mean"))
    by_type["revenue_share_pct"] = order_totals.groupby("customer_type")["revenue"].sum() / order_totals["revenue"].sum() * 100
    truth = truth.assign(churned=truth["churn_month"].notna())
    by_type["churned_pct"] = truth.groupby("customer_type")["churned"].mean() * 100
    logger.info("By hidden customer type:\n%s", by_type.round(1).to_string())
    repeat_buyers = truth[truth["customer_type"] != "one_time"]
    by_channel = repeat_buyers.groupby("acquisition_channel")["churned"].mean().mul(100).round(1)
    logger.info("Churned %% by channel (excluding one-time type):\n%s", by_channel.to_string())


def report_issues(issues: pd.DataFrame) -> None:
    """Count of injected issues per table and type."""
    counts = issues.groupby(["table", "issue_type"]).size().unstack(fill_value=0)
    logger.info("Injected issues:\n%s", counts.to_string())


def main() -> None:
    """Run every check and log the report."""
    setup_logging()
    cfg = load_config()
    raw = load_tables(project_path(cfg["paths"]["raw_dir"]))
    issues = pd.read_csv(project_path(cfg["paths"]["issue_log"]))
    truth = pd.read_csv(project_path(cfg["paths"]["customer_truth"]))
    lines = build_delivered_lines(exclude_known_issues(raw, issues), cfg)

    report_volumes(raw, cfg["volume"]["max_rows_per_table"])
    report_revenue(lines)
    report_pareto(lines)
    report_timing(lines)
    report_returns(lines, cfg)
    report_customer_types(lines, truth)
    report_issues(issues)


if __name__ == "__main__":
    main()
