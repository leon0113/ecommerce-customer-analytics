"""Reusable analysis functions: named SQL queries, Pareto share, RFM, churn and cohort retention.

Notebooks and the Excel export call these so every number is computed one way only.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import project_path

QUERY_PATH = project_path("sql/analysis_queries.sql")

# RFM segments, checked in order: the first matching rule names the customer.
SEGMENT_RULES: list[tuple[str, str]] = [
    ("Champions", "r >= 4 and f >= 4"),
    ("Loyal", "r >= 3 and f >= 3"),
    ("New", "r >= 4 and f == 1"),
    ("Promising", "r >= 4"),
    ("At Risk", "r <= 2 and f >= 3"),
    ("Hibernating", "r <= 2"),
    ("Needs Attention", "r == 3"),
]
SEGMENT_ORDER = [name for name, _ in SEGMENT_RULES]


def load_queries(path: Path = QUERY_PATH) -> dict[str, str]:
    """Split the SQL file into queries keyed by their '-- name: <name>' marker."""
    queries: dict[str, list[str]] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("-- name:"):
            current = line.split(":", 1)[1].strip()
            queries[current] = []
        elif current is not None:
            queries[current].append(line)
    return {name: "\n".join(lines).strip() for name, lines in queries.items()}


def open_db(cfg: dict) -> sqlite3.Connection:
    """Connect to the analysis database built by src.load_db."""
    return sqlite3.connect(project_path(cfg["paths"]["db_path"]))


def run_query(conn: sqlite3.Connection, name: str) -> pd.DataFrame:
    """Run a named query; columns ending in '_date' come back as datetimes."""
    df = pd.read_sql_query(load_queries()[name], conn)
    for column in df.columns[df.columns.str.endswith("_date")]:
        df[column] = pd.to_datetime(df[column])
    return df


def top_share(values: pd.Series, fraction: float) -> float:
    """Share of the total held by the largest `fraction` of values (e.g. top 20% of customers)."""
    ordered = values.sort_values(ascending=False)
    top_n = int(np.ceil(fraction * len(ordered)))
    return float(ordered.iloc[:top_n].sum() / ordered.sum())


def pareto_curve(values: pd.Series) -> pd.DataFrame:
    """Cumulative share of customers (largest first) against cumulative share of revenue."""
    ordered = values.sort_values(ascending=False).to_numpy()
    return pd.DataFrame({
        "customer_share": np.arange(1, len(ordered) + 1) / len(ordered),
        "revenue_share": ordered.cumsum() / ordered.sum(),
    })


def churn_threshold_days(gaps: pd.Series, quantile: float) -> int:
    """Days without an order after which a customer counts as churned, taken from repeat-order gaps.

    Using a high quantile of real gaps means only an unusually long silence is called churn.
    """
    return int(np.ceil(gaps.dropna().quantile(quantile)))


def label_churn(customers: pd.DataFrame, snapshot: pd.Timestamp, threshold_days: int) -> pd.DataFrame:
    """Add recency, tenure and churn status: churned, active, or too_new to judge.

    Customers whose first order is more recent than the threshold cannot have churned
    yet, so they are kept out of churn rates instead of counting as active.
    """
    out = customers.copy()
    out["recency_days"] = (snapshot - out["last_order_date"]).dt.days
    out["tenure_days"] = (snapshot - out["first_order_date"]).dt.days
    out["churn_status"] = np.select(
        [out["tenure_days"] <= threshold_days, out["recency_days"] > threshold_days],
        ["too_new", "churned"],
        "active",
    )
    out["churned"] = out["churn_status"] == "churned"
    return out


def churn_rate_by(customers: pd.DataFrame, column: str) -> pd.DataFrame:
    """Churn rate per group among judgeable customers, with lift against the overall rate."""
    judged = customers[customers["churn_status"] != "too_new"]
    overall = judged["churned"].mean()
    table = judged.groupby(column, observed=True)["churned"].agg(customers="size", churn_rate="mean")
    table["lift"] = table["churn_rate"] / overall
    return table.sort_values("churn_rate", ascending=False)


def add_bins(values: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    """Bucket a numeric column into labelled, ordered bins."""
    return pd.cut(values, bins=edges, labels=labels)


def flag_at_risk(customers: pd.DataFrame, gaps: pd.Series, threshold_days: int) -> pd.Series:
    """Active customers silent for longer than a typical (median) repeat gap but not yet churned."""
    median_gap = gaps.dropna().median()
    return (customers["churn_status"] == "active") & customers["recency_days"].between(median_gap, threshold_days)


def score_rfm(customers: pd.DataFrame, frequency_edges: list[float]) -> pd.DataFrame:
    """Add 1-5 recency, frequency and monetary scores and a segment name.

    Recency and monetary use quintiles. Frequency uses fixed order-count bins, because most
    customers have one or two orders and quintiles of tied values are meaningless.
    """
    out = customers.copy()
    out["r"] = pd.qcut(out["recency_days"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    out["f"] = pd.cut(out["orders"], bins=frequency_edges, labels=[1, 2, 3, 4, 5]).astype(int)
    out["m"] = pd.qcut(out["revenue"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    out["rfm_segment"] = assign_segment(out[["r", "f"]])
    return out


def assign_segment(scores: pd.DataFrame) -> pd.Categorical:
    """Name each customer by the first rule in SEGMENT_RULES their R and F scores satisfy."""
    conditions = [scores.eval(rule).to_numpy() for _, rule in SEGMENT_RULES]
    names = np.select(conditions, SEGMENT_ORDER, default="Needs Attention")
    return pd.Categorical(names, categories=SEGMENT_ORDER, ordered=True)


def segment_profile(customers: pd.DataFrame) -> pd.DataFrame:
    """Size, revenue share and average behaviour of each RFM segment."""
    profile = customers.groupby("rfm_segment", observed=True).agg(
        customers=("customer_id", "size"),
        revenue=("revenue", "sum"),
        avg_recency_days=("recency_days", "mean"),
        avg_orders=("orders", "mean"),
        avg_revenue=("revenue", "mean"),
        avg_discount_share=("discount_share", "mean"),
        avg_return_rate=("return_rate", "mean"),
    )
    profile.insert(1, "customer_share", profile["customers"] / profile["customers"].sum())
    profile.insert(3, "revenue_share", profile["revenue"] / profile["revenue"].sum())
    return profile


def build_customer_table(conn: sqlite3.Connection, cfg: dict) -> tuple[pd.DataFrame, dict[str, float]]:
    """One row per buyer with churn status, RFM scores, driver bands and at-risk flag.

    Also returns the churn rule used (threshold and median repeat gap) so reports can quote it.
    """
    settings = cfg["analysis"]
    gaps = run_query(conn, "order_revenue")["days_since_previous_order"]
    threshold = churn_threshold_days(gaps, settings["churn_gap_quantile"])
    customers = label_churn(run_query(conn, "customer_summary"), pd.Timestamp(settings["snapshot_date"]), threshold)
    customers = score_rfm(customers, settings["rfm_frequency_edges"])
    customers["buyer_type"] = np.where(customers["orders"] > 1, "repeat", "one order")
    customers["discount_band"] = add_bins(
        customers["discount_share"], settings["discount_share_bins"], settings["discount_share_labels"]
    )
    customers["return_band"] = add_bins(
        customers["return_rate"], settings["return_rate_bins"], settings["return_rate_labels"]
    )
    customers["at_risk"] = flag_at_risk(customers, gaps, threshold)
    rule = {
        "threshold_days": threshold,
        "median_gap_days": float(gaps.median()),
        "repeat_gaps": int(gaps.notna().sum()),
    }
    return customers, rule


def last_quarter_index(activity: pd.DataFrame) -> int:
    """Index of the latest quarter with any activity in the data."""
    return int((activity["cohort_index"] + activity["quarters_since_first"]).max())


def cohort_age(activity: pd.DataFrame) -> pd.Series:
    """Quarters each customer's cohort has had since first order, up to the last quarter in the data."""
    first_rows = activity.drop_duplicates("customer_id").set_index("customer_id")
    return last_quarter_index(activity) - first_rows["cohort_index"]


def retention_matrix(activity: pd.DataFrame, max_quarters: int) -> pd.DataFrame:
    """Share of each first-order cohort buying again N quarters later (quarter 0 = 100%).

    Quarters a cohort has not reached yet are left empty rather than shown as zero.
    """
    offsets = range(max_quarters + 1)
    counts = (
        activity.groupby(["cohort_quarter", "quarters_since_first"])["customer_id"].nunique()
        .unstack(fill_value=0)
        .reindex(columns=offsets, fill_value=0)
    )
    matrix = counts.div(counts[0], axis=0)
    ages = last_quarter_index(activity) - activity.groupby("cohort_quarter")["cohort_index"].first()
    reached = np.asarray(offsets)[None, :] <= ages.reindex(matrix.index).to_numpy()[:, None]
    return matrix.where(reached)


def retention_by_group(activity: pd.DataFrame, group: str, max_quarters: int) -> pd.DataFrame:
    """Retention curve per group: share of customers buying N quarters after their first order.

    Each quarter only counts customers whose cohort is old enough to have reached it.
    """
    customers = activity.drop_duplicates("customer_id").set_index("customer_id")
    age = cohort_age(activity)
    bought = pd.crosstab(activity["customer_id"], activity["quarters_since_first"]) > 0
    curves = {}
    for offset in range(1, max_quarters + 1):
        eligible = age.index[age >= offset]
        bought_then = bought.get(offset, pd.Series(False, index=bought.index)).reindex(eligible, fill_value=False)
        curves[offset] = bought_then.groupby(customers.loc[eligible, group]).mean()
    return pd.DataFrame(curves)
