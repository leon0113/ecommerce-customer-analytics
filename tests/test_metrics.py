"""Unit tests for analysis metrics on small hand-made frames with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import metrics


def test_load_queries_finds_every_named_query() -> None:
    queries = metrics.load_queries()
    assert {"kpi_summary", "monthly_revenue", "customer_summary", "customer_quarter_activity"} <= set(queries)
    assert all(query.strip() for query in queries.values())


def test_top_share_and_pareto_curve() -> None:
    revenue = pd.Series([50.0, 20.0, 10.0, 10.0, 10.0])
    assert metrics.top_share(revenue, 0.2) == pytest.approx(0.5)
    curve = metrics.pareto_curve(revenue)
    assert curve["revenue_share"].iloc[-1] == pytest.approx(1.0)
    assert curve["customer_share"].iloc[0] == pytest.approx(0.2)


def test_churn_threshold_is_the_gap_quantile_rounded_up() -> None:
    gaps = pd.Series([np.nan, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    assert metrics.churn_threshold_days(gaps, 0.9) == 91


def test_label_churn_separates_churned_active_and_too_new() -> None:
    snapshot = pd.Timestamp("2026-01-01")
    customers = pd.DataFrame({
        "first_order_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2025-12-01"]),
        "last_order_date": pd.to_datetime(["2024-06-01", "2025-12-15", "2025-12-01"]),
    })
    labelled = metrics.label_churn(customers, snapshot, threshold_days=140)
    assert labelled["churn_status"].tolist() == ["churned", "active", "too_new"]


def test_churn_rate_by_ignores_too_new_customers() -> None:
    customers = pd.DataFrame({
        "group": ["a", "a", "b", "b", "b"],
        "churn_status": ["churned", "active", "churned", "churned", "too_new"],
        "churned": [True, False, True, True, False],
    })
    table = metrics.churn_rate_by(customers, "group")
    assert table.loc["a", "churn_rate"] == pytest.approx(0.5)
    assert table.loc["b", "churn_rate"] == pytest.approx(1.0)
    assert table.loc["b", "customers"] == 2


def test_score_rfm_uses_fixed_frequency_bins_and_names_segments() -> None:
    customers = pd.DataFrame({
        "recency_days": [5, 10, 15, 20, 400, 500, 600, 700, 800, 900],
        "orders": [8, 4, 1, 2, 1, 6, 3, 1, 2, 1],
        "revenue": [900.0, 500, 100, 200, 50, 800, 300, 60, 150, 40],
    })
    scored = metrics.score_rfm(customers, [0, 1, 2, 3, 5, np.inf])
    assert scored["f"].tolist() == [5, 4, 1, 2, 1, 5, 3, 1, 2, 1]
    assert scored["r"].iloc[0] == 5 and scored["r"].iloc[-1] == 1
    assert scored["rfm_segment"].iloc[0] == "Champions"
    assert scored["rfm_segment"].iloc[2] == "New"
    assert scored["rfm_segment"].iloc[5] == "Loyal"
    assert scored["rfm_segment"].iloc[6] == "At Risk"
    assert scored["rfm_segment"].iloc[-1] == "Hibernating"


def test_retention_matrix_blanks_unreached_quarters_and_keeps_zero_activity() -> None:
    activity = pd.DataFrame({
        "customer_id": [1, 1, 2, 3, 3],
        "cohort_index": [8092, 8092, 8092, 8093, 8093],
        "cohort_quarter": ["2023-Q1", "2023-Q1", "2023-Q1", "2023-Q2", "2023-Q2"],
        "quarters_since_first": [0, 2, 0, 0, 1],
    })
    matrix = metrics.retention_matrix(activity, max_quarters=2)
    assert matrix.loc["2023-Q1"].tolist() == [1.0, 0.0, 0.5]
    assert matrix.loc["2023-Q2", 1] == 1.0
    assert np.isnan(matrix.loc["2023-Q2", 2])


def test_retention_by_group_counts_only_cohorts_old_enough() -> None:
    activity = pd.DataFrame({
        "customer_id": [1, 1, 2, 3],
        "acquisition_channel": ["organic", "organic", "organic", "social"],
        "cohort_index": [8092, 8092, 8092, 8093],
        "cohort_quarter": ["2023-Q1", "2023-Q1", "2023-Q1", "2023-Q2"],
        "quarters_since_first": [0, 1, 0, 0],
    })
    curves = metrics.retention_by_group(activity, "acquisition_channel", max_quarters=1)
    assert curves.loc["organic", 1] == pytest.approx(0.5)
    assert "social" not in curves.index
