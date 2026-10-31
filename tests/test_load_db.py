"""Checks that the cleaned tables load into SQLite intact and that the schema rejects bad rows."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.load_db import build_database, connect, create_schema


def test_database_row_counts_match_cleaned_tables(tables: dict, tmp_path: Path) -> None:
    counts = build_database(tables, tmp_path / "test.db")
    assert counts == {name: len(table) for name, table in tables.items()}


def test_database_revenue_matches_pandas(tables: dict, tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    build_database(tables, db_path)
    query = """
        SELECT SUM(oi.quantity * oi.unit_price * (1 - o.discount_pct))
        FROM order_items AS oi
        JOIN orders AS o ON o.order_id = oi.order_id
        WHERE o.order_status = 'delivered'
    """
    with sqlite3.connect(db_path) as conn:
        sql_revenue = conn.execute(query).fetchone()[0]
    lines = tables["order_items"].merge(tables["orders"], on="order_id")
    lines = lines[lines["order_status"] == "delivered"]
    pandas_revenue = (lines["quantity"] * lines["unit_price"] * (1 - lines["discount_pct"])).sum()
    assert sql_revenue == pytest.approx(pandas_revenue, abs=0.01)


def test_schema_rejects_orphan_foreign_key(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    create_schema(conn)
    orphan_order = pd.DataFrame({
        "order_id": [1], "customer_id": [999], "order_date": ["2024-01-01 10:00:00"],
        "order_status": ["delivered"], "discount_pct": [0.0], "delivery_district": ["Dhaka"],
    })
    with pytest.raises(sqlite3.IntegrityError):
        orphan_order.to_sql("orders", conn, if_exists="append", index=False)
    conn.close()
