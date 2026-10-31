"""Load the cleaned tables into SQLite using sql/schema.sql.

Tables load parent-first with foreign keys enforced, so any broken reference fails the
load instead of slipping into the database. Run with: python -m src.load_db
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from src.common import TABLE_NAMES, load_config, load_tables, project_path, setup_logging

logger = logging.getLogger(__name__)

SCHEMA_PATH = project_path("sql/schema.sql")


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection with foreign key enforcement on (SQLite leaves it off by default)."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_schema(conn: sqlite3.Connection, schema_path: Path = SCHEMA_PATH) -> None:
    """Drop and recreate every table from the schema file."""
    conn.executescript(schema_path.read_text(encoding="utf-8"))


def insert_tables(conn: sqlite3.Connection, tables: dict[str, pd.DataFrame]) -> None:
    """Append each table into its pre-created SQL table, parents before children."""
    for name in TABLE_NAMES:
        tables[name].to_sql(name, conn, if_exists="append", index=False)
    conn.commit()


def table_row_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Row count of every table in the database."""
    return {name: conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0] for name in TABLE_NAMES}


def build_database(tables: dict[str, pd.DataFrame], db_path: Path) -> dict[str, int]:
    """Create the database from cleaned tables; raise if counts or foreign keys do not check out."""
    with connect(db_path) as conn:
        create_schema(conn)
        insert_tables(conn, tables)
        counts = table_row_counts(conn)
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    conn.close()

    mismatched = {name: (len(tables[name]), counts[name]) for name in TABLE_NAMES if len(tables[name]) != counts[name]}
    if mismatched:
        raise ValueError(f"Row counts differ between cleaned tables and database: {mismatched}")
    if violations:
        raise ValueError(f"Foreign key violations: {violations[:5]}")
    return counts


def main() -> None:
    """Load the processed CSVs into the configured SQLite database."""
    setup_logging()
    cfg = load_config()
    tables = load_tables(project_path(cfg["paths"]["processed_dir"]))
    db_path = project_path(cfg["paths"]["db_path"])
    counts = build_database(tables, db_path)
    for name, count in counts.items():
        logger.info("%-12s %6d rows", name, count)
    logger.info("Database written to %s", cfg["paths"]["db_path"])


if __name__ == "__main__":
    main()
