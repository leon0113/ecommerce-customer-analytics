"""Shared fixtures: config, raw generated data, the cleaned result and the ground-truth issue log."""

from __future__ import annotations

import pandas as pd
import pytest

from src.clean import clean_all
from src.common import load_config, load_tables, project_path


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="session")
def raw(cfg: dict) -> dict[str, pd.DataFrame]:
    raw_dir = project_path(cfg["paths"]["raw_dir"])
    if not (raw_dir / "orders.csv").exists():
        pytest.skip("Raw data not generated; run python -m src.generate_data first")
    return load_tables(raw_dir)


@pytest.fixture(scope="session")
def issues(cfg: dict, raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.read_csv(project_path(cfg["paths"]["issue_log"]))


@pytest.fixture(scope="session")
def cleaned(raw: dict[str, pd.DataFrame], cfg: dict) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    return clean_all(raw, cfg)


@pytest.fixture(scope="session")
def tables(cleaned: tuple[dict[str, pd.DataFrame], pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return cleaned[0]


@pytest.fixture(scope="session")
def cleaning_log(cleaned: tuple[dict[str, pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
    return cleaned[1]


def issue_keys(issues: pd.DataFrame, table: str, issue_type: str, column: str | None = None) -> set[int]:
    """Row keys of one kind of injected issue."""
    mask = (issues["table"] == table) & (issues["issue_type"] == issue_type)
    if column is not None:
        mask &= issues["column"] == column
    return set(issues.loc[mask, "row_key"].astype("int64"))
