"""Shared helpers: project paths, config loading, logging setup and date parsing."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_NAMES = ["customers", "categories", "products", "orders", "order_items", "payments", "returns"]


def project_path(relative: str) -> Path:
    """Resolve a config path against the project root so scripts work from any directory."""
    return PROJECT_ROOT / relative


def load_config(path: Path = PROJECT_ROOT / "config.yaml") -> dict:
    """Load the project configuration."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a single, readable log format for all scripts."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def load_tables(directory: Path, names: list[str] = TABLE_NAMES) -> dict[str, pd.DataFrame]:
    """Read one CSV per table from a directory; phone numbers stay text to keep the leading zero."""
    return {name: pd.read_csv(directory / f"{name}.csv", dtype={"phone": "string"}) for name in names}


def known_date_formats(cfg: dict) -> list[str]:
    """Canonical date format first, then the variants seen in source systems."""
    formats = cfg["date_formats"]
    return [formats["canonical"], *formats["variants"]]


def parse_mixed_dates(values: pd.Series, formats: list[str]) -> pd.Series:
    """Parse strings written in any of several known formats; anything else becomes NaT.

    Explicit formats are used instead of guessing, because guessing silently swaps
    day and month on values like 05/03/2024.
    """
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    for fmt in formats:
        parsed = parsed.fillna(pd.to_datetime(values, format=fmt, errors="coerce"))
    return parsed
