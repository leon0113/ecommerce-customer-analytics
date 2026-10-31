"""Inject deliberate data quality issues into raw tables and keep a ground-truth log of each one.

Methods change the frame in place (except `duplicates`, which returns a new frame) and
return the row labels they touched, so callers can keep issues from overlapping.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class IssueInjector:
    """Applies the messiness rates from config.yaml and records every change it makes."""

    def __init__(self, cfg: dict, date_formats: dict, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.formats = date_formats
        self.rng = rng
        self._log: list[pd.DataFrame] = []

    def issue_log(self) -> pd.DataFrame:
        """All injected issues as one table: which row, which column, what type, original value."""
        return pd.concat(self._log, ignore_index=True)

    def _pick(
        self,
        df: pd.DataFrame,
        rate: float,
        column: str | None = None,
        eligible: np.ndarray | None = None,
        weights: pd.Series | None = None,
    ) -> pd.Index:
        """Choose round(rate x candidate rows) labels; weights make some rows likelier than others."""
        mask = np.ones(len(df), dtype=bool)
        if column is not None:
            mask &= df[column].notna().to_numpy()
        if eligible is not None:
            mask &= eligible
        candidates = df.index[mask]
        size = round(rate * len(candidates))
        if size == 0:
            return candidates[:0]
        probs = None
        if weights is not None:
            w = weights.loc[candidates].to_numpy(dtype=float)
            probs = w / w.sum()
        return pd.Index(self.rng.choice(candidates, size=size, replace=False, p=probs))

    def _record(
        self, df: pd.DataFrame, rows: pd.Index, table: str, key: str, column: str, issue_type: str
    ) -> None:
        """Log the rows about to be changed, with their current (original) values."""
        self._log.append(pd.DataFrame({
            "table": table,
            "row_key": df.loc[rows, key].to_numpy(),
            "column": column,
            "issue_type": issue_type,
            "original_value": df.loc[rows, column].astype(str).to_numpy(),
        }))

    def missing(
        self, df: pd.DataFrame, table: str, key: str, column: str, weights: pd.Series | None = None
    ) -> pd.Index:
        """Blank out values, more often in rows with higher weight."""
        rows = self._pick(df, self.cfg["missing_rate"], column, weights=weights)
        self._record(df, rows, table, key, column, "missing_value")
        df.loc[rows, column] = np.nan
        return rows

    def text_noise(
        self, df: pd.DataFrame, table: str, key: str, column: str, eligible: np.ndarray | None = None
    ) -> pd.Index:
        """Upper-case, lower-case, or pad values with stray whitespace."""
        rows = self._pick(df, self.cfg["text_noise_rate"], column, eligible)
        self._record(df, rows, table, key, column, "text_noise")
        original = df.loc[rows, column].astype(str)
        style = self.rng.integers(0, 3, size=len(rows))
        df.loc[rows, column] = np.select(
            [style == 0, style == 1], [original.str.upper(), original.str.lower()], "  " + original + " "
        )
        return rows

    def typos(
        self, df: pd.DataFrame, table: str, key: str, column: str, eligible: np.ndarray | None = None
    ) -> pd.Index:
        """Use a known variant spelling (Chittagong) where one exists, else drop a vowel (Gazpur)."""
        rows = self._pick(df, self.cfg["typo_rate"], column, eligible)
        self._record(df, rows, table, key, column, "typo")
        original = df.loc[rows, column].astype(str)
        dropped_vowel = original.str.replace(r"[aeiou]", "", n=1, regex=True)
        df.loc[rows, column] = original.map(self.cfg["name_variants"]).fillna(dropped_vowel).to_numpy()
        return rows

    def date_formats(
        self, df: pd.DataFrame, table: str, key: str, column: str, eligible: np.ndarray | None = None
    ) -> pd.Index:
        """Rewrite canonical date strings in one of the alternative formats."""
        formats = self.formats
        rows = self._pick(df, self.cfg["mixed_date_format_rate"], column, eligible)
        self._record(df, rows, table, key, column, "mixed_date_format")
        parsed = pd.to_datetime(df.loc[rows, column], format=formats["canonical"])
        choice = self.rng.integers(0, len(formats["variants"]), size=len(rows))
        rewritten = df.loc[rows, column].copy()
        for i, fmt in enumerate(formats["variants"]):
            rewritten[choice == i] = parsed[choice == i].dt.strftime(fmt)
        df.loc[rows, column] = rewritten.to_numpy()
        return rows

    def impossible_dates(self, df: pd.DataFrame, table: str, key: str, column: str) -> pd.Index:
        """Half become invalid calendar dates (30 February), half move years into the future."""
        canonical = self.formats["canonical"]
        rows = self._pick(df, self.cfg["impossible_date_rate"], column)
        self._record(df, rows, table, key, column, "impossible_date")
        parsed = pd.to_datetime(df.loc[rows, column], format=canonical)
        invalid = parsed.dt.strftime("%Y-02-30 %H:%M:%S")
        future = (parsed + pd.DateOffset(years=self.cfg["impossible_year_shift"])).dt.strftime(canonical)
        df.loc[rows, column] = np.where(self.rng.random(len(rows)) < 0.5, invalid, future)
        return rows

    def negative_quantities(self, df: pd.DataFrame, table: str, key: str, column: str) -> pd.Index:
        """Flip the sign of quantities."""
        rows = self._pick(df, self.cfg["negative_quantity_rate"], column)
        self._record(df, rows, table, key, column, "negative_quantity")
        df.loc[rows, column] = -df.loc[rows, column]
        return rows

    def price_outliers(self, df: pd.DataFrame, table: str, key: str, column: str) -> pd.Index:
        """Multiply prices as if the decimal point was lost at data entry."""
        rows = self._pick(df, self.cfg["price_outlier_rate"], column)
        self._record(df, rows, table, key, column, "price_outlier")
        df.loc[rows, column] = df.loc[rows, column] * self.cfg["price_outlier_multiplier"]
        return rows

    def orphan_keys(
        self, df: pd.DataFrame, table: str, key: str, column: str, max_valid_id: int
    ) -> pd.Index:
        """Point foreign keys at ids that do not exist in the parent table."""
        rows = self._pick(df, self.cfg["orphan_fk_rate"], column)
        self._record(df, rows, table, key, column, "orphan_fk")
        df.loc[rows, column] = max_valid_id + self.rng.integers(1, 1000, size=len(rows))
        return rows

    def duplicates(self, df: pd.DataFrame, table: str, key: str) -> pd.DataFrame:
        """Return the frame with exact copies of some rows placed next to their originals."""
        rows = self._pick(df, self.cfg["duplicate_row_rate"])
        self._record(df, rows, table, key, key, "duplicate_row")
        return pd.concat([df, df.loc[rows]]).sort_index(kind="stable").reset_index(drop=True)
