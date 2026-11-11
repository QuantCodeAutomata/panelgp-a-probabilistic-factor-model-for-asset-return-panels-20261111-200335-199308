"""Panel preprocessing for the empirical application (paper Section 6.1).

Implements the three input transforms applied before PanelGP/IPCA
estimation: cross-sectional characteristic ranking to [-0.5, 0.5] with
missing values set to zero, training-sample standardization of the
macro-financial context, and the return-scaling transform s_r^2 = mean(r^2)
over training rows (no centering).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rank_transform_characteristics(panel: pd.DataFrame, char_cols: list[str]) -> pd.DataFrame:
    """Cross-sectional rank transform to [-0.5, 0.5], missing set to zero.

    Within each month, observed values of a characteristic are ranked and
    mapped to [-0.5, 0.5] via (rank - 1) / (n_observed - 1) - 0.5; a
    characteristic observed for a single stock in a month maps to 0. Missing
    entries are set to zero after ranking (paper Sec. 6.1).

    Args:
        panel: long panel with a ``date`` column and one column per raw
            characteristic in ``char_cols``.
        char_cols: names of the raw characteristic columns to transform.

    Returns:
        A copy of ``panel`` with each characteristic column replaced by its
        ranked version.
    """
    out = panel.copy()

    def _rank_month(group: pd.DataFrame) -> pd.DataFrame:
        ranked = group[char_cols].rank(method="average")
        n_obs = group[char_cols].notna().sum()
        scaled = (ranked - 1) / (n_obs - 1).replace(0, np.nan) - 0.5
        return scaled.fillna(0.0)

    out[char_cols] = out.groupby("date", group_keys=False).apply(_rank_month)
    return out


def filter_characteristics_by_coverage(
    panel: pd.DataFrame, char_cols: list[str], train_mask: pd.Series,
    min_char_coverage: float,
) -> list[str]:
    """Keeps characteristics observed for at least ``min_char_coverage`` of
    training-period stock-months (paper Sec. 6.1: "137 characteristics
    observed for at least 70% of stock-months in the 1960-1989 training
    period")."""
    train = panel.loc[train_mask, char_cols]
    coverage = train.notna().mean(axis=0)
    return [c for c in char_cols if coverage[c] >= min_char_coverage]


def filter_rows_by_row_coverage(
    panel: pd.DataFrame, char_cols: list[str], min_row_coverage: float,
) -> pd.DataFrame:
    """Keeps stock-months with at least ``min_row_coverage`` of the kept
    characteristics observed (paper Sec. 6.1)."""
    coverage = panel[char_cols].notna().mean(axis=1)
    return panel.loc[coverage >= min_row_coverage].reset_index(drop=True)


def standardize_context(context: pd.DataFrame, train_mask: pd.Series) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Standardizes context columns to zero mean, unit variance using only
    training-sample moments (paper Sec. 6.1)."""
    mean = context.loc[train_mask].mean(axis=0)
    std = context.loc[train_mask].std(axis=0, ddof=0)
    std = std.replace(0, 1.0)
    standardized = (context - mean) / std
    return standardized, mean, std


def compute_return_scale(returns: np.ndarray, train_mask: np.ndarray) -> float:
    """Return scale s_r such that r_n^o = r_n / s_r, with
    s_r^2 = (1/N_train) * sum(r_n^2) over training rows only, no centering
    (paper Sec. 6.1)."""
    train_returns = np.asarray(returns)[np.asarray(train_mask)]
    s_r2 = float(np.mean(train_returns ** 2))
    assert s_r2 > 0, "degenerate training returns: s_r^2 must be positive"
    return float(np.sqrt(s_r2))
