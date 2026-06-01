"""Exploratory data analysis helpers."""

import pandas as pd

from config.pipeline_config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN


def missing_values_summary(df: pd.DataFrame) -> pd.Series:
    """Return missing value counts by column."""
    return df.isna().sum().sort_values(ascending=False)


def target_distribution(df: pd.DataFrame) -> pd.Series:
    """Return target class balance as proportions."""
    return df[TARGET_COLUMN].value_counts(normalize=True)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive stats for numeric features."""
    return df[NUMERIC_FEATURES].describe().T


def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return value counts for categorical features."""
    summary_frames = []
    for col in CATEGORICAL_FEATURES:
        counts = df[col].value_counts(dropna=False).rename("count")
        summary_frames.append(counts.to_frame().assign(feature=col))
    return pd.concat(summary_frames)
