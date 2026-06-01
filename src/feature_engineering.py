"""Feature engineering utilities."""

import pandas as pd


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-specific features. Currently a no-op placeholder."""
    return df.copy()
