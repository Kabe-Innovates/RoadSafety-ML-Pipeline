"""Data loading utilities."""

from pathlib import Path
from typing import Union

import pandas as pd

from config.pipeline_config import DATASET_PATH


def load_dataset(path: Union[str, Path] = DATASET_PATH) -> pd.DataFrame:
    """Load the raw dataset from a CSV file."""
    return pd.read_csv(path)
