"""Utility helpers for saving outputs and reproducibility."""

from pathlib import Path
from typing import Dict
import json
import random

import joblib
import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def save_model(model, path: Path) -> None:
    """Persist a trained model to disk."""
    ensure_dir(path.parent)
    joblib.dump(model, path)


def save_metrics(metrics: Dict[str, dict], path: Path) -> None:
    """Save metrics as a CSV file."""
    ensure_dir(path.parent)
    pd.DataFrame(metrics).T.to_csv(path, index=True)


def save_metrics_json(metrics: Dict[str, dict], path: Path) -> None:
    """Save metrics as a JSON file."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
