"""Evaluate a saved model on the default train-test split."""

import argparse
import json
from pathlib import Path

import joblib

from config.pipeline_config import MODELS_DIR, RANDOM_SEED
from src.data_loader import load_dataset
from src.feature_engineering import add_domain_features
from src.modeling import evaluate_models
from src.preprocessing import split_features_target, train_test_split_data
from src.utils import set_random_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved model.")
    parser.add_argument(
        "--model",
        default="random_forest",
        help="Model name to evaluate (e.g., random_forest)",
    )
    args = parser.parse_args()

    set_random_seed(RANDOM_SEED)

    df = load_dataset()
    df = add_domain_features(df)

    X, y = split_features_target(df)
    _, X_test, _, y_test = train_test_split_data(X, y)

    model_path = Path(MODELS_DIR) / f"{args.model}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    metrics = evaluate_models({args.model: model}, X_test, y_test)[args.model]

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
