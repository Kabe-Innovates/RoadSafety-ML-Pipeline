"""Train the baseline models and save metrics."""

from pathlib import Path

from config.pipeline_config import MODELS_DIR, RANDOM_SEED, RESULTS_DIR
from src.data_loader import load_dataset
from src.feature_engineering import add_domain_features
from src.modeling import build_model_pipelines, evaluate_models, train_models
from src.preprocessing import build_preprocessor, split_features_target, train_test_split_data
from src.utils import save_metrics, save_metrics_json, save_model, set_random_seed


def main() -> None:
    set_random_seed(RANDOM_SEED)

    df = load_dataset()
    df = add_domain_features(df)

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    preprocessor = build_preprocessor()
    models = build_model_pipelines(preprocessor)

    trained_models = train_models(models, X_train, y_train)
    metrics = evaluate_models(trained_models, X_test, y_test)

    results_path = Path(RESULTS_DIR) / "model_metrics.csv"
    results_json_path = Path(RESULTS_DIR) / "model_metrics.json"
    save_metrics(metrics, results_path)
    save_metrics_json(metrics, results_json_path)

    for name, model in trained_models.items():
        save_model(model, Path(MODELS_DIR) / f"{name}.joblib")

    print("Training complete. Metrics saved to results/.")


if __name__ == "__main__":
    main()
