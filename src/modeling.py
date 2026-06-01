"""Modeling utilities for training and evaluation."""

from typing import Dict

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def build_model_pipelines(preprocessor) -> Dict[str, Pipeline]:
    """Build model pipelines with a shared preprocessor."""
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LogisticRegression(max_iter=1000)),
            ]
        ),
        "decision_tree": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", DecisionTreeClassifier(random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", RandomForestClassifier(n_estimators=200, random_state=42)),
            ]
        ),
        "svm": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", SVC(probability=True, random_state=42)),
            ]
        ),
    }


def train_models(models: Dict[str, Pipeline], X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Pipeline]:
    """Fit all models on the training data."""
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def evaluate_models(models: Dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, dict]:
    """Evaluate models and return a metrics dictionary."""
    metrics = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        metrics[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
        }
    return metrics
