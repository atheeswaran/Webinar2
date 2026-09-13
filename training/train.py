
"""Train the fraud classifier.

Logs parameters + metrics to MLflow and writes:
  models/model.pkl        - trained pipeline (DVC-tracked)
  models/metrics.json     - metrics consumed by DVC (`dvc metrics show`)
"""
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
N_TREES = 200
MIN_RECALL = 0.60   # quality gate: fraud must not slip through

FEATURES = ["amount", "hour", "age", "merchant_risk", "distance_from_home"]
LABEL = "is_fraud"


def build_pipeline(n_trees: int = N_TREES) -> Pipeline:
    preprocessor = ColumnTransformer(
        [("scale", StandardScaler(), FEATURES)],
        remainder="drop",
    )
    return Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=n_trees, max_depth=8, random_state=SEED, n_jobs=-1)),
    ])


def train(data_path: str = "data/fraud_data.csv") -> dict:
    df = pd.read_csv(data_path)
    X = df[FEATURES]
    y = df[LABEL]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y)

    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.3).astype(int)

    metrics = {
        "accuracy": round(accuracy_score(y_te, pred), 4),
        "precision": round(precision_score(y_te, pred, zero_division=0), 4),
        "recall": round(recall_score(y_te, pred, zero_division=0), 4),
        "f1": round(f1_score(y_te, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_te, proba), 4),
    }
    if metrics["recall"] < MIN_RECALL:
        raise SystemExit(
            f"QUALITY GATE FAILED: recall {metrics['recall']} < {MIN_RECALL}")

    Path("models").mkdir(exist_ok=True)
    joblib.dump(pipe, "models/model.pkl")
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    mlflow.set_experiment("fraud-detection")
    with mlflow.start_run(run_name="ci-train"):
        mlflow.log_params({"n_trees": N_TREES, "seed": SEED,
                           "n_rows": len(df), "data_md5": _md5(data_path)})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipe, "fraud_model",
                                 input_example=X_te.head(5))
        print("MLflow run id:", mlflow.active_run().info.run_id)
    return metrics


def _md5(path: str) -> str:
    import hashlib
    return hashlib.md5(open(path, "rb").read()).hexdigest()


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
