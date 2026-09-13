
"""Evaluate a trained model and emit threshold curve data for DVC plots."""
import json
import sys

import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score

FEATURES = ["amount", "hour", "age", "merchant_risk", "distance_from_home"]


def main(model_path: str, data_path: str) -> None:
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    proba = model.predict_proba(df[FEATURES])[:, 1]
    rows = []
    for t in [x / 20 for x in range(1, 20)]:
        pred = (proba >= t).astype(int)
        rows.append({
            "threshold": round(t, 2),
            "precision": round(precision_score(df["is_fraud"], pred, zero_division=0), 4),
            "recall": round(recall_score(df["is_fraud"], pred, zero_division=0), 4),
        })
    with open("models/precision_recall.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote models/precision_recall.json ({len(rows)} thresholds)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "models/model.pkl",
         sys.argv[2] if len(sys.argv) > 2 else "data/fraud_data.csv")
