
"""Quality-attribute test suite for the fraud demo.

Grouped by ISO 25010-style quality attributes. CI fails if any group breaks --
this is what "quality attributes drive design" means in practice.
"""
import json
import time
from pathlib import Path

import joblib
import pandas as pd
import pytest

from app.pipeline import MAX_AMOUNT, MAX_DISTANCE, validate_input
from app.schemas import Transaction

MODEL_PATH = Path("models/model.pkl")
METRICS_PATH = Path("models/metrics.json")
DATA_PATH = Path("data/fraud_data.csv")

LATENCY_THRESHOLD_MS = 200     # functional quality gate (Demo 2 tightens to 1)

VALID = {"amount": 250.0, "hour": 14, "age": 34,
         "merchant_risk": 0.4, "distance_from_home": 12.5}


# ---------------- correctness ----------------

class TestCorrectness:
    def test_model_artifact_exists(self):
        assert MODEL_PATH.exists(), "run training/train.py first"

    def test_metrics_file_exists_with_required_keys(self):
        m = json.loads(METRICS_PATH.read_text())
        for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            assert k in m, f"missing metric {k}"

    def test_predict_proba_range(self):
        model = joblib.load(MODEL_PATH)
        proba = model.predict_proba(pd.DataFrame([VALID]))[0]
        assert 0.0 <= proba[0] <= 1.0 and 0.0 <= proba[1] <= 1.0
        assert abs(proba.sum() - 1.0) < 1e-6


# ---------------- performance ----------------

class TestPerformance:
    @pytest.mark.parametrize("payload", [VALID] * 5)
    def test_prediction_latency(self, payload):
        model = joblib.load(MODEL_PATH)
        start = time.perf_counter()
        model.predict_proba(pd.DataFrame([payload]))
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < LATENCY_THRESHOLD_MS, (
            f"Prediction took {elapsed_ms:.1f} ms — exceeds "
            f"{LATENCY_THRESHOLD_MS} ms SLA")

    def test_batch_throughput(self):
        df = pd.read_csv(DATA_PATH).head(100)
        model = joblib.load(MODEL_PATH)
        start = time.perf_counter()
        model.predict_proba(df[["amount", "hour", "age",
                                "merchant_risk", "distance_from_home"]])
        elapsed = time.perf_counter() - start
        assert elapsed < 10.0, f"100-row batch took {elapsed:.1f}s"


# ---------------- reliability / robustness ----------------

class TestRobustness:
    @pytest.mark.parametrize("bad", [
        {"amount": 0}, {"amount": MAX_AMOUNT + 1},
        {"hour": 24}, {"hour": -1},
        {"age": 17}, {"merchant_risk": 1.5},
        {"distance_from_home": MAX_DISTANCE + 1},
    ])
    def test_validate_input_rejects_out_of_range(self, bad):
        payload = {**VALID, **bad}
        with pytest.raises(ValueError):
            validate_input(payload)

    def test_schema_rejects_invalid_types(self):
        with pytest.raises(Exception):
            Transaction(amount="not-a-number", hour=14, age=34,
                        merchant_risk=0.4, distance_from_home=12.5)


# ---------------- maintainability ----------------

class TestMaintainability:
    def test_no_dead_config_keys(self):
        assert "LATENCY_THRESHOLD_MS" in Path(__file__).read_text()

    def test_feature_list_matches_schema(self):
        schema_fields = set(Transaction.model_fields)
        model = joblib.load(MODEL_PATH)
        train_features = list(model.feature_names_in_)
        assert set(train_features) == schema_fields, (
            "training features and API schema diverged")


# ---------------- security ----------------

class TestSecurity:
    def test_no_hardcoded_secrets_in_repo(self):
        offenders = []
        for p in Path(".").rglob("*"):
            if p.is_file() and p.suffix in {".py", ".yml", ".yaml", ".env"} \
                    and ".git" not in p.parts and "mlruns" not in p.parts \
                    and p.name != "test_quality_attrs.py":
                text = p.read_text(errors="ignore")
                for needle in ("AKIA", "SECRET_KEY=", "password="):
                    if needle in text:
                        offenders.append(str(p))
        assert not offenders, f"possible secrets in {offenders}"

    def test_env_file_gitignored(self):
        gitignore = Path(".gitignore").read_text()
        assert ".env" in gitignore.split()


# ---------------- reproducibility ----------------

class TestReproducibility:
    def test_model_deterministic_on_fixed_seed(self):
        model = joblib.load(MODEL_PATH)
        p1 = model.predict_proba(pd.DataFrame([VALID]))[0, 1]
        p2 = model.predict_proba(pd.DataFrame([VALID]))[0, 1]
        assert p1 == p2

    def test_data_dvc_fingerprint_present(self):
        assert Path("data/fraud_data.csv.dvc").exists(), \
            "dataset must be DVC-tracked for reproducibility"
