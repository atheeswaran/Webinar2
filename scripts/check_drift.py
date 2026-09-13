
"""Data-drift check: compare the current dataset against a reference snapshot.

Uses a simple PSI (Population Stability Index) per feature -- the same idea
industrial monitoring stacks (Evidently, Fiddler, NannyML) apply.
Exits 1 when drift exceeds threshold -> CI job fails -> triggers retraining.
"""
import json
import sys

import numpy as np
import pandas as pd

FEATURES = ["amount", "hour", "age", "merchant_risk", "distance_from_home"]
PSI_THRESHOLD = 0.25


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    qs = np.quantile(expected, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    e_perc = np.histogram(expected, qs)[0] / len(expected)
    a_perc = np.histogram(actual, qs)[0] / len(actual)
    e_perc = np.clip(e_perc, 1e-4, None)
    a_perc = np.clip(a_perc, 1e-4, None)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def main(current_path: str, reference_path: str) -> None:
    cur = pd.read_csv(current_path)
    ref = pd.read_csv(reference_path)
    report = {f: round(psi(ref[f].values, cur[f].values), 4) for f in FEATURES}
    drifted = {f: v for f, v in report.items() if v > PSI_THRESHOLD}
    print(json.dumps({"psi": report, "threshold": PSI_THRESHOLD,
                      "drifted_features": list(drifted)}, indent=2))
    if drifted:
        sys.exit(f"DRIFT DETECTED in {list(drifted)} -> retrain recommended")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
