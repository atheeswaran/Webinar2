
"""Generate the synthetic card-transaction dataset for the fraud demo.

Industrial framing: a nightly ETL job in a payments platform lands a fresh
fraud_data.csv. Changing the seed simulates a *different* data snapshot --
DVC detects the change via the md5 fingerprint and versions it.
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42          # change to 99 to simulate a new data snapshot
N_ROWS = 2000      # change to 1000 to simulate a smaller extract

FEATURES = ["amount", "hour", "age", "merchant_risk", "distance_from_home"]
LABEL = "is_fraud"


def generate(n_rows: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    amount = rng.lognormal(mean=4.0, sigma=1.0, size=n_rows)
    hour = rng.integers(0, 24, size=n_rows)
    age = rng.integers(18, 80, size=n_rows)
    merchant_risk = rng.uniform(0, 1, size=n_rows)
    distance = rng.exponential(scale=50, size=n_rows)

    # Fraud is more likely at night, high amounts, risky merchants, far away
    logit = (-12.0
             + 1.2 * np.log1p(amount)
             + 2.5 * ((hour >= 23) | (hour <= 5)).astype(float)
             + 5.0 * merchant_risk
             + 0.03 * distance
             - 0.02 * np.maximum(age - 40, 0))
    prob = 1 / (1 + np.exp(-logit))
    is_fraud = (rng.uniform(size=n_rows) < prob).astype(int)

    df = pd.DataFrame({
        "amount": amount.round(2),
        "hour": hour,
        "age": age,
        "merchant_risk": merchant_risk.round(3),
        "distance_from_home": distance.round(1),
        "is_fraud": is_fraud,
    })
    return df


if __name__ == "__main__":
    df = generate()
    out = Path(__file__).parent / "fraud_data.csv"
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} rows -> {out}  (fraud rate {df['is_fraud'].mean():.3f})")
