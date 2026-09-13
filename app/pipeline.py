
"""Feature validation + preprocessing pipeline shared by training and serving."""
import numpy as np

MAX_AMOUNT = 100_000.0
MAX_DISTANCE = 20_000.0


def validate_input(payload: dict) -> dict:
    """Business-rule validation. Raises ValueError on bad input.

    Mirrors the checks the training data generator applies, so the model
    never sees out-of-distribution values in production.
    """
    amount = float(payload["amount"])
    if not 0 < amount <= MAX_AMOUNT:
        raise ValueError(f"amount {amount} out of range (0, {MAX_AMOUNT}]")
    hour = int(payload["hour"])
    if not 0 <= hour <= 23:
        raise ValueError(f"hour {hour} must be 0-23")
    age = int(payload["age"])
    if not 18 <= age <= 110:
        raise ValueError(f"age {age} must be 18-110")
    risk = float(payload["merchant_risk"])
    if not 0.0 <= risk <= 1.0:
        raise ValueError("merchant_risk must be in [0, 1]")
    dist = float(payload["distance_from_home"])
    if not 0 <= dist <= MAX_DISTANCE:
        raise ValueError(f"distance_from_home {dist} out of range")
    return {
        "amount": amount,
        "hour": hour,
        "age": age,
        "merchant_risk": risk,
        "distance_from_home": dist,
    }


def featurize(payload: dict) -> np.ndarray:
    v = validate_input(payload)
    return np.array([[v[f] for f in
                      ("amount", "hour", "age", "merchant_risk",
                       "distance_from_home")]])
