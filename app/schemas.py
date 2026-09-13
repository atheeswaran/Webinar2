
"""Pydantic schemas for the fraud-scoring API."""
from pydantic import BaseModel, Field


class Transaction(BaseModel):
    amount: float = Field(gt=0, le=100_000)
    hour: int = Field(ge=0, le=23)
    age: int = Field(ge=18, le=110)
    merchant_risk: float = Field(ge=0, le=1)
    distance_from_home: float = Field(ge=0, le=20_000)


class FraudScore(BaseModel):
    is_fraud: bool
    fraud_probability: float
    model_version: str
