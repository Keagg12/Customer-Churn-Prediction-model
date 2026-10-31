from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    target: str = Field(min_length=1, max_length=200)
    positive_labels: list[Any] = Field(min_length=1)
    feature_columns: list[str] = Field(min_length=1)
    id_column: Optional[str] = None
    test_size: float = Field(default=0.2, gt=0.05, lt=0.5)
    seed: int = 42
    balance: bool = False
    persist: bool = False
    model_name: Optional[Literal[
        "Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"
    ]] = None


class RiskBandsRequest(BaseModel):
    low_max: float = Field(gt=0, lt=1)
    high_min: float = Field(gt=0, lt=1)


class SelectionRequest(BaseModel):
    customer_id: str = Field(min_length=1, max_length=500)
