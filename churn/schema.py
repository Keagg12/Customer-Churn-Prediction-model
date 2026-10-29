"""The user's choice of target/features, validated against the real data."""
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


def norm_value(v) -> Optional[str]:
    """Stable text form of a cell value so 1, 1.0, '1' and True/'true' compare sensibly."""
    if v is None or (not isinstance(v, (list, tuple)) and pd.isna(v)):
        return None
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v)).lower()
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return str(int(v)) if float(v).is_integer() else str(float(v))
    return str(v).strip().lower()


@dataclass
class Schema:
    target: str
    positive_labels: List = field(default_factory=list)  # values of `target` that mean "churned"
    feature_columns: List[str] = field(default_factory=list)
    id_column: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Schema":
        return cls(**d)


def target_value_counts(df: pd.DataFrame, target: str) -> pd.DataFrame:
    s = df[target].dropna()
    vc = s.map(lambda v: norm_value(v)).value_counts()
    return pd.DataFrame({"value": vc.index, "count": vc.values})


def validate_schema(df: pd.DataFrame, schema: Schema):
    """Return (errors, warnings). Errors block training."""
    errors, warnings = [], []
    cols = set(df.columns)
    if schema.target not in cols:
        return [f"Target column '{schema.target}' is not in the data."], warnings
    if schema.id_column and schema.id_column not in cols:
        errors.append(f"ID column '{schema.id_column}' is not in the data.")
    missing = [c for c in schema.feature_columns if c not in cols]
    if missing:
        errors.append(f"Feature columns not in the data: {missing}")
    if schema.target in schema.feature_columns:
        errors.append("The target column cannot also be a feature.")
    if schema.id_column and schema.id_column in schema.feature_columns:
        errors.append("The ID column should not be used as a feature.")
    if not schema.feature_columns:
        errors.append("No feature columns selected.")
    if not schema.positive_labels:
        errors.append("Choose which target value(s) mean 'churned'.")
        return errors, warnings
    present = {norm_value(v) for v in df[schema.target].dropna()}
    wanted = {norm_value(p) for p in schema.positive_labels}
    absent = wanted - present
    if absent:
        errors.append(f"Positive label(s) {sorted(absent)} never appear in '{schema.target}'.")
    if wanted - absent == present:
        errors.append("Every target value is marked as churned, so there is no non-churn class.")
    n_missing = int(df[schema.target].isna().sum())
    if n_missing:
        warnings.append(f"{n_missing} rows have no target value and will be dropped.")
    return errors, warnings


def make_target(df: pd.DataFrame, schema: Schema) -> pd.Series:
    """1 = churned, 0 = not churned, NaN = target missing (caller drops these)."""
    col = df[schema.target]
    keep = col.notna()
    pos = {norm_value(p) for p in schema.positive_labels}
    y = col[keep].map(norm_value).isin(pos).astype(int)
    return y.reindex(df.index)
