"""Generic preprocessing driven by the data types found, not by assumed column names."""
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class FeatureSpec:
    numeric: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)
    excluded: Dict[str, str] = field(default_factory=dict)  # column -> reason
    notes: List[str] = field(default_factory=list)

    @property
    def used(self) -> List[str]:
        return self.numeric + self.categorical


def infer_feature_spec(df: pd.DataFrame, features, max_categories: int = 50) -> FeatureSpec:
    spec = FeatureSpec()
    for c in features:
        s = df[c]
        nn = s.dropna()
        if nn.empty:
            spec.excluded[c] = "all values missing"
        elif nn.nunique() <= 1:
            spec.excluded[c] = "constant (a single value)"
        elif pd.api.types.is_datetime64_any_dtype(s):
            spec.excluded[c] = "date/time column (not used automatically)"
        elif pd.api.types.is_bool_dtype(s):
            spec.categorical.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            spec.numeric.append(c)
        else:
            conv = pd.to_numeric(nn, errors="coerce")
            if conv.notna().mean() >= 0.95:
                spec.numeric.append(c)
                spec.notes.append(f"'{c}' holds numbers stored as text; converted to numeric.")
            elif nn.nunique() > max_categories:
                spec.excluded[c] = (f"{nn.nunique()} distinct text values (more than {max_categories}); "
                                    "likely free text or an identifier")
            else:
                spec.categorical.append(c)
    return spec


def prepare_frame(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Select and type the model inputs consistently for training and scoring."""
    missing = [c for c in spec.used if c not in df.columns]
    if missing:
        raise ValueError(f"These required columns are missing from the data: {missing}")
    out = pd.DataFrame(index=df.index)
    for c in spec.numeric:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c in spec.categorical:
        s = df[c]
        out[c] = s.map(lambda v: np.nan if pd.isna(v) else str(v)).astype(object)
    return out


def build_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    parts = []
    if spec.numeric:
        parts.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), spec.numeric))
    if spec.categorical:
        parts.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), spec.categorical))
    if not parts:
        raise ValueError("No usable feature columns after type checks.")
    return ColumnTransformer(parts, verbose_feature_names_out=False)


def feature_origins(fitted_prep: ColumnTransformer, spec: FeatureSpec) -> List[str]:
    """For every transformed column, the original column it came from."""
    origins = list(spec.numeric)
    if spec.categorical:
        enc = fitted_prep.named_transformers_["cat"].named_steps["onehot"]
        for col, cats in zip(spec.categorical, enc.categories_):
            origins += [col] * len(cats)
    return origins
