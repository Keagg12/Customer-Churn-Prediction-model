"""Using a trained model: risk bands, saving/loading the model, and scoring customers."""
from dataclasses import dataclass
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn

from .models import TrainResult
from .preprocess import FeatureSpec, prepare_frame
from .schema import Schema

LEVELS = ("Low", "Medium", "High")


@dataclass
class RiskBands:
    low_max: float = 0.30   # below this -> Low
    high_min: float = 0.70  # at or above this -> High

    def __post_init__(self):
        if not (0 < self.low_max < self.high_min < 1):
            raise ValueError("Thresholds must satisfy 0 < low < high < 1.")

    def assign(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        return np.where(p >= self.high_min, "High", np.where(p < self.low_max, "Low", "Medium"))


def band_summary(probabilities, bands: RiskBands) -> pd.DataFrame:
    lv = pd.Series(bands.assign(probabilities))
    counts = lv.value_counts().reindex(LEVELS, fill_value=0)
    return pd.DataFrame({"risk_level": counts.index, "customers": counts.values,
                         "share": (counts.values / max(len(lv), 1)).round(4)})


@dataclass
class Bundle:
    pipeline: object
    schema: Schema
    spec: FeatureSpec
    model_name: str
    metrics: dict
    bands: RiskBands
    background: pd.DataFrame
    trained_at: str
    versions: dict


def make_bundle(result: TrainResult, model_name: str = None, bands: RiskBands = None) -> Bundle:
    name = model_name or result.best_name
    return Bundle(result.pipelines[name], result.schema, result.spec, name,
                  {k: float(v) for k, v in result.metrics.loc[name].items()},
                  bands or RiskBands(), result.background,
                  datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  {"scikit-learn": sklearn.__version__, "pandas": pd.__version__})


def save_bundle(path, bundle: Bundle) -> None:
    joblib.dump({"bundle": bundle}, path)


def load_bundle(path) -> Bundle:
    """Only load model files you created yourself (they are Python pickles)."""
    return joblib.load(path)["bundle"]


def score_table(bundle: Bundle, df: pd.DataFrame) -> pd.DataFrame:
    X = prepare_frame(df, bundle.spec)
    proba = bundle.pipeline.predict_proba(X)[:, 1]
    out = pd.DataFrame(index=df.index)
    idc = bundle.schema.id_column
    if idc and idc in df.columns:
        out[idc] = df[idc]
    out["churn_probability"] = np.round(proba, 6)
    out["risk_level"] = bundle.bands.assign(proba)
    return out
