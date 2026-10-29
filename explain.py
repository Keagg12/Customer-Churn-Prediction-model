"""Per-customer explanations, reported against the ORIGINAL columns (one-hot pieces are summed back)."""
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from .preprocess import FeatureSpec, feature_origins, prepare_frame


class ExplainError(Exception):
    pass


def positive_class_values(sv) -> np.ndarray:
    """Normalise SHAP output shapes (list per class, or n x f x classes) to n x f for the churn class."""
    if isinstance(sv, list):
        arr = np.asarray(sv[1] if len(sv) > 1 else sv[0])
    else:
        arr = np.asarray(sv)
        if arr.ndim == 3:
            arr = arr[:, :, -1]
    return arr


def aggregate_by_origin(values: np.ndarray, origins: List[str], index=None) -> pd.DataFrame:
    df = pd.DataFrame(values, columns=origins, index=index)
    return df.T.groupby(level=0, sort=False).sum().T


def explanation_unit(clf) -> str:
    if isinstance(clf, LogisticRegression):
        return "log-odds of churn"
    if isinstance(clf, (DecisionTreeClassifier, RandomForestClassifier)):
        return "probability of churn"
    return "probability of churn (permutation-based SHAP)"


def _shap_values(clf, X: np.ndarray, B: np.ndarray) -> np.ndarray:
    try:
        import shap
    except ImportError as exc:
        raise ExplainError("The 'shap' package is required for this model. Install it with: pip install shap") from exc
    if isinstance(clf, (DecisionTreeClassifier, RandomForestClassifier)):
        sv = shap.TreeExplainer(clf).shap_values(X)
    else:
        f = lambda a: clf.predict_proba(a)[:, 1]
        sv = shap.Explainer(f, B[:100])(X).values
    return positive_class_values(sv)


def explain_rows(pipeline, spec: FeatureSpec, rows: pd.DataFrame, background: pd.DataFrame) -> pd.DataFrame:
    """Contribution of each original column to each row's churn score.
    Positive = pushes risk up. Describes what the MODEL relies on, not proven causes of churn."""
    prep, clf = pipeline.named_steps["prep"], pipeline.named_steps["clf"]
    X = prep.transform(prepare_frame(rows, spec))
    B = prep.transform(prepare_frame(background, spec))
    if isinstance(clf, LogisticRegression):
        vals = (X - B.mean(axis=0)) * clf.coef_[0]   # exact for a linear model
    else:
        vals = _shap_values(clf, X, B)
    return aggregate_by_origin(vals, feature_origins(prep, spec), index=rows.index)


def top_drivers(contrib_row: pd.Series, k: int = 5, positive_only: bool = True) -> List[Tuple[str, float]]:
    s = contrib_row[contrib_row > 0] if positive_only else contrib_row
    s = s.reindex(s.abs().sort_values(ascending=False).index)
    return [(str(i), float(v)) for i, v in s.head(k).items()]
