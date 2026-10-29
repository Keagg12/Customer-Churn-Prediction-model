"""Train and honestly evaluate candidate churn models on whatever data the user mapped."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .preprocess import FeatureSpec, build_preprocessor, infer_feature_spec, prepare_frame
from .schema import Schema, make_target, validate_schema


def candidate_models(seed: int = 42, balance: bool = True) -> Dict[str, object]:
    cw = "balanced" if balance else None
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight=cw),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, class_weight=cw, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight=cw,
                                                n_jobs=-1, random_state=seed),
        "Gradient Boosting": HistGradientBoostingClassifier(class_weight=cw, random_state=seed),
    }
    return models


def metrics_at(y, p, threshold: float = 0.5) -> dict:
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    both = len(np.unique(y)) == 2
    return {
        "roc_auc": roc_auc_score(y, p) if both else np.nan,
        "average_precision": average_precision_score(y, p) if both else np.nan,
        "brier": brier_score_loss(y, p),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p, dtype=float)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), q=min(bins, len(df)), labels=False)
    g = df.groupby("bin").agg(mean_predicted=("p", "mean"), observed_rate=("y", "mean"), customers=("y", "size"))
    return g.reset_index(drop=True)


def leakage_screen(Xtr, ytr, Xte, yte, spec: FeatureSpec, threshold: float = 0.95) -> List[tuple]:
    """Flag single columns that predict the outcome almost perfectly on their own.
    That is usually a sign of leakage (the column is only known after the customer left)."""
    flagged = []
    if len(np.unique(yte)) < 2:
        return flagged
    for col in spec.used:
        sub = FeatureSpec(numeric=[col] if col in spec.numeric else [],
                          categorical=[col] if col in spec.categorical else [])
        try:
            pipe = Pipeline([("prep", build_preprocessor(sub)),
                             ("clf", DecisionTreeClassifier(max_depth=3, random_state=0))])
            pipe.fit(Xtr[sub.used], ytr)
            auc = roc_auc_score(yte, pipe.predict_proba(Xte[sub.used])[:, 1])
        except Exception:
            continue
        if auc >= threshold:
            flagged.append((col, round(float(auc), 4)))
    return flagged


@dataclass
class TrainResult:
    pipelines: Dict[str, Pipeline]
    metrics: pd.DataFrame            # one row per model: test metrics + cross-validation score
    best_name: str
    selection_metric: str
    spec: FeatureSpec
    schema: Schema
    split: dict
    warnings: List[str] = field(default_factory=list)
    leakage: List[tuple] = field(default_factory=list)
    y_test: Optional[np.ndarray] = None
    test_proba: Dict[str, np.ndarray] = field(default_factory=dict)
    background: Optional[pd.DataFrame] = None  # raw training rows used as the explanation baseline


def train_and_evaluate(df: pd.DataFrame, schema: Schema, test_size: float = 0.2, seed: int = 42,
                       balance: bool = True, cv_folds: int = 5, max_categories: int = 50,
                       selection_metric: str = "average_precision", model_names=None) -> TrainResult:
    errors, warns = validate_schema(df, schema)
    if errors:
        raise ValueError(" ".join(errors))
    y_all = make_target(df, schema)
    keep = y_all.notna()
    d = df.loc[keep].reset_index(drop=True)
    y = y_all[keep].astype(int).reset_index(drop=True)
    counts = y.value_counts()
    if len(counts) < 2 or counts.min() < 4:
        raise ValueError(f"Not enough examples of both classes to train and test (counts: {counts.to_dict()}).")

    spec = infer_feature_spec(d, schema.feature_columns, max_categories)
    if not spec.used:
        raise ValueError("No usable feature columns after automatic type checks.")
    X = prepare_frame(d, spec)

    groups = None
    if schema.id_column and d[schema.id_column].duplicated().any():
        groups = d[schema.id_column].astype(str).values
        warns.append("The ID column repeats; rows of the same ID are kept on one side of the split.")
    if groups is not None:
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(X, y, groups))
    else:
        tr, te = train_test_split(np.arange(len(y)), test_size=test_size, random_state=seed, stratify=y)
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr].values, y.iloc[te].values
    if len(np.unique(yte)) < 2 or len(np.unique(ytr)) < 2:
        raise ValueError("The split left one class missing in train or test; the data has too few churners.")

    folds = min(cv_folds, int(np.bincount(ytr).min()))
    if folds < 2:
        raise ValueError("Too few churn (or non-churn) examples in the training part for cross-validation.")
    if groups is not None:
        cv = list(StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed).split(Xtr, ytr, groups[tr]))
    else:
        cv = list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(Xtr, ytr))

    leakage = leakage_screen(Xtr, ytr, Xte, yte, spec)
    if leakage:
        warns.append("Possible leakage: " + ", ".join(f"{c} (alone AUC {a})" for c, a in leakage)
                     + ". Check that these are known BEFORE a customer leaves.")

    cands = candidate_models(seed, balance)
    if model_names:
        cands = {k: v for k, v in cands.items() if k in model_names}
    pipelines, rows, probas = {}, [], {}
    for name, clf in cands.items():
        try:
            pipe = Pipeline([("prep", build_preprocessor(spec)), ("clf", clf)])
            cv_score = float(np.mean(cross_val_score(pipe, Xtr, ytr, cv=cv, scoring=selection_metric)))
            pipe.fit(Xtr, ytr)
            p = pipe.predict_proba(Xte)[:, 1]
        except Exception as exc:
            warns.append(f"{name} could not be trained: {exc}")
            continue
        pipelines[name], probas[name] = pipe, p
        rows.append({"model": name, f"cv_{selection_metric}": cv_score, **metrics_at(yte, p)})
    if not rows:
        raise ValueError("No model could be trained. " + " ".join(warns))
    metrics = pd.DataFrame(rows).set_index("model")
    best = metrics[f"cv_{selection_metric}"].idxmax()
    split = {"rows_used": int(len(y)), "train": int(len(tr)), "test": int(len(te)),
             "churn_rate_all": float(y.mean()), "churn_rate_train": float(ytr.mean()),
             "churn_rate_test": float(yte.mean()), "cv_folds": folds, "grouped_by_id": groups is not None,
             "balanced_class_weights": balance, "seed": seed}
    background = d.iloc[tr][spec.used].sample(n=min(200, len(tr)), random_state=seed)
    return TrainResult(pipelines, metrics, best, selection_metric, spec, schema, split, warns, leakage,
                       yte, probas, background)
