"""Workflow logic behind the dashboard, with no Streamlit in it so it can be tested on its own.

Covers: reading config/dataset.json, training and packaging a model with its held-out evaluation,
saving/loading that package, historical churn patterns, and the evaluation views (ROC, confusion
matrix, band lift, importance). Metrics are only ever computed from labeled held-out rows."""
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_curve

from .explain import ExplainError, explain_rows
from .models import train_and_evaluate
from .preprocess import infer_feature_spec
from .schema import Schema, make_target
from .scoring import Bundle, RiskBands, make_bundle


# ----------------------------------------------------------------------------- configuration
def load_dataset_config(path) -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def schema_from_config(df: pd.DataFrame, cfg: dict) -> Optional[Schema]:
    """Build the training Schema from config/dataset.json. None when the config does not fit this table."""
    target = cfg.get("target")
    if target not in df.columns or not cfg.get("positive_labels"):
        return None
    idc = cfg.get("id_column") if cfg.get("id_column") in df.columns else None
    skip = {target, idc} | set(cfg.get("exclude", []))
    return Schema(target=target, positive_labels=list(cfg["positive_labels"]),
                  feature_columns=[c for c in df.columns if c not in skip], id_column=idc)


def load_descriptions(path) -> Dict[str, str]:
    """Column dictionary from data_descriptions.csv (column name -> description). Empty if absent."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        d = pd.read_csv(p)
        return dict(zip(d["Column_name"].astype(str), d["Description"].astype(str)))
    except Exception:
        return {}


def pretty(name: str) -> str:
    """'SupportTicketsPerMonth' -> 'Support tickets per month'."""
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|_+", " ", str(name)).split()
    if not words:
        return str(name)
    text = " ".join(words)
    return text[0].upper() + text[1:].lower()


def file_fingerprint(path) -> dict:
    p = Path(path)
    s = p.stat()
    return {"name": p.name, "size": int(s.st_size), "mtime_ns": int(s.st_mtime_ns)}


# ----------------------------------------------------------------------------- model package
@dataclass
class Evaluation:
    """Everything the Model Insights page may show. All of it comes from the labeled held-out split."""
    metrics: pd.DataFrame
    best_name: str
    selection_metric: str
    split: dict
    warnings: List[str]
    leakage: List[tuple]
    excluded: Dict[str, str]
    y_test: np.ndarray
    test_proba: Dict[str, np.ndarray]
    train_source: dict
    trained_at: str
    balance: bool = False


@dataclass
class ModelPackage:
    bundle: Bundle
    evaluation: Evaluation

    @property
    def key(self) -> str:
        return f"{self.bundle.model_name}|{self.evaluation.trained_at}"


def train_package(df: pd.DataFrame, schema: Schema, source: dict, test_size: float = 0.2, seed: int = 42,
                  balance: bool = False, bands: Optional[RiskBands] = None, model_names=None,
                  model_name: Optional[str] = None) -> ModelPackage:
    """Train and compare the candidate models, then package the chosen one with its evaluation."""
    res = train_and_evaluate(df, schema, test_size=test_size, seed=seed, balance=balance, model_names=model_names)
    bundle = make_bundle(res, model_name or res.best_name, bands or RiskBands())
    ev = Evaluation(metrics=res.metrics, best_name=res.best_name, selection_metric=res.selection_metric,
                    split=res.split, warnings=list(res.spec.notes) + list(res.warnings), leakage=res.leakage,
                    excluded=dict(res.spec.excluded), y_test=np.asarray(res.y_test),
                    test_proba={k: np.asarray(v) for k, v in res.test_proba.items()},
                    train_source=source, trained_at=bundle.trained_at, balance=balance)
    return ModelPackage(bundle, ev)


def save_package(path, pkg: ModelPackage) -> None:
    """Writes the same {'bundle': ...} layout that churn.scoring.load_bundle reads, plus the evaluation."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"bundle": pkg.bundle, "evaluation": pkg.evaluation}, path)


def load_package(path) -> Tuple[Optional[ModelPackage], str]:
    """(package, "") on success, (None, reason) when missing, unreadable, from another scikit-learn, or
    saved without an evaluation (for example by the command line tool). Only load files you made yourself."""
    p = Path(path)
    if not p.exists():
        return None, "no saved model"
    try:
        data = joblib.load(p)
        bundle, ev = data["bundle"], data.get("evaluation")
    except Exception as exc:
        return None, f"saved model could not be read ({exc})"
    if ev is None:
        return None, "saved model has no evaluation (it was made by the command line tool)"
    saved = (bundle.versions or {}).get("scikit-learn")
    if saved and saved != sklearn.__version__:
        return None, f"saved model was built with scikit-learn {saved}, this machine has {sklearn.__version__}"
    return ModelPackage(bundle, ev), ""


def is_stale(pkg: ModelPackage, train_path) -> bool:
    """True when the training file on disk differs from the one the saved model was trained on."""
    p = Path(train_path)
    src = pkg.evaluation.train_source or {}
    if not p.exists() or src.get("name") != p.name:
        return False
    fp = file_fingerprint(p)
    return (fp["size"], fp["mtime_ns"]) != (src.get("size"), src.get("mtime_ns"))


# ----------------------------------------------------------------------------- scoring helpers
def customer_ids(raw: pd.DataFrame, id_column: Optional[str]) -> pd.Series:
    if id_column and id_column in raw.columns:
        return raw[id_column].astype(str)
    return pd.Series([f"row {i + 1}" for i in range(len(raw))], index=raw.index)


def predict_proba(bundle: Bundle, raw: pd.DataFrame) -> np.ndarray:
    from .preprocess import prepare_frame
    return bundle.pipeline.predict_proba(prepare_frame(raw, bundle.spec))[:, 1]


def missing_inputs(bundle: Bundle, raw: pd.DataFrame) -> List[str]:
    return [c for c in bundle.spec.used if c not in raw.columns]


def input_checks(bundle: Bundle, raw: pd.DataFrame) -> dict:
    """Plain-language findings about a table that is about to be scored."""
    miss = missing_inputs(bundle, raw)
    out = {"missing_columns": miss, "rows": int(len(raw)), "missing_values": {}, "unseen_categories": {},
           "non_numeric": {}}
    if miss:
        return out
    for c in bundle.spec.used:
        n = int(raw[c].isna().sum())
        if n:
            out["missing_values"][c] = n
    for c in bundle.spec.numeric:
        conv = pd.to_numeric(raw[c], errors="coerce")
        bad = int((conv.isna() & raw[c].notna()).sum())
        if bad:
            out["non_numeric"][c] = bad
    prep = bundle.pipeline.named_steps["prep"]
    if bundle.spec.categorical:
        enc = prep.named_transformers_["cat"].named_steps["onehot"]
        for c, known in zip(bundle.spec.categorical, enc.categories_):
            seen = set(raw[c].dropna().astype(str).unique()) - set(map(str, known))
            if seen:
                out["unseen_categories"][c] = sorted(seen)[:5]
    return out


def explain_subset(bundle: Bundle, raw_rows: pd.DataFrame) -> pd.DataFrame:
    """Contribution of every model input to each given customer's score (see explain.explain_rows)."""
    return explain_rows(bundle.pipeline, bundle.spec, raw_rows, bundle.background)


def global_importance(bundle: Bundle, sample: pd.DataFrame, max_rows: int = 1500) -> pd.Series:
    """Share of the model's total attention per input: mean |contribution| over a sample of customers.
    Needs no labels. Shows what the model relies on, not what causes churn."""
    rows = sample.sample(n=min(max_rows, len(sample)), random_state=0) if len(sample) > max_rows else sample
    c = explain_subset(bundle, rows).abs().mean()
    total = float(c.sum())
    return (c / total if total > 0 else c).sort_values(ascending=False)


# ----------------------------------------------------------------------------- historical patterns
def _bin_label(lo, hi, first: bool, last: bool) -> str:
    if first:
        return f"up to {hi:g}"
    if last:
        return f"above {lo:g}"
    return f"{lo:g} to {hi:g}"


def churn_patterns(df: pd.DataFrame, schema: Schema, bins: int = 5, max_values: int = 12,
                   max_categories: int = 12) -> pd.DataFrame:
    """Observed churn rate by group for every usable input column of LABELED data.
    Columns: feature, kind, order, group, customers, churn_rate, lift (rate / overall rate)."""
    y = make_target(df, schema)
    keep = y.notna().to_numpy()
    d, yy = df.loc[keep], y[keep].astype(int)
    base = float(yy.mean()) if len(yy) else float("nan")
    spec = infer_feature_spec(d, schema.feature_columns, max_categories=max_categories)
    rows = []

    def add(feature, kind, group_series, labels):
        g = pd.DataFrame({"g": group_series.to_numpy(), "y": yy.to_numpy()}).dropna(subset=["g"])
        agg = g.groupby("g", observed=True)["y"].agg(["size", "mean"]).reset_index()
        for i, r in agg.iterrows():
            rows.append({"feature": feature, "kind": kind, "order": int(i), "group": labels(r["g"]),
                         "customers": int(r["size"]), "churn_rate": float(r["mean"]),
                         "lift": float(r["mean"] / base) if base else np.nan})

    for c in spec.numeric:
        s = pd.to_numeric(d[c], errors="coerce")
        if s.nunique() <= max_values:
            add(c, "numeric", s, lambda v: f"{v:g}")
            continue
        edges = np.unique(np.quantile(s.dropna(), np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            continue
        cut = pd.cut(s, edges, include_lowest=True, labels=False)
        n = len(edges) - 1
        add(c, "numeric", cut, lambda b, e=edges, n=n: _bin_label(e[int(b)], e[int(b) + 1], int(b) == 0, int(b) == n - 1))
    for c in spec.categorical:
        add(c, "category", d[c].astype(str), lambda v: str(v))
    return pd.DataFrame(rows, columns=["feature", "kind", "order", "group", "customers", "churn_rate", "lift"])


def pattern_strength(patterns: pd.DataFrame, min_share: float = 0.02) -> pd.DataFrame:
    """Rank inputs by how far observed churn spreads between their groups (ignoring tiny groups).
    This describes the historical data; it is not a causal effect."""
    if patterns.empty:
        return pd.DataFrame(columns=["feature", "low_group", "low_rate", "high_group", "high_rate", "spread"])
    out = []
    for f, g in patterns.groupby("feature", sort=False):
        g = g[g["customers"] >= min_share * g["customers"].sum()]
        if len(g) < 2:
            continue
        lo, hi = g.loc[g["churn_rate"].idxmin()], g.loc[g["churn_rate"].idxmax()]
        out.append({"feature": f, "low_group": lo["group"], "low_rate": lo["churn_rate"],
                    "high_group": hi["group"], "high_rate": hi["churn_rate"],
                    "spread": hi["churn_rate"] - lo["churn_rate"]})
    return pd.DataFrame(out).sort_values("spread", ascending=False).reset_index(drop=True)


def insight_sentences(strength: pd.DataFrame, base: float, k: int = 4) -> List[str]:
    out = []
    for _, r in strength.head(k).iterrows():
        out.append(f"{pretty(r['feature'])}: churn is {r['high_rate']:.1%} where it is {r['high_group']} "
                   f"and {r['low_rate']:.1%} where it is {r['low_group']} (overall {base:.1%}).")
    return out


# ----------------------------------------------------------------------------- evaluation views
def roc_points(y, proba: Dict[str, np.ndarray], max_points: int = 250) -> pd.DataFrame:
    rows = []
    for name, p in proba.items():
        fpr, tpr, _ = roc_curve(y, p)
        if len(fpr) > max_points:
            idx = np.unique(np.linspace(0, len(fpr) - 1, max_points).astype(int))
            fpr, tpr = fpr[idx], tpr[idx]
        rows.append(pd.DataFrame({"model": name, "fpr": fpr, "tpr": tpr}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["model", "fpr", "tpr"])


def confusion_at(y, p, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p, dtype=float) >= threshold).astype(int)
    return {"tn": int(((y == 0) & (pred == 0)).sum()), "fp": int(((y == 0) & (pred == 1)).sum()),
            "fn": int(((y == 1) & (pred == 0)).sum()), "tp": int(((y == 1) & (pred == 1)).sum())}


def band_lift(y, p, bands: RiskBands) -> pd.DataFrame:
    """Held-out check of the risk bands: how many customers, how many actually churned, share of all churners."""
    y = np.asarray(y).astype(int)
    lv = bands.assign(p)
    total = max(int(y.sum()), 1)
    rows = []
    for level in ("High", "Medium", "Low"):
        m = lv == level
        n = int(m.sum())
        rows.append({"risk_level": level, "customers": n, "share_of_customers": n / max(len(y), 1),
                     "observed_churn_rate": float(y[m].mean()) if n else float("nan"),
                     "share_of_churners": float(y[m].sum()) / total})
    return pd.DataFrame(rows)


def top_fraction_check(y, p, fraction: float) -> dict:
    """In held-out data: among the highest-scored `fraction` of customers, what share actually churned,
    and what share of all churners did they hold? Measures ranking quality, not the effect of any action."""
    y = np.asarray(y).astype(int)
    n = max(int(round(len(y) * min(max(fraction, 0.0), 1.0))), 1)
    order = np.argsort(-np.asarray(p, dtype=float), kind="stable")[:n]
    return {"customers": n, "observed_churn_rate": float(y[order].mean()),
            "share_of_churners": float(y[order].sum()) / max(int(y.sum()), 1)}


def probability_histogram(p, bands: RiskBands, step: float = 0.025) -> pd.DataFrame:
    p = np.asarray(p, dtype=float)
    edges = np.arange(0, 1 + step, step)
    counts, _ = np.histogram(p, bins=edges)
    mid = (edges[:-1] + edges[1:]) / 2
    return pd.DataFrame({"low": edges[:-1], "high": edges[1:], "customers": counts, "risk_level": bands.assign(mid)})


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
