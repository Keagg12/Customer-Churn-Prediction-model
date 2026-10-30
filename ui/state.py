"""Session state, caching and the data/model lifecycle shared by every page.

Order of preference everywhere: what the person loaded in this session, then the files in data/ and
models/, then a clear setup state. Nothing is invented: with no data there are no customers on screen."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

from churn import data as D
from churn import workspace as W
from churn.batch import recommend_batch
from churn.explain import ExplainError
from churn.recommend import Rule, load_rules, validate_rules
from churn.schema import Schema
from churn.scoring import RiskBands, band_summary

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR, CONFIG_DIR = ROOT / "data", ROOT / "config"
MODEL_PATH = ROOT / "models" / "model.joblib"
TRAIN_CSV, TEST_CSV = DATA_DIR / "train.csv", DATA_DIR / "test.csv"
RULES_PATH, DATASET_CFG, DESCRIPTIONS = CONFIG_DIR / "rules.json", CONFIG_DIR / "dataset.json", DATA_DIR / "data_descriptions.csv"

PAGES = ["Overview", "Customer Explorer", "Customer Intelligence", "Model Insights", "Retention Center",
         "Prediction Workspace"]
S = st.session_state


def go(page: str) -> None:
    """Button callback: switch page (callbacks run before widgets are drawn, so the nav radio follows)."""
    S["nav"] = page


# ----------------------------------------------------------------------------- reading files
@st.cache_resource(show_spinner=False)
def _read_cached(path: str, size: int, mtime_ns: int) -> pd.DataFrame:
    return D.load_table(path)


def read_local(path: Path) -> Tuple[Optional[pd.DataFrame], str]:
    """(table, "") or (None, reason). A missing file is not an error: (None, "")."""
    if not path.exists():
        return None, ""
    s = path.stat()
    try:
        return _read_cached(str(path), int(s.st_size), int(s.st_mtime_ns)), ""
    except D.DataLoadError as exc:
        return None, f"{path.name}: {exc}"


@st.cache_resource(show_spinner=False)
def _load_saved(path: str, size: int, mtime_ns: int):
    return W.load_package(Path(path))


def saved_package() -> Tuple[Optional[W.ModelPackage], str]:
    if not MODEL_PATH.exists():
        return None, "no saved model"
    s = MODEL_PATH.stat()
    return _load_saved(str(MODEL_PATH), int(s.st_size), int(s.st_mtime_ns))


# ----------------------------------------------------------------------------- training
def clear_derived_caches() -> None:
    _load_saved.clear()
    _score.clear()
    _importance.clear()
    _actions.clear()
    _patterns.clear()


def train_and_activate(df: pd.DataFrame, schema: Schema, source: dict, *, test_size: float = 0.2, seed: int = 42,
                       balance: bool = False, persist: bool = True, model_name: Optional[str] = None) -> W.ModelPackage:
    """Train, then make the package the active model. persist=True also saves it to models/model.joblib."""
    pkg = W.train_package(df, schema, source, test_size=test_size, seed=seed, balance=balance, model_name=model_name)
    saved = False
    if persist:
        try:
            W.save_package(MODEL_PATH, pkg)
            saved = True
        except OSError as exc:
            S["save_error"] = f"The model could not be saved to models/model.joblib ({exc}). It is active for this session only."
    clear_derived_caches()
    if saved:
        S.pop("pkg_override", None)
    else:
        S["pkg_override"] = pkg
    S.pop("bands", None)
    return pkg


# ----------------------------------------------------------------------------- scoring caches
@st.cache_resource(show_spinner=False, max_entries=4)
def _score(_pkg: W.ModelPackage, pkg_key: str, _raw: pd.DataFrame, raw_key: str) -> np.ndarray:
    return W.predict_proba(_pkg.bundle, _raw)


@st.cache_resource(show_spinner=False, max_entries=8)
def _frame(_proba: np.ndarray, _ids: pd.Series, raw_key: str, pkg_key: str, low: float, high: float) -> pd.DataFrame:
    return pd.DataFrame({"customer_id": _ids.to_numpy(), "churn_probability": _proba,
                         "risk_level": RiskBands(low, high).assign(_proba)})


@st.cache_data(show_spinner=False, max_entries=4)
def _importance(pkg_key: str, raw_key: str, _pkg: W.ModelPackage, _raw: pd.DataFrame):
    try:
        return W.global_importance(_pkg.bundle, _raw[_pkg.bundle.spec.used]), ""
    except ExplainError as exc:
        return pd.Series(dtype=float), str(exc)


@st.cache_data(show_spinner=False, max_entries=6)
def _patterns(labeled_key: str, _df: pd.DataFrame, _schema: Schema):
    pat = W.churn_patterns(_df, _schema)
    return pat, W.pattern_strength(pat)


@st.cache_data(show_spinner=False, max_entries=24)
def _actions(pkg_key: str, raw_key: str, low: float, high: float, rules_sig: str, pos_bytes: bytes,
             _pkg: W.ModelPackage, _raw: pd.DataFrame, _proba: np.ndarray, _rules: list):
    pos = np.frombuffer(pos_bytes, dtype=np.int64)
    rows = _raw.iloc[pos]
    try:
        contrib = W.explain_subset(_pkg.bundle, rows)
    except ExplainError as exc:
        return None, str(exc)
    levels = RiskBands(low, high).assign(_proba[pos])
    out = recommend_batch(_rules, levels, contrib, rows)
    out.index = pos
    return out, ""


# ----------------------------------------------------------------------------- the context every page receives
@dataclass
class Portfolio:
    raw: pd.DataFrame            # customers being scored (default index, so position == label)
    proba: np.ndarray
    ids: pd.Series
    name: str
    key: str
    in_sample: bool = False


@dataclass
class Ctx:
    labeled: Optional[pd.DataFrame] = None
    labeled_name: str = ""
    labeled_key: str = ""
    schema: Optional[Schema] = None
    pkg: Optional[W.ModelPackage] = None
    bands: RiskBands = field(default_factory=RiskBands)
    portfolio: Optional[Portfolio] = None
    frame: Optional[pd.DataFrame] = None
    rules: List[Rule] = field(default_factory=list)
    rules_sig: str = ""
    rule_problems: List[str] = field(default_factory=list)
    descriptions: Dict[str, str] = field(default_factory=dict)
    patterns: Optional[pd.DataFrame] = None
    strength: Optional[pd.DataFrame] = None
    base_rate: Optional[float] = None
    n_labeled: int = 0
    n_churned: int = 0
    notes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    stale: bool = False

    @property
    def ready(self) -> bool:
        return self.pkg is not None and self.portfolio is not None and self.frame is not None

    @property
    def linear(self) -> bool:
        return self.pkg is not None and isinstance(self.pkg.bundle.pipeline.named_steps["clf"], LogisticRegression)

    @property
    def action_cap(self) -> int:
        """Most customers whose rule matches are computed at once (SHAP on other model types is slower)."""
        return 3000 if self.linear else 150

    def importance(self) -> pd.Series:
        if not self.ready:
            return pd.Series(dtype=float)
        s, err = _importance(self.pkg.key, self.portfolio.key, self.pkg, self.portfolio.raw)
        if err and err not in self.notes:
            self.notes.append(err)
        return s

    def summary(self) -> pd.DataFrame:
        return band_summary(self.frame["churn_probability"], self.bands)

    def actions(self, positions) -> Optional[pd.DataFrame]:
        """Top factors and rule matches for customers at these positions (None if the model cannot be explained here)."""
        pos = np.asarray(positions, dtype=np.int64)
        if not self.ready or len(pos) == 0:
            return pd.DataFrame(columns=["top_factors", "n_matched", "action_ids", "actions", "next_action"])
        out, err = _actions(self.pkg.key, self.portfolio.key, self.bands.low_max, self.bands.high_min, self.rules_sig,
                            pos.tobytes(), self.pkg, self.portfolio.raw, self.portfolio.proba, self.rules)
        if err and err not in self.errors:
            self.errors.append(err)
        return out


def _rules(ctx: Ctx) -> None:
    if not RULES_PATH.exists():
        return
    try:
        ctx.rules = load_rules(RULES_PATH)
        s = RULES_PATH.stat()
        ctx.rules_sig = f"{s.st_size}-{s.st_mtime_ns}"
    except (ValueError, KeyError, TypeError) as exc:
        ctx.errors.append(f"config/rules.json is not valid: {exc}")


def build_ctx() -> Ctx:
    """Resolve data, model, risk bands and the scored customers for this run. Cheap when everything is cached."""
    ctx = Ctx()
    cfg = W.load_dataset_config(DATASET_CFG)
    ctx.descriptions = W.load_descriptions(DESCRIPTIONS)

    # labeled history: an uploaded file the person trained on in this session, else data/train.csv
    ov = S.get("labeled_override")
    if ov is not None:
        ctx.labeled, ctx.schema, ctx.labeled_name = ov["df"], ov["schema"], ov["name"]
        ctx.labeled_key = f"upload|{ov['name']}|{len(ov['df'])}|{ov.get('size', 0)}"
    else:
        df, err = read_local(TRAIN_CSV)
        if err:
            ctx.errors.append(err)
        if df is not None:
            ctx.labeled, ctx.labeled_name = df, "data/train.csv"
            s = TRAIN_CSV.stat()
            ctx.labeled_key = f"train|{s.st_size}|{s.st_mtime_ns}"
            ctx.schema = W.schema_from_config(df, cfg)
            if ctx.schema is None:
                ctx.errors.append("config/dataset.json does not match the columns of data/train.csv "
                                  "(target, positive_labels). Fix the file or train from the Prediction Workspace.")
    if ctx.labeled is not None and ctx.schema is not None:
        ctx.patterns, ctx.strength = _patterns(ctx.labeled_key, ctx.labeled, ctx.schema)
        from churn.schema import make_target
        y = make_target(ctx.labeled, ctx.schema).dropna()
        ctx.base_rate = float(y.mean()) if len(y) else None
        ctx.n_labeled, ctx.n_churned = int(len(y)), int(y.sum())

    # model
    ctx.pkg = S.get("pkg_override")
    if ctx.pkg is None:
        pkg, reason = saved_package()
        if pkg is None and reason and reason != "no saved model":
            ctx.notes.append(f"The saved model was not used: {reason}. A new one is trained from data/train.csv.")
        ctx.pkg = pkg
    if ctx.pkg is None and ctx.labeled is not None and ctx.schema is not None and ov is None \
            and not S.get("auto_train_failed"):
        n = len(ctx.labeled)
        with st.spinner(f"First launch: training and comparing four models on {n:,} labeled customers. "
                        "This can take a few minutes; the result is saved to models/ so it only happens once."):
            try:
                ctx.pkg = train_and_activate(ctx.labeled, ctx.schema, W.file_fingerprint(TRAIN_CSV))
            except Exception as exc:  # surface any training problem as a setup state instead of a crash
                S["auto_train_failed"] = f"{type(exc).__name__}: {exc}"
    if S.get("auto_train_failed"):
        ctx.errors.append(f"Automatic training failed: {S['auto_train_failed']}")
    if S.get("save_error"):
        ctx.notes.append(S["save_error"])
    if ctx.pkg is not None:
        ctx.stale = ov is None and S.get("pkg_override") is None and W.is_stale(ctx.pkg, TRAIN_CSV)
        low, high = S.get("bands", (ctx.pkg.bundle.bands.low_max, ctx.pkg.bundle.bands.high_min))
        try:
            ctx.bands = RiskBands(low, high)
        except ValueError:
            ctx.bands = ctx.pkg.bundle.bands
            S.pop("bands", None)

    _rules(ctx)
    if ctx.pkg is not None:
        ctx.rule_problems = validate_rules(ctx.rules, ctx.pkg.bundle.spec.used)

    # customers being scored: an uploaded file, else data/test.csv, else the training rows (flagged in-sample)
    if ctx.pkg is not None:
        po = S.get("portfolio_override")
        raw, name, key, in_sample = None, "", "", False
        if po is not None:
            raw, name, key = po["raw"], po["name"], f"upload|{po['name']}|{len(po['raw'])}|{po.get('size', 0)}"
        else:
            raw, err = read_local(TEST_CSV)
            if err:
                ctx.errors.append(err)
            if raw is not None:
                s = TEST_CSV.stat()
                name, key = "data/test.csv", f"test|{s.st_size}|{s.st_mtime_ns}"
            elif ctx.labeled is not None:
                raw, name, key, in_sample = ctx.labeled, ctx.labeled_name + " (training rows)", ctx.labeled_key, True
        if raw is not None:
            miss = W.missing_inputs(ctx.pkg.bundle, raw)
            if miss:
                ctx.errors.append(f"{name} is missing columns the model needs: {', '.join(miss)}.")
            else:
                if not raw.index.equals(pd.RangeIndex(len(raw))):
                    raw = raw.reset_index(drop=True)
                proba = _score(ctx.pkg, ctx.pkg.key, raw, key)
                ids = W.customer_ids(raw, ctx.pkg.bundle.schema.id_column)
                ctx.portfolio = Portfolio(raw, proba, ids, name, key, in_sample)
                ctx.frame = _frame(proba, ids, key, ctx.pkg.key, ctx.bands.low_max, ctx.bands.high_min)
    return ctx


def reset_session_to_files() -> None:
    """Forget everything loaded in this session and go back to the files in data/ and models/."""
    for k in ("pkg_override", "labeled_override", "portfolio_override", "bands", "save_error", "auto_train_failed",
              "selected_id"):
        S.pop(k, None)
    clear_derived_caches()


def find_customer(ctx: Ctx, text: str) -> Optional[int]:
    """Position of a customer by exact ID (case-insensitive), or None."""
    if not ctx.ready or not text.strip():
        return None
    hit = np.flatnonzero(ctx.frame["customer_id"].str.lower().to_numpy() == text.strip().lower())
    return int(hit[0]) if len(hit) else None
