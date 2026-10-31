"""Request-scoped orchestration around the existing churn business logic."""
from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from churn import data as D
from churn import workspace as W
from churn.batch import recommend_batch
from churn.explain import ExplainError, explanation_unit, top_drivers
from churn.models import calibration_table
from churn.recommend import LEVELS, Rule, load_rules, recommend, validate_rules
from churn.schema import Schema, make_target, target_value_counts, validate_schema
from churn.scoring import RiskBands, band_summary

from .schemas import TrainRequest

PAGES = [
    "Overview",
    "Customer Explorer",
    "Customer Intelligence",
    "Model Insights",
    "Retention Center",
    "Prediction Workspace",
]
MAX_UPLOAD_MB = int(os.getenv("CHURN_MAX_UPLOAD_MB", "500"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
ALLOW_PERSIST = os.getenv("CHURN_ALLOW_PERSIST", "true").strip().lower() in {"1", "true", "yes", "on"}


class ApiError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if not isinstance(value, (str, bytes, int, float, bool, list, tuple, dict)):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): _json_value(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _float_map(values: pd.Series) -> dict[str, float]:
    return {str(key): float(value) for key, value in values.items() if np.isfinite(value)}


@dataclass
class SessionData:
    train_upload: Optional[pd.DataFrame] = None
    train_name: str = ""
    train_digest: str = ""
    labeled_override: Optional[pd.DataFrame] = None
    schema_override: Optional[Schema] = None
    labeled_name: str = ""
    pkg_override: Optional[W.ModelPackage] = None
    portfolio_override: Optional[pd.DataFrame] = None
    portfolio_name: str = ""
    portfolio_digest: str = ""
    bands: Optional[tuple[float, float]] = None
    selected_id: str = ""
    last_accessed: float = field(default_factory=time.monotonic)


@dataclass
class RuntimeContext:
    labeled: Optional[pd.DataFrame]
    schema: Optional[Schema]
    labeled_name: str
    pkg: Optional[W.ModelPackage]
    bands: RiskBands
    raw: Optional[pd.DataFrame]
    portfolio_name: str
    portfolio_key: str
    in_sample: bool
    probabilities: Optional[np.ndarray]
    frame: Optional[pd.DataFrame]
    patterns: pd.DataFrame
    strength: pd.DataFrame
    rules: list[Rule]
    rule_problems: list[str]
    descriptions: dict[str, str]
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    stale: bool = False

    @property
    def ready(self) -> bool:
        return self.pkg is not None and self.raw is not None and self.frame is not None

    @property
    def action_cap(self) -> int:
        return 3000 if self.pkg and self.pkg.bundle.model_name == "Logistic Regression" else 150


class DashboardService:
    """Loads the shared local project once and keeps browser-session overrides in memory."""

    def __init__(self, root: Optional[Path] = None, auto_train: Optional[bool] = None):
        self.root = Path(root or Path(__file__).resolve().parents[1])
        self.data_dir = Path(os.getenv("CHURN_DATA_DIR", self.root / "data"))
        self.config_dir = Path(os.getenv("CHURN_CONFIG_DIR", self.root / "config"))
        self.model_path = Path(os.getenv("CHURN_MODEL_PATH", self.root / "models" / "model.joblib"))
        self.train_path = self.data_dir / "train.csv"
        self.test_path = self.data_dir / "test.csv"
        auto_value = os.getenv("CHURN_AUTO_TRAIN", "false").strip().lower() in {"1", "true", "yes", "on"}
        self.auto_train = auto_value if auto_train is None else auto_train
        self.config: dict = {}
        self.descriptions: dict[str, str] = {}
        self.train_df: Optional[pd.DataFrame] = None
        self.test: Optional[pd.DataFrame] = None
        self.schema: Optional[Schema] = None
        self.pkg: Optional[W.ModelPackage] = None
        self.rules: list[Rule] = []
        self.startup_errors: list[str] = []
        self.startup_notes: list[str] = []
        self._initialized = False
        self._init_lock = threading.RLock()
        self._sessions: dict[str, SessionData] = {}
        self.session_ttl = max(60, int(os.getenv("CHURN_SESSION_TTL_SECONDS", "1800")))
        self.max_sessions = max(1, int(os.getenv("CHURN_MAX_SESSIONS", "128")))
        self._session_lock = threading.RLock()
        self._score_cache: dict[tuple, tuple[np.ndarray, pd.DataFrame]] = {}
        self._derived_cache: dict[tuple, tuple[pd.DataFrame, pd.DataFrame]] = {}
        self._cache_lock = threading.RLock()

    def initialize(self) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            self.config = W.load_dataset_config(self.config_dir / "dataset.json")
            self.descriptions = W.load_descriptions(self.data_dir / "data_descriptions.csv")
            try:
                self.rules = load_rules(self.config_dir / "rules.json")
            except FileNotFoundError:
                self.startup_notes.append("config/rules.json is missing; predictions are available without retention actions.")
            except (ValueError, KeyError, TypeError) as exc:
                self.startup_errors.append(f"config/rules.json is invalid: {exc}")

            self.train_df = self._read_local(self.train_path, required=False)
            if self.train_df is not None:
                self.schema = W.schema_from_config(self.train_df, self.config)
                if self.schema is None:
                    self.startup_errors.append("config/dataset.json does not match data/train.csv (target and positive_labels).")
            self.test = self._read_local(self.test_path, required=False)

            self.pkg, reason = W.load_package(self.model_path)
            if self.pkg is None and reason and reason != "no saved model":
                self.startup_notes.append(f"The saved model was not used: {reason}.")
            if self.pkg is None and self.auto_train and self.train_df is not None and self.schema is not None:
                try:
                    source = W.file_fingerprint(self.train_path)
                    self.pkg = W.train_package(self.train_df, self.schema, source)
                    W.save_package(self.model_path, self.pkg)
                except Exception as exc:
                    self.startup_errors.append(f"Automatic training failed: {type(exc).__name__}: {exc}")
            if self.pkg is not None:
                problems = validate_rules(self.rules, self.pkg.bundle.spec.used)
                if problems:
                    self.startup_notes.extend(problems)
            self._initialized = True

    def _read_local(self, path: Path, required: bool) -> Optional[pd.DataFrame]:
        if not path.exists():
            if required:
                raise ApiError(503, f"Required data file not found: {path.name}.")
            return None
        try:
            return D.load_table(path)
        except D.DataLoadError as exc:
            self.startup_errors.append(str(exc))
            return None

    def _session(self, session_id: str) -> SessionData:
        if len(session_id) > 160:
            raise ApiError(400, "Session ID is too long.")
        with self._session_lock:
            now = time.monotonic()
            expired = [key for key, value in self._sessions.items() if now - value.last_accessed > self.session_ttl]
            for key in expired:
                self._sessions.pop(key, None)
            if expired:
                with self._cache_lock:
                    self._score_cache.clear()
                    self._derived_cache.clear()
            session = self._sessions.get(session_id)
            if session is None:
                if len(self._sessions) >= self.max_sessions:
                    oldest = min(self._sessions, key=lambda key: self._sessions[key].last_accessed)
                    self._sessions.pop(oldest, None)
                session = SessionData()
                self._sessions[session_id] = session
            session.last_accessed = now
            return session

    def _labelled_for(self, session: SessionData) -> tuple[Optional[pd.DataFrame], Optional[Schema], str, str]:
        if session.labeled_override is not None:
            return session.labeled_override, session.schema_override, session.labeled_name, session.train_digest
        if self.train_df is None:
            return None, None, "", ""
        try:
            stat = self.train_path.stat()
            key = f"{stat.st_size}:{stat.st_mtime_ns}"
        except OSError:
            key = f"local:{len(self.train_df)}"
        return self.train_df, self.schema, "data/train.csv", key

    def _context(self, session_id: str) -> RuntimeContext:
        self.initialize()
        session = self._session(session_id)
        labeled, schema, labeled_name, labeled_key = self._labelled_for(session)
        pkg = session.pkg_override or self.pkg
        raw: Optional[pd.DataFrame]
        name = ""
        key = ""
        in_sample = False
        if session.portfolio_override is not None:
            raw, name = session.portfolio_override, session.portfolio_name
            key = session.portfolio_digest
        elif self.test is not None:
            raw, name = self.test, "data/test.csv"
            try:
                stat = self.test_path.stat()
                key = f"test:{stat.st_size}:{stat.st_mtime_ns}"
            except OSError:
                key = f"test:{len(self.test)}"
        elif labeled is not None:
            raw, name, key, in_sample = labeled, f"{labeled_name} (training rows)", labeled_key, True
        else:
            raw, name, key = None, "", ""

        bands = RiskBands(*(session.bands or ((pkg.bundle.bands.low_max, pkg.bundle.bands.high_min) if pkg else (0.15, 0.30))))
        errors, notes = list(self.startup_errors), list(self.startup_notes)
        patterns = pd.DataFrame(columns=["feature", "kind", "order", "group", "customers", "churn_rate", "lift"])
        strength = pd.DataFrame(columns=["feature", "low_group", "low_rate", "high_group", "high_rate", "spread"])
        if labeled is not None and schema is not None:
            pattern_key = (labeled_key, schema.target, tuple(schema.feature_columns))
            with self._cache_lock:
                cached = self._derived_cache.get(("patterns", *pattern_key))
            if cached is None:
                patterns = W.churn_patterns(labeled, schema)
                strength = W.pattern_strength(patterns)
                with self._cache_lock:
                    self._derived_cache[("patterns", *pattern_key)] = (patterns, strength)
            else:
                patterns, strength = cached

        stale = False
        if pkg is not None and raw is not None:
            if pkg is self.pkg and session.pkg_override is None:
                stale = W.is_stale(pkg, self.train_path)
            missing = W.missing_inputs(pkg.bundle, raw)
            if missing:
                errors.append(f"{name} is missing required model columns: {', '.join(missing)}.")
            else:
                raw = raw.reset_index(drop=True)
                score_key = (pkg.key, key, bands.low_max, bands.high_min)
                with self._cache_lock:
                    result = self._score_cache.get(score_key)
                if result is None:
                    probabilities = W.predict_proba(pkg.bundle, raw)
                    ids = W.customer_ids(raw, pkg.bundle.schema.id_column)
                    frame = pd.DataFrame({
                        "customer_id": ids.to_numpy(),
                        "churn_probability": probabilities,
                        "risk_level": bands.assign(probabilities),
                    })
                    result = (probabilities, frame)
                    with self._cache_lock:
                        if len(self._score_cache) >= 8:
                            self._score_cache.pop(next(iter(self._score_cache)))
                        self._score_cache[score_key] = result
                probabilities, frame = result
        else:
            probabilities, frame = None, None

        rule_problems = validate_rules(self.rules, pkg.bundle.spec.used) if pkg is not None else []
        return RuntimeContext(
            labeled=labeled,
            schema=schema,
            labeled_name=labeled_name,
            pkg=pkg,
            bands=bands,
            raw=raw,
            portfolio_name=name,
            portfolio_key=key,
            in_sample=in_sample,
            probabilities=probabilities,
            frame=frame,
            patterns=patterns,
            strength=strength,
            rules=self.rules,
            rule_problems=rule_problems,
            descriptions=self.descriptions,
            errors=errors,
            notes=notes,
            stale=stale,
        )

    def _require_ready(self, ctx: RuntimeContext) -> None:
        if not ctx.ready:
            if ctx.errors:
                raise ApiError(503, " ".join(ctx.errors))
            raise ApiError(409, "A usable model and customer portfolio are not available. Check data/train.csv and config/dataset.json.")

    def _actions(self, ctx: RuntimeContext, positions: np.ndarray) -> tuple[Optional[pd.DataFrame], str]:
        if not ctx.ready or len(positions) == 0:
            return pd.DataFrame(columns=["top_factors", "n_matched", "action_ids", "actions", "next_action"]), ""
        rows = ctx.raw.iloc[positions]
        try:
            contrib = W.explain_subset(ctx.pkg.bundle, rows)
        except ExplainError as exc:
            return None, str(exc)
        levels = ctx.bands.assign(ctx.probabilities[positions])
        out = recommend_batch(ctx.rules, levels, contrib, rows)
        out.index = positions
        return out, ""

    def _importance(self, ctx: RuntimeContext) -> tuple[pd.Series, str]:
        if not ctx.ready:
            return pd.Series(dtype=float), ""
        cache_key = ("importance", ctx.pkg.key, ctx.portfolio_key)
        with self._cache_lock:
            cached = self._derived_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            result = (W.global_importance(ctx.pkg.bundle, ctx.raw[ctx.pkg.bundle.spec.used]), "")
        except ExplainError as exc:
            result = (pd.Series(dtype=float), str(exc))
        with self._cache_lock:
            self._derived_cache[cache_key] = result
        return result

    def bootstrap(self, session_id: str) -> dict:
        ctx = self._context(session_id)
        session = self._session(session_id)
        return {
            "pages": PAGES,
            "ready": ctx.ready,
            "data": {
                "train_rows": int(len(ctx.labeled)) if ctx.labeled is not None else 0,
                "portfolio_rows": int(len(ctx.raw)) if ctx.raw is not None else 0,
                "labeled_name": ctx.labeled_name,
                "portfolio_name": ctx.portfolio_name,
                "in_sample": ctx.in_sample,
                "has_train": ctx.labeled is not None and ctx.schema is not None,
                "has_test": self.test is not None,
                "stale_model": ctx.stale,
            },
            "model": {
                "name": ctx.pkg.bundle.model_name if ctx.pkg else None,
                "trained_at": ctx.pkg.bundle.trained_at if ctx.pkg else None,
                "has_evaluation": bool(ctx.pkg and ctx.pkg.evaluation),
                "saved": self.model_path.exists() and session.pkg_override is None,
            },
            "bands": {"low_max": ctx.bands.low_max, "high_min": ctx.bands.high_min},
            "errors": ctx.errors,
            "notes": ctx.notes,
        }

    def overview(self, session_id: str) -> dict:
        ctx = self._context(session_id)
        observed = None
        if ctx.labeled is not None and ctx.schema is not None:
            target = make_target(ctx.labeled, ctx.schema)
            target = target.dropna()
            observed = {
                "rate": float(target.mean()) if len(target) else None,
                "customers": int(len(target)),
                "churned": int(target.sum()),
                "source": ctx.labeled_name,
            }
        bands = []
        histogram = []
        attention = []
        if ctx.ready:
            bands = _records(band_summary(ctx.frame["churn_probability"], ctx.bands))
            histogram = _records(W.probability_histogram(ctx.probabilities, ctx.bands))
            positions = np.argsort(-ctx.probabilities, kind="stable")[:8]
            actions, action_error = self._actions(ctx, positions)
            if action_error:
                ctx.notes.append(action_error)
            for i, pos in enumerate(positions):
                action = actions.iloc[i] if actions is not None else None
                factors = action["top_factors"][:2] if action is not None else []
                attention.append({
                    "customer_id": str(ctx.frame.iloc[pos]["customer_id"]),
                    "probability": float(ctx.probabilities[pos]),
                    "risk_level": str(ctx.frame.iloc[pos]["risk_level"]),
                    "why": factors,
                })
        return {
            "observed": observed,
            "predicted": {
                "customers": int(len(ctx.frame)) if ctx.ready else 0,
                "mean_probability": float(np.mean(ctx.probabilities)) if ctx.ready else None,
                "risk_bands": bands,
                "histogram": histogram,
                "attention": attention,
            },
            "patterns": _records(ctx.strength.head(8)) if not ctx.strength.empty else [],
            "pattern_groups": _records(ctx.patterns),
            "ready": ctx.ready,
            "portfolio_name": ctx.portfolio_name,
            "in_sample": ctx.in_sample,
            "stale_model": ctx.stale,
            "errors": ctx.errors,
            "notes": ctx.notes,
        }

    def customers(self, session_id: str, *, query: str = "", risk_levels: Optional[list[str]] = None,
                  probability_min: float = 0.0, probability_max: float = 1.0, segment: Optional[str] = None,
                  segment_values: Optional[list[str]] = None, sort_by: str = "probability", descending: bool = True,
                  limit: int = 100, offset: int = 0) -> dict:
        ctx = self._context(session_id)
        self._require_ready(ctx)
        if probability_min > probability_max:
            raise ApiError(422, "probability_min must not exceed probability_max.")
        frame, raw = ctx.frame, ctx.raw
        probs = ctx.probabilities
        levels = risk_levels or ["High", "Medium", "Low"]
        invalid_levels = set(levels) - set(LEVELS)
        if invalid_levels:
            raise ApiError(422, f"Unknown risk level(s): {', '.join(sorted(invalid_levels))}.")
        mask = frame["risk_level"].isin(levels).to_numpy(copy=True)
        mask &= (probs >= probability_min) & (probs <= probability_max)
        if query.strip():
            mask &= frame["customer_id"].astype(str).str.lower().str.contains(query.strip().lower(), regex=False).to_numpy()
        if segment:
            if segment not in raw.columns:
                raise ApiError(422, f"Unknown segment column: {segment}.")
            if segment_values:
                mask &= raw[segment].astype(str).isin(segment_values).to_numpy()
        positions = np.flatnonzero(mask)
        spec = ctx.pkg.bundle.spec
        if sort_by == "probability":
            values = probs[positions]
        elif sort_by in spec.numeric:
            values = pd.to_numeric(raw[sort_by], errors="coerce").to_numpy()[positions]
        else:
            raise ApiError(422, "sort_by must be probability or a numeric model feature.")
        fill = -np.inf if descending else np.inf
        order = np.argsort(np.nan_to_num(values, nan=fill) * (-1 if descending else 1), kind="stable")
        positions = positions[order]
        total = int(len(positions))
        page_positions = positions[offset:offset + limit]
        action_count = min(len(page_positions), ctx.action_cap)
        actions, action_error = self._actions(ctx, page_positions[:action_count])
        importance, importance_error = self._importance(ctx)
        char_cols = [name for name in importance.index if name in raw.columns][:5] or spec.used[:5]
        items = []
        for i, pos in enumerate(page_positions):
            row = raw.iloc[pos]
            action = actions.iloc[i] if actions is not None and i < action_count else None
            items.append({
                "customer_id": str(frame.iloc[pos]["customer_id"]),
                "probability": float(probs[pos]),
                "risk_level": str(frame.iloc[pos]["risk_level"]),
                "attributes": {c: _json_value(row.get(c)) for c in char_cols},
                "top_factors": action["top_factors"] if action is not None else [],
                "suggested_action": action["next_action"] if action is not None else "",
                "matched_rules": action["action_ids"] if action is not None else [],
            })
        return {
            "items": items,
            "total": total,
            "offset": offset,
            "limit": limit,
            "action_limit": action_count,
            "action_warning": action_error or importance_error or ("Actions are computed only for the first rows in this page." if action_count < len(page_positions) else ""),
            "segment_columns": spec.categorical,
            "segment_options": {
                column: sorted(raw[column].dropna().astype(str).unique().tolist())
                for column in spec.categorical
            },
            "numeric_columns": spec.numeric,
        }

    def customer_detail(self, session_id: str, customer_id: str) -> dict:
        ctx = self._context(session_id)
        self._require_ready(ctx)
        matches = np.flatnonzero(ctx.frame["customer_id"].astype(str).str.casefold().to_numpy() == customer_id.casefold())
        if not len(matches):
            raise ApiError(404, "No customer with that ID exists in the current portfolio.")
        pos = int(matches[0])
        row = ctx.raw.iloc[pos]
        probability = float(ctx.probabilities[pos])
        level = str(ctx.frame.iloc[pos]["risk_level"])
        try:
            contrib = W.explain_subset(ctx.pkg.bundle, ctx.raw.iloc[[pos]]).iloc[0]
            drivers = top_drivers(contrib, 6)
            explanation = [
                {"feature": str(name), "label": W.pretty(name), "contribution": float(value),
                 "direction": "raises" if value > 0 else "lowers", "value": _json_value(row.get(name)),
                 "typical": self._typical(ctx.raw[name], name in ctx.pkg.bundle.spec.numeric)}
                for name, value in contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(10).items()
            ]
            rec = recommend(ctx.rules, level, top_drivers(contrib, 5), row)
            explain_err = ""
            unit = explanation_unit(ctx.pkg.bundle.pipeline.named_steps["clf"])
        except ExplainError as exc:
            explanation, drivers, rec, unit = [], [], {"actions": [], "summary": "Rules unavailable because the model could not be explained."}, ""
            explain_err = str(exc)
        return {
            "customer_id": str(ctx.frame.iloc[pos]["customer_id"]),
            "probability": probability,
            "risk_level": level,
            "rank_percentile": float((ctx.probabilities < probability).mean()),
            "portfolio_mean": float(np.mean(ctx.probabilities)),
            "model_name": ctx.pkg.bundle.model_name,
            "explanation_unit": unit,
            "attributes": {str(c): _json_value(row.get(c)) for c in ctx.pkg.bundle.spec.used},
            "descriptions": {str(c): ctx.descriptions.get(c, "") for c in ctx.pkg.bundle.spec.used},
            "explanation": explanation,
            "drivers": [{"feature": name, "contribution": value} for name, value in drivers],
            "recommendations": rec,
            "explanation_error": explain_err,
            "rule_warnings": ctx.rule_problems,
        }

    @staticmethod
    def _typical(series: pd.Series, numeric: bool) -> Any:
        if numeric:
            value = pd.to_numeric(series, errors="coerce").median()
            return _json_value(value)
        modes = series.dropna().astype(str).mode()
        return str(modes.iloc[0]) if len(modes) else None

    def retention(self, session_id: str, *, risk_levels: Optional[list[str]] = None,
                  rule_ids: Optional[list[str]] = None, segment: Optional[str] = None,
                  segment_values: Optional[list[str]] = None, sort_by: str = "probability", limit: int = 100) -> dict:
        ctx = self._context(session_id)
        self._require_ready(ctx)
        levels = risk_levels or ["High"]
        invalid_levels = set(levels) - set(LEVELS)
        if invalid_levels:
            raise ApiError(422, f"Unknown risk level(s): {', '.join(sorted(invalid_levels))}.")
        mask = ctx.frame["risk_level"].isin(levels).to_numpy(copy=True)
        if segment:
            if segment not in ctx.raw.columns:
                raise ApiError(422, f"Unknown segment column: {segment}.")
            if segment_values:
                mask &= ctx.raw[segment].astype(str).isin(segment_values).to_numpy()
        positions = np.flatnonzero(mask)
        positions = positions[np.argsort(-ctx.probabilities[positions], kind="stable")]
        work = positions[:ctx.action_cap]
        actions, action_error = self._actions(ctx, work)
        if actions is None:
            raise ApiError(503, action_error or "Could not explain the active model.")
        keep = np.ones(len(work), dtype=bool)
        if rule_ids:
            keep &= np.array([bool(set(rule_ids) & set(ids)) for ids in actions["action_ids"]])
        selected = work[keep]
        selected_actions = actions.loc[selected]
        if sort_by == "rules":
            order = np.lexsort((-ctx.probabilities[selected], -selected_actions["n_matched"].to_numpy()))
            selected, selected_actions = selected[order], selected_actions.iloc[order]
        elif sort_by != "probability":
            raise ApiError(422, "sort_by must be probability or rules.")
        selected, selected_actions = selected[:limit], selected_actions.iloc[:limit]
        action_text = {rule.id: rule.action for rule in ctx.rules}
        counts = {rule.id: 0 for rule in ctx.rules}
        rows = []
        for pos, (_, action) in zip(selected, selected_actions.iterrows()):
            ids = list(action["action_ids"])
            for rule_id in ids:
                if rule_id in counts:
                    counts[rule_id] += 1
            rows.append({
                "customer_id": str(ctx.frame.iloc[pos]["customer_id"]),
                "probability": float(ctx.probabilities[pos]),
                "risk_level": str(ctx.frame.iloc[pos]["risk_level"]),
                "top_factors": list(action["top_factors"]),
                "action_ids": ids,
                "actions": list(action["actions"]),
            })
        no_rule = sum(int(value["n_matched"] == 0) for _, value in selected_actions.iterrows())
        return {
            "rows": rows,
            "total_matching": int(len(positions)),
            "shown": len(rows),
            "expected_churn": float(sum(row["probability"] for row in rows)),
            "rule_counts": [{"rule_id": key, "action": action_text.get(key, key), "customers": value} for key, value in counts.items()],
            "no_rule_count": int(no_rule),
            "action_warning": action_error or (f"Rule evaluation was limited to the {len(work):,} highest-risk matching customers." if len(positions) > len(work) else ""),
            "rules": [{"id": rule.id, "action": rule.action, "priority": rule.priority} for rule in ctx.rules],
            "segment_columns": ctx.pkg.bundle.spec.categorical,
            "segment_options": {
                column: sorted(ctx.raw[column].dropna().astype(str).unique().tolist())
                for column in ctx.pkg.bundle.spec.categorical
            },
        }

    def model_insights(self, session_id: str, fraction: float = 0.1) -> dict:
        ctx = self._context(session_id)
        if ctx.pkg is None:
            raise ApiError(409, "No model evaluation is available. Provide labeled data and train a model first.")
        ev = ctx.pkg.evaluation
        if ev is None:
            raise ApiError(409, "The saved model does not include held-out evaluation results.")
        if not 0.01 <= fraction <= 0.5:
            raise ApiError(422, "fraction must be between 0.01 and 0.5.")
        chosen = ctx.pkg.bundle.model_name
        p = ev.test_proba[chosen]
        imp, imp_error = self._importance(ctx)
        roc = W.roc_points(ev.y_test, ev.test_proba)
        calibration = calibration_table(ev.y_test, p)
        confusion = W.confusion_at(ev.y_test, p, ctx.bands.high_min)
        bands = W.band_lift(ev.y_test, p, ctx.bands)
        top_fraction = W.top_fraction_check(ev.y_test, p, fraction)
        return {
            "selected_model": chosen,
            "selection_metric": ev.selection_metric,
            "trained_at": ev.trained_at,
            "train_source": ev.train_source,
            "split": ev.split,
            "balance": ev.balance,
            "metrics": [{"model": str(model), **{str(k): _json_value(v) for k, v in row.items()}} for model, row in ev.metrics.iterrows()],
            "roc": _records(roc),
            "calibration": _records(calibration),
            "confusion": {**confusion, "threshold": ctx.bands.high_min},
            "bands": _records(bands),
            "top_fraction": top_fraction,
            "fraction": fraction,
            "importance": [{"feature": str(name), "label": W.pretty(name), "value": float(value)} for name, value in imp.items()],
            "warnings": list(ev.warnings) + ([imp_error] if imp_error else []),
            "leakage_flags": [{"feature": name, "auc": float(auc)} for name, auc in ev.leakage],
            "excluded": ev.excluded,
        }

    def workspace_status(self, session_id: str) -> dict:
        ctx = self._context(session_id)
        session = self._session(session_id)
        return {
            "train_exists": self.train_path.exists(),
            "test_exists": self.test_path.exists(),
            "model_exists": self.model_path.exists(),
            "labeled_rows": len(ctx.labeled) if ctx.labeled is not None else 0,
            "schema": ctx.schema.to_dict() if ctx.schema else None,
            "target_counts": {
                str(value): {str(row["value"]): int(row["count"]) for _, row in target_value_counts(ctx.labeled, value).iterrows()}
                for value in [ctx.schema.target] if ctx.labeled is not None and ctx.schema is not None
            },
            "model_name": ctx.pkg.bundle.model_name if ctx.pkg else None,
            "trained_at": ctx.pkg.bundle.trained_at if ctx.pkg else None,
            "saved_model_name": self.model_path.name,
            "portfolio_name": ctx.portfolio_name,
            "portfolio_rows": len(ctx.raw) if ctx.raw is not None else 0,
            "stale_model": ctx.stale,
            "rules": [{"id": r.id, "action": r.action, "risk_levels": r.risk_levels, "driver": r.driver,
                       "conditions": r.conditions, "priority": r.priority} for r in ctx.rules],
            "rule_warnings": ctx.rule_problems,
            "errors": ctx.errors,
            "notes": ctx.notes,
            "session_has_train_upload": session.train_upload is not None,
        }

    def store_train_upload(self, session_id: str, filename: str, digest: str, frame: pd.DataFrame) -> dict:
        session = self._session(session_id)
        session.train_upload, session.train_name, session.train_digest = frame, filename, digest
        return self._upload_summary(filename, digest, frame)

    def train(self, session_id: str, request: TrainRequest) -> dict:
        self.initialize()
        session = self._session(session_id)
        if session.train_upload is not None:
            frame, name = session.train_upload, session.train_name
            source = {"name": f"upload:{name}", "size": len(frame), "mtime_ns": None}
        elif self.train_df is not None:
            frame, name = self.train_df, "data/train.csv"
            source = W.file_fingerprint(self.train_path)
        else:
            raise ApiError(409, "No labeled data is available. Upload a labeled file first.")
        schema = Schema(
            target=request.target,
            positive_labels=request.positive_labels,
            feature_columns=request.feature_columns,
            id_column=request.id_column,
        )
        errors, warnings = validate_schema(frame, schema)
        if errors:
            raise ApiError(422, " ".join(errors))
        try:
            pkg = W.train_package(
                frame,
                schema,
                source,
                test_size=request.test_size,
                seed=request.seed,
                balance=request.balance,
                model_name=request.model_name,
            )
        except ValueError as exc:
            raise ApiError(422, f"Cannot train: {exc}") from exc
        persist = request.persist and ALLOW_PERSIST
        if persist:
            try:
                W.save_package(self.model_path, pkg)
            except OSError as exc:
                raise ApiError(500, f"Model could not be saved: {exc}") from exc
            self.pkg = pkg
            self.train_df = frame if name == "data/train.csv" else self.train_df
            session.pkg_override = None
        else:
            session.pkg_override = pkg
        session.labeled_override = frame
        session.schema_override = schema
        session.labeled_name = name
        session.portfolio_override = None
        session.portfolio_name = ""
        session.portfolio_digest = ""
        session.bands = None
        with self._cache_lock:
            self._score_cache.clear()
            self._derived_cache.clear()
        return {
            "model_name": pkg.bundle.model_name,
            "trained_at": pkg.bundle.trained_at,
            "saved": request.persist,
            "metrics": [{"model": str(model), **{str(k): _json_value(v) for k, v in row.items()}} for model, row in pkg.evaluation.metrics.iterrows()],
            "split": pkg.evaluation.split,
            "warnings": warnings + pkg.evaluation.warnings,
        }

    def score_upload(self, session_id: str, filename: str, digest: str, frame: pd.DataFrame) -> dict:
        ctx = self._context(session_id)
        if ctx.pkg is None:
            raise ApiError(409, "Train or load a model before scoring a file.")
        checks = W.input_checks(ctx.pkg.bundle, frame)
        if checks["missing_columns"]:
            raise ApiError(422, "Missing required model columns: " + ", ".join(checks["missing_columns"]))
        session = self._session(session_id)
        session.portfolio_override = frame.reset_index(drop=True)
        session.portfolio_name = filename
        session.portfolio_digest = f"upload:{digest}"
        result = self._context(session_id)
        return {
            "filename": filename,
            "rows": len(frame),
            "checks": checks,
            "risk_bands": _records(band_summary(result.probabilities, result.bands)),
            "mean_probability": float(np.mean(result.probabilities)),
            "sample": _records(result.frame.head(100)),
        }

    def score_csv(self, session_id: str) -> str:
        ctx = self._context(session_id)
        self._require_ready(ctx)
        return ctx.frame.to_csv(index=False)

    def set_bands(self, session_id: str, low_max: float, high_min: float) -> dict:
        try:
            bands = RiskBands(low_max, high_min)
        except ValueError as exc:
            raise ApiError(422, str(exc)) from exc
        self._session(session_id).bands = (bands.low_max, bands.high_min)
        with self._cache_lock:
            self._score_cache.clear()
        return {"low_max": bands.low_max, "high_min": bands.high_min}

    def reset(self, session_id: str) -> dict:
        with self._session_lock:
            self._sessions.pop(session_id, None)
        with self._cache_lock:
            self._score_cache.clear()
        return self.bootstrap(session_id)

    @staticmethod
    def _upload_summary(filename: str, digest: str, frame: pd.DataFrame) -> dict:
        return {
            "filename": filename,
            "digest": digest,
            "rows": int(len(frame)),
            "columns": [str(c) for c in frame.columns],
            "preview": _records(frame.head(12)),
            "target_counts": {
                str(column): {str(k): int(v) for k, v in frame[column].value_counts(dropna=False).items()}
                for column in frame.columns if frame[column].nunique(dropna=True) <= 20
            },
        }
