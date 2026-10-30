"""Prediction Workspace: optional uploads, validation, explicit retraining, scoring new customers, risk bands.

Uploaded files live in this browser session only. Nothing is written to data/ and the saved model in models/
changes only when the person asks for it."""
import numpy as np
import pandas as pd
import streamlit as st

from churn import data as D
from churn import workspace as W
from churn.schema import Schema, target_value_counts, validate_schema
from churn.scoring import RiskBands

from . import theme
from .state import (DATASET_CFG, MODEL_PATH, S, TRAIN_CSV, Ctx, read_local, reset_session_to_files, train_and_activate)

UPLOAD_TYPES = ["csv", "tsv", "txt", "xlsx", "xls"]


def _read_upload(up, slot: str):
    """Read an uploaded file once per file, not on every rerun. Returns (table, error)."""
    key = (up.name, up.size)
    cached = S.get(slot)
    if cached and cached[0] == key:
        return cached[1], ""
    try:
        df = D.load_table(up, filename=up.name)
    except D.DataLoadError as exc:
        S.pop(slot, None)
        return None, str(exc)
    S[slot] = (key, df)
    return df, ""


def _flash() -> None:
    msg = S.pop("ws_flash", None)
    if msg:
        theme.md(st, theme.callout(msg))


def _status(ctx: Ctx) -> None:
    rows = []

    def add(name, value, detail=""):
        d = f"<div class='ci-sub'>{theme.esc(detail)}</div>" if detail else ""
        rows.append(f"<div class='ci-status'><span class='ci-label'>{theme.esc(name)}</span><div>{theme.esc(value)}{d}</div><span></span></div>")

    if ctx.labeled is not None:
        add("Labeled history", f"{ctx.labeled_name} · {ctx.n_labeled:,} customers · {ctx.base_rate:.1%} churned" if ctx.base_rate is not None
            else f"{ctx.labeled_name} · {len(ctx.labeled):,} rows", "Used for training, held-out evaluation and observed churn patterns.")
    else:
        add("Labeled history", "none loaded", "Put data/train.csv in place or upload a labeled file on the Train tab.")
    if ctx.pkg is not None:
        ev = ctx.pkg.evaluation
        saved = "saved in models/model.joblib" if MODEL_PATH.exists() and not S.get("pkg_override") else "this session only"
        add("Model", f"{ctx.pkg.bundle.model_name} · trained {ev.trained_at[:10]}", f"{saved}; trained on {ev.train_source.get('name', 'unknown')}.")
    else:
        add("Model", "none", "Train one on the Train tab.")
    if ctx.portfolio is not None:
        add("Customers scored", f"{ctx.portfolio.name} · {len(ctx.portfolio.raw):,} customers",
            "Unlabeled customers are only ever scored, never used to measure accuracy." if not ctx.portfolio.in_sample else
            "These are training rows, so the scores are in-sample.")
    else:
        add("Customers scored", "none", "Place data/test.csv, or upload customers on the Score tab.")
    add("Risk bands", f"Low below {ctx.bands.low_max:.0%} · High from {ctx.bands.high_min:.0%}", "Adjust on the Risk bands tab.")
    theme.md(st, f"<div class='ci-panel'>{''.join(rows)}</div>")
    if ctx.stale:
        theme.md(st, theme.callout("data/train.csv has changed since this model was trained. Retrain on the Train tab to include the change.", "warn"))
    for e in ctx.errors:
        theme.md(st, theme.callout(e, "err"))
    for n in ctx.notes:
        theme.md(st, theme.callout(n, "warn"))
    st.button("Forget uploads and return to the files in data/ and models/", on_click=reset_session_to_files, key="ws_reset")


def _schema_controls(df: pd.DataFrame, cfg: dict):
    """Choose target, churn value(s), ID and excluded columns for an uploaded labeled file. Returns a Schema or None."""
    cols = list(df.columns)
    default = W.schema_from_config(df, cfg)
    with st.expander("Columns: which one says who churned", expanded=default is None):
        t_idx = cols.index(cfg["target"]) if cfg.get("target") in cols else None
        target = st.selectbox("Column that says whether the customer churned", cols, index=t_idx, key="ws_target",
                              placeholder="Choose the churn column")
        if not target:
            return None
        vc = target_value_counts(df, target)
        known = list(vc["value"])
        want = [v for v in cfg.get("positive_labels", []) if v in known] if target == cfg.get("target") else []
        pos = st.multiselect("Value(s) that mean the customer churned", known, default=want, key="ws_pos")
        id_opts = [None] + [c for c in cols if c != target]
        id_idx = id_opts.index(cfg["id_column"]) if cfg.get("id_column") in id_opts else 0
        idc = st.selectbox("Customer ID column (optional)", id_opts, index=id_idx, key="ws_id")
        skip = {target, idc} | set(cfg.get("exclude", []))
        feats = st.multiselect("Columns the model may use", [c for c in cols if c != target],
                               default=[c for c in cols if c not in skip], key="ws_feats")
        st.caption("Remove anything only known after a customer leaves, such as a cancellation date or reason. It would make results look better than they are.")
        return Schema(target=target, positive_labels=pos, feature_columns=[c for c in feats if c != idc], id_column=idc)


def _train_tab(ctx: Ctx) -> None:
    cfg = W.load_dataset_config(DATASET_CFG)
    theme.md(st, theme.section("Train a model", "Compares four models on a held-out part of the labeled data, then activates one. Nothing retrains unless you press the button."))
    up = st.file_uploader("Labeled customer file (optional)", type=UPLOAD_TYPES, key="ws_train_up",
                          help="A table with one row per customer and a column saying who churned. Leave empty to use data/train.csv.")
    df, name, source, from_upload = None, "", {}, False
    if up is not None:
        df, err = _read_upload(up, "_ws_train_df")
        if err:
            theme.md(st, theme.callout(err, "err"))
        else:
            name, from_upload, source = up.name, True, {"name": up.name, "size": int(up.size), "mtime_ns": None}
    elif TRAIN_CSV.exists():
        df, err = (ctx.labeled, "") if (ctx.labeled is not None and ctx.labeled_name == "data/train.csv") else (None, "")
        if df is None:
            df, err = read_local(TRAIN_CSV)
        if err:
            theme.md(st, theme.callout(err, "err"))
        elif df is not None:
            name, source = "data/train.csv", W.file_fingerprint(TRAIN_CSV)
    if df is None:
        theme.md(st, theme.callout("No labeled data available. Upload a labeled file above or place train.csv in the data folder."))
        return
    theme.md(st, f"<div class='ci-sub'>{theme.esc(name)} · {len(df):,} rows · {len(df.columns)} columns</div>")

    schema = _schema_controls(df, cfg) if from_upload else W.schema_from_config(df, cfg)
    if schema is None:
        theme.md(st, theme.callout("The column setup does not fit this table. Set it in the Columns panel, or fix config/dataset.json.", "err"))
        return
    errs, warns = validate_schema(df, schema)
    for m in errs:
        theme.md(st, theme.callout(m, "err"))
    for m in warns:
        theme.md(st, theme.callout(m, "warn"))
    if errs:
        return
    if not from_upload:
        theme.md(st, f"<div class='ci-sub'>Target <span class='ci-mono'>{theme.esc(schema.target)}</span>, churned = "
                     f"<span class='ci-mono'>{theme.esc(', '.join(map(str, schema.positive_labels)))}</span>, ID "
                     f"<span class='ci-mono'>{theme.esc(schema.id_column or 'none')}</span>, from config/dataset.json. "
                     f"Excluded: {theme.esc(', '.join(cfg.get('exclude', [])) or 'nothing')}.</div>")

    c1, c2, c3 = st.columns(3)
    test_size = c1.slider("Held out for testing", 0.1, 0.4, 0.2, 0.05, key="ws_test")
    seed = int(c2.number_input("Random seed", value=42, step=1, key="ws_seed"))
    model_choice = c3.selectbox("Model to activate", ["Best by cross-validation (recommended)", "Logistic Regression", "Gradient Boosting",
                                                      "Random Forest", "Decision Tree"], key="ws_model")
    balance = st.checkbox("Balance classes", value=False, key="ws_balance",
                          help="Off by default. Balancing raises every predicted probability well above real churn rates.")
    if balance:
        theme.md(st, theme.callout("Balancing makes the predicted churn probabilities too high. Ranking barely changes. Leave it off unless you know you need it.", "warn"))
    persist = st.checkbox("Save to models/model.joblib (replaces the saved model)", value=not from_upload, key="ws_persist")

    if st.button("Train and compare models", type="primary", key="ws_train_btn"):
        with st.spinner(f"Training four models on {len(df):,} rows. This can take a few minutes."):
            try:
                pkg = train_and_activate(df, schema, source, test_size=test_size, seed=seed, balance=balance, persist=persist,
                                         model_name=None if model_choice.startswith("Best") else model_choice)
            except ValueError as exc:
                theme.md(st, theme.callout(f"Cannot train: {exc}", "err"))
                return
            except Exception as exc:  # report unexpected training problems in the page
                theme.md(st, theme.callout(f"Training failed: {type(exc).__name__}: {exc}", "err"))
                return
        if from_upload:
            S["labeled_override"] = {"df": df, "schema": schema, "name": name, "size": int(up.size)}
        else:
            S.pop("labeled_override", None)
        S.pop("auto_train_failed", None)
        S["ws_flash"] = (f"Trained on {name}. Active model: {pkg.bundle.model_name}. "
                         f"{'Saved to models/model.joblib. ' if persist and not S.get('pkg_override') else 'Active for this session only. '}"
                         "See Model Insights for the held-out results.")
        st.rerun()


def _score_tab(ctx: Ctx) -> None:
    theme.md(st, theme.section("Score customers", "Upload a customer file to score it with the active model, or leave it empty to keep the current file."))
    if ctx.pkg is None:
        theme.md(st, theme.callout("Train or load a model first (Train a model tab)."))
        return
    bundle = ctx.pkg.bundle
    up = st.file_uploader("Customer file to score", type=UPLOAD_TYPES, key="ws_score_up",
                          help="Needs the same input columns the model was trained on. A churn column, if present, is ignored.")
    if up is None:
        theme.md(st, theme.callout("Required input columns: " + ", ".join(W.pretty(c) for c in bundle.spec.used) + "."))
    else:
        raw, err = _read_upload(up, "_ws_score_df")
        if err:
            theme.md(st, theme.callout(err, "err"))
        else:
            chk = W.input_checks(bundle, raw)
            theme.md(st, f"<div class='ci-sub'>{theme.esc(up.name)} · {chk['rows']:,} rows · {len(raw.columns)} columns</div>")
            if chk["missing_columns"]:
                theme.md(st, theme.callout("These required columns are missing: " + ", ".join(chk["missing_columns"]) + ".", "err"))
            else:
                theme.md(st, theme.callout("All required columns are present."))
                if bundle.schema.target in raw.columns:
                    theme.md(st, theme.callout(f"The file has a “{bundle.schema.target}” column. It is ignored: scoring never uses it, "
                                               "and accuracy is not measured on scored customers.", "warn"))
                if chk["missing_values"]:
                    theme.md(st, theme.callout("Empty values will be filled with typical values: " + ", ".join(
                        f"{W.pretty(k)} ({v:,})" for k, v in chk["missing_values"].items()) + ".", "warn"))
                if chk["non_numeric"]:
                    theme.md(st, theme.callout("Non-numeric entries in numeric columns will be treated as empty: " + ", ".join(
                        f"{W.pretty(k)} ({v:,})" for k, v in chk["non_numeric"].items()) + ".", "warn"))
                if chk["unseen_categories"]:
                    theme.md(st, theme.callout("Values the model never saw in training (treated as unknown): " + "; ".join(
                        f"{W.pretty(k)}: {', '.join(v)}" for k, v in chk["unseen_categories"].items()) + ".", "warn"))
                if st.button("Score this file", type="primary", key="ws_score_btn"):
                    S["portfolio_override"] = {"raw": raw.reset_index(drop=True), "name": up.name, "size": int(up.size)}
                    S["ws_flash"] = f"Scored {len(raw):,} customers from {up.name}. All pages now show this file."
                    st.rerun()
    if ctx.portfolio is not None and not ctx.errors:
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        theme.md(st, theme.section("Current scores", f"{ctx.portfolio.name} · {len(ctx.frame):,} customers"))
        theme.md(st, theme.band_bar(ctx.summary()))
        out = ctx.frame.rename(columns={"customer_id": "customer_id"})
        st.download_button("Download all scores (CSV)", out.to_csv(index=False), "churn_scores.csv", "text/csv", key="ws_dl")


def _restore_bands(low: float, high: float) -> None:
    """Button callback: drop the session override and move the sliders back with it."""
    S.pop("bands", None)
    S["ws_low"], S["ws_high"] = float(low), float(high)


def _bands_tab(ctx: Ctx) -> None:
    theme.md(st, theme.section("Risk bands", "Probability cut-offs that turn a score into Low, Medium or High. Set them to match how many customers your team can contact."))
    if ctx.pkg is None:
        theme.md(st, theme.callout("Train or load a model first."))
        return
    c1, c2 = st.columns(2)
    low = c1.slider("Low risk below", 0.05, 0.50, float(ctx.bands.low_max), 0.01, key="ws_low", format="%.2f")
    high = c2.slider("High risk from", 0.10, 0.90, float(ctx.bands.high_min), 0.01, key="ws_high", format="%.2f")
    if low >= high:
        theme.md(st, theme.callout("Low risk must be below the High risk cut-off.", "err"))
        return
    theme.md(st, "<div class='ci-sub'>Defaults are 15% and 30%, chosen from the calibrated scores of the training data. "
                 "Changes apply to every page for this session.</div>")
    if (low, high) != (ctx.bands.low_max, ctx.bands.high_min):
        if st.button("Apply bands", type="primary", key="ws_apply_bands"):
            S["bands"] = (low, high)
            S["ws_flash"] = f"Risk bands set: Low below {low:.0%}, High from {high:.0%}."
            st.rerun()
    if ctx.ready:
        preview = ctx.frame["churn_probability"].to_numpy()
        pv = pd.DataFrame({"risk_level": RiskBands(low, high).assign(preview)})
        counts = pv["risk_level"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
        summ = pd.DataFrame({"risk_level": counts.index, "customers": counts.values, "share": counts.values / max(len(pv), 1)})
        theme.md(st, "<div class='ci-label' style='margin-top:1rem'>Preview with these cut-offs</div>")
        theme.md(st, theme.band_bar(summ))
    d = ctx.pkg.bundle.bands
    st.button("Restore the model's default bands", on_click=_restore_bands, args=(d.low_max, d.high_min), key="ws_default_bands")


def render(ctx: Ctx) -> None:
    theme.md(st, theme.header("Prediction Workspace", "Data, model and scoring",
                              "The dashboard runs on the files in data/ by default. Use this page to load your own data, retrain, or score new customers."))
    _flash()
    tab_status, tab_train, tab_score, tab_bands = st.tabs(["Status", "Train a model", "Score customers", "Risk bands"])
    with tab_status:
        _status(ctx)
    with tab_train:
        _train_tab(ctx)
    with tab_score:
        _score_tab(ctx)
    with tab_bands:
        _bands_tab(ctx)
