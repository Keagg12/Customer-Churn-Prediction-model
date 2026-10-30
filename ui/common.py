"""Pieces shared by several pages: the setup state and the standing disclaimers."""
import pandas as pd
import streamlit as st

from . import theme
from .state import DATA_DIR, MODEL_PATH, RULES_PATH, TEST_CSV, TRAIN_CSV, Ctx, go

RULES_NOTE = ("Recommended actions come from the business rules in config/rules.json. The triggers use thresholds "
              "taken from the data; the actions themselves are judgement, and their effect on churn has not been tested.")
MODEL_NOTE = ("The model explanation shows what the model relies on for this score. It describes the model, "
              "not proven causes of churn.")


def _check(ok: bool, text: str, detail: str = "") -> str:
    mark = "Found" if ok else "Missing"
    color = theme.RISK["Low"] if ok else theme.RISK["High"]
    d = f"<div class='ci-sub'>{theme.esc(detail)}</div>" if detail else ""
    return (f"<div class='ci-status'><span class='ci-label' style='color:{color}'>{mark}</span>"
            f"<div>{theme.esc(text)}{d}</div><span></span></div>")


def setup_state(ctx: Ctx, title: str = "Setup needed") -> None:
    """What is missing and how to fix it. Shown instead of any customer data when the app cannot score yet."""
    rows = [
        _check(TRAIN_CSV.exists(), "data/train.csv", "Labeled history used to train the model and to report observed churn."),
        _check(ctx.schema is not None or not TRAIN_CSV.exists(), "config/dataset.json matches the training columns",
               "Names the churn column, the value that means churned, and the ID column."),
        _check(TEST_CSV.exists(), "data/test.csv", "Customers to score. Optional: without it the training rows are scored and marked in-sample."),
        _check(ctx.pkg is not None, "A trained model", f"Saved to {MODEL_PATH.relative_to(MODEL_PATH.parent.parent)} after the first training."),
        _check(RULES_PATH.exists(), "config/rules.json", "Optional: without it no actions are recommended."),
    ]
    theme.md(st, f"<div class='ci-panel'>{theme.section(title, 'The dashboard shows real customers only. It never fills in sample records.')}{''.join(rows)}</div>")
    for e in ctx.errors:
        theme.md(st, theme.callout(e, "err"))
    for n in ctx.notes:
        theme.md(st, theme.callout(n, "warn"))
    theme.md(st, theme.callout(f"Put train.csv, test.csv and data_descriptions.csv in {DATA_DIR.name}/ next to app.py and reload, "
                               "or load files yourself in the Prediction Workspace."))
    st.button("Open Prediction Workspace", type="primary", on_click=go, args=("Prediction Workspace",), key="setup_open_ws")


def require_ready(ctx: Ctx) -> bool:
    """True when customers are scored. Otherwise draws the setup state and returns False."""
    if ctx.ready:
        return True
    setup_state(ctx)
    return False


def source_line(ctx: Ctx) -> str:
    if not ctx.ready:
        return ""
    p = ctx.portfolio
    return f"{p.name} · {len(p.raw):,} customers · model: {ctx.pkg.bundle.model_name}"


def selectable_table(tbl: pd.DataFrame, key: str, config: dict, height: int = 520):
    """Show a table; return the selected row index (position in tbl) or None. Falls back on older Streamlit."""
    try:
        ev = theme.stretch(st.dataframe, tbl, on_select="rerun", selection_mode="single-row", key=key,
                           hide_index=True, column_config=config, height=height)
        rows = list(ev.selection.rows)
        return rows[0] if rows else None
    except Exception:
        theme.stretch(st.dataframe, tbl, hide_index=True, column_config=config, height=height)
        st.caption("Row selection needs a newer Streamlit. Pick a customer below instead.")
        ids = list(tbl["Customer"].head(100))
        pick = st.selectbox("Customer", ids, index=None, placeholder="Choose a customer", key=key + "_fallback")
        return ids.index(pick) if pick else None


def table_config() -> dict:
    cc = st.column_config
    return {"Churn probability": cc.ProgressColumn("Churn probability", min_value=0, max_value=100, format="%.1f%%"),
            "Suggested next action": cc.TextColumn("Suggested next action", width="large")}
