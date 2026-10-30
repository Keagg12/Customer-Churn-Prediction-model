"""Overview: observed history on one side, model predictions on the other, kept visibly apart."""
import numpy as np
import pandas as pd
import streamlit as st

from churn import workspace as W

from . import theme
from .common import MODEL_NOTE, setup_state
from .state import Ctx, go


def _observed_cell(ctx: Ctx) -> str:
    if ctx.base_rate is None:
        return theme.big_number("Observed · historical", "n/a", "No labeled history is loaded.", "3rem", theme.MUTED)
    return theme.big_number(f"Observed · {ctx.labeled_name}", f"{ctx.base_rate:.1%}",
                            f"churn rate across {ctx.n_labeled:,} labeled customers ({ctx.n_churned:,} churned).", "4.4rem")


def _predicted_cells(ctx: Ctx) -> list:
    if not ctx.ready:
        return [theme.big_number("Predicted · model scores", "not yet", "A trained model and customers to score are needed.", "3rem", theme.MUTED)]
    f, hi = ctx.frame, ctx.bands.high_min
    n_high = int((f["risk_level"] == "High").sum())
    avg = float(f["churn_probability"].mean())
    return [
        theme.big_number(f"Predicted · high risk (≥ {hi:.0%})", f"{n_high:,}",
                         f"{n_high / max(len(f), 1):.0%} of the {len(f):,} customers scored in {ctx.portfolio.name}.", "3.5rem", theme.RISK["High"]),
        theme.big_number("Predicted · average probability", f"{avg:.1%}", "mean model score across the same customers.", "2.5rem"),
    ]


def _chart_controls(ctx: Ctx):
    s = ctx.strength
    feats = list(s["feature"].head(6)) if s is not None and len(s) else []
    if not feats:
        return None
    return st.radio("Compare observed churn by", feats, horizontal=True, key="ov_feature", format_func=W.pretty)


def render(ctx: Ctx) -> None:
    if ctx.labeled is None and not ctx.ready:
        theme.md(st, theme.header("Overview", "Customer churn overview", "Nothing is loaded yet."))
        setup_state(ctx)
        return

    lede = ("Observed churn is what happened to labeled historical customers. Predicted risk is the model's estimate for "
            "customers whose outcome is not known. The two are shown separately.")
    theme.md(st, theme.header("Overview", "Customer churn, observed and predicted", lede))

    cells = [_observed_cell(ctx)] + _predicted_cells(ctx)
    template = {1: "1fr", 2: "5fr 6fr", 3: "5fr 4fr 3fr"}[len(cells)]
    theme.md(st, theme.ledger(cells, template))
    if ctx.ready:
        theme.md(st, theme.band_bar(ctx.summary()))
        if ctx.portfolio.in_sample:
            theme.md(st, theme.callout("data/test.csv was not found, so the training rows are being scored. Those scores are "
                                       "in-sample and look better than they would on new customers.", "warn"))
        if ctx.stale:
            theme.md(st, theme.callout("data/train.csv has changed since the model was trained. Retrain in the Prediction Workspace "
                                       "to include the change.", "warn"))
    for e in ctx.errors:
        theme.md(st, theme.callout(e, "err"))
    for n in ctx.notes:
        theme.md(st, theme.callout(n, "warn"))
    if not ctx.ready:
        setup_state(ctx, "Predictions are not available yet")

    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)
    main, side = st.columns([7, 4], gap="large")

    with main:
        theme.md(st, theme.section("Where churn concentrates",
                                   "Observed churn rate among labeled historical customers, by group. The dashed line is the overall rate."))
        feat = _chart_controls(ctx)
        if feat is None or ctx.patterns is None or ctx.base_rate is None:
            theme.md(st, theme.callout("Patterns need labeled historical data. Load it in the Prediction Workspace."))
        else:
            g = ctx.patterns[ctx.patterns["feature"] == feat].copy()
            big = g["customers"] >= 0.02 * g["customers"].sum()
            g["top"] = False
            if big.any():
                g.loc[g.loc[big, "churn_rate"].idxmax(), "top"] = True
            theme.vega(st, g[["group", "order", "customers", "churn_rate", "top"]], theme.spec_group_rates(ctx.base_rate))
            kind = g["kind"].iloc[0] if len(g) else "category"
            how = "Numeric inputs are split into fifths of customers." if kind == "numeric" else "One bar per category."
            theme.md(st, f"<div class='ci-sub'>{theme.esc(how)} Rates describe the historical data and are not proof that "
                         "this input causes churn.</div>")
        if ctx.ready:
            st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
            theme.md(st, theme.section("Predicted risk across customers",
                                       f"How the model's scores spread over {len(ctx.frame):,} customers in {ctx.portfolio.name}. "
                                       "Colours are the risk bands."))
            theme.vega(st, W.probability_histogram(ctx.frame["churn_probability"], ctx.bands), theme.spec_risk_histogram())
            theme.md(st, f"<div class='ci-sub'>{theme.esc(MODEL_NOTE)}</div>")

    with side:
        theme.md(st, theme.section("What the history shows", "Largest differences in observed churn between groups."))
        if ctx.strength is not None and ctx.base_rate is not None and len(ctx.strength):
            sentences = W.insight_sentences(ctx.strength, ctx.base_rate, k=4)
            names = list(ctx.strength["feature"].head(4))
            theme.md(st, theme.insight_list(zip([W.pretty(n) for n in names], sentences)))
        else:
            theme.md(st, theme.callout("No historical patterns available."))
        if ctx.ready:
            st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)
            theme.md(st, theme.section("Needs attention first", "Highest predicted churn probability right now."))
            top_n = 8
            order = np.argsort(-ctx.frame["churn_probability"].to_numpy(), kind="stable")[:top_n]
            acts = ctx.actions(order)
            rows = []
            for pos in order:
                why = ""
                if acts is not None and pos in acts.index:
                    why = ", ".join(W.pretty(x) for x in acts.loc[pos, "top_factors"][:2])
                r = ctx.frame.iloc[pos]
                rows.append({"customer": r["customer_id"], "probability": float(r["churn_probability"]),
                             "level": r["risk_level"], "why": ("Main factors: " + why) if why else ""})
            theme.md(st, theme.attention_rows(rows))
            st.button("Open Retention Center", type="primary", on_click=go, args=("Retention Center",), key="ov_open_rc")
