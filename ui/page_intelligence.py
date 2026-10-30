"""Customer Intelligence: one customer, the model's explanation, and (separately) the rule-based playbook."""
import numpy as np
import pandas as pd
import streamlit as st

from churn import workspace as W
from churn.explain import ExplainError, explanation_unit, top_drivers
from churn.recommend import recommend

from . import theme
from .common import MODEL_NOTE, RULES_NOTE, require_ready
from .state import S, Ctx, find_customer, go


def _fmt(v) -> str:
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return "—"
    if isinstance(v, (float, np.floating)):
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return str(v)


def _typical(series: pd.Series, numeric: bool) -> str:
    if numeric:
        return _fmt(pd.to_numeric(series, errors="coerce").median())
    mode = series.dropna().astype(str).mode()
    return str(mode.iloc[0]) if len(mode) else "—"


def _pick_changed() -> None:
    S["ci_id_input"] = S.get("ci_pick") or ""


def render(ctx: Ctx) -> None:
    if not require_ready(ctx):
        theme.md(st, theme.header("Customer Intelligence", "Customer profile"))
        return
    f, raw, bundle = ctx.frame, ctx.portfolio.raw, ctx.pkg.bundle

    sel = S.get("selected_id", "")
    if S.get("_ci_sync") != sel:
        S["ci_id_input"], S["_ci_sync"] = sel, sel
    top = np.argsort(-f["churn_probability"].to_numpy(), kind="stable")[:100]
    labels = {f["customer_id"].iloc[i]: f"{f['customer_id'].iloc[i]}  ·  {f['churn_probability'].iloc[i]:.1%}" for i in top}

    typed = st.session_state.get("ci_id_input", "")
    pos = find_customer(ctx, typed)
    shown_default = False
    if pos is None and not typed.strip():
        pos, shown_default = int(top[0]), True
    title = f"Customer {f['customer_id'].iloc[pos]}" if pos is not None else "Customer profile"
    theme.md(st, theme.header("Customer Intelligence", title,
                              "The model's score and explanation for one customer, followed by the business rules that apply."))

    a, b = st.columns([2, 3])
    a.text_input("Customer ID", key="ci_id_input", placeholder="Type an exact customer ID")
    b.selectbox("or choose from the 100 highest-risk customers", list(labels), index=None, key="ci_pick",
                format_func=lambda x: labels[x], placeholder="Highest predicted risk first", on_change=_pick_changed)
    if pos is None:
        theme.md(st, theme.callout(f"No customer with ID “{typed}” in {ctx.portfolio.name}.", "warn"))
        return
    if shown_default:
        theme.md(st, theme.callout("Showing the highest-risk customer. Enter an ID or choose another to look someone else up."))
    else:
        S["selected_id"] = f["customer_id"].iloc[pos]
        S["_ci_sync"] = S["selected_id"]

    prob = float(f["churn_probability"].iloc[pos])
    level = f["risk_level"].iloc[pos]
    rank_pct = float((f["churn_probability"].to_numpy() < prob).mean())
    avg = float(f["churn_probability"].mean())
    row = raw.iloc[pos]

    left, right = st.columns([7, 5], gap="large")
    with left:
        theme.md(st, f"<div class='ci-panel'><div class='ci-label'>Predicted churn probability · model estimate</div>"
                     f"<div class='ci-big' style='font-size:4.6rem'>{prob:.1%}</div>{theme.pill(level)}"
                     f"<div class='ci-sub' style='margin-top:0.7rem'>Scored by {theme.esc(bundle.model_name)}. The average across "
                     f"{len(f):,} customers is {avg:.1%}; this customer scores higher than {rank_pct:.0%} of them.</div>"
                     f"{theme.gauge(prob, ctx.bands.low_max, ctx.bands.high_min)}</div>")

        st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)
        theme.md(st, theme.section("Customer attributes", "The inputs the model uses for this customer."))
        theme.md(st, theme.kv_grid([(W.pretty(c), _fmt(row.get(c))) for c in bundle.spec.used]))

        st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
        theme.md(st, theme.section("Why the model scored this customer this way", MODEL_NOTE))
        contrib = None
        try:
            contrib = W.explain_subset(bundle, raw.iloc[[pos]]).iloc[0]
        except ExplainError as exc:
            theme.md(st, theme.callout(str(exc), "err"))
        if contrib is not None:
            unit = explanation_unit(bundle.pipeline.named_steps["clf"])
            shown = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)
            chart = pd.DataFrame({"name": [W.pretty(i) for i in shown.index], "value": shown.to_numpy(), "order": range(len(shown))})
            theme.vega(st, chart, theme.spec_bars_horizontal(f"Effect on this score ({unit}); red raises risk, green lowers it",
                                                             ".2f", 300, signed=True))
            drivers = top_drivers(contrib, 6)
            if drivers:
                rows = []
                for name, val in drivers:
                    numeric = name in bundle.spec.numeric
                    rows.append({"Input": W.pretty(name), "This customer": _fmt(row.get(name)),
                                 "Typical customer": _typical(raw[name], numeric), "Raises risk by": f"{val:.3f}"})
                theme.md(st, theme.html_table(pd.DataFrame(rows).set_index("Input")))
                theme.md(st, "<div class='ci-sub' style='margin-top:0.5rem'>Typical is the median (numbers) or most common value "
                             "(categories) among the scored customers.</div>")
            else:
                theme.md(st, theme.callout("No input raises this customer's risk above the baseline."))

    with right:
        theme.md(st, theme.section("Retention playbook", "Business rules that apply to this customer. This is separate from the model."))
        if contrib is None:
            theme.md(st, theme.callout("Rules depend on the model's top factors, which could not be computed for this model."))
        else:
            rec = recommend(ctx.rules, level, top_drivers(contrib, 5), row)
            if rec["actions"]:
                theme.md(st, "".join(theme.action_card(a["action"], a["because"]) for a in rec["actions"]))
            else:
                theme.md(st, theme.callout(rec["summary"]))
            theme.md(st, f"<div class='ci-sub'>{theme.esc(rec['summary'])}</div>")
        for m in ctx.rule_problems:
            theme.md(st, theme.callout(m, "warn"))
        theme.md(st, theme.callout(RULES_NOTE))
        st.button("Back to Customer Explorer", on_click=go, args=("Customer Explorer",), key="ci_back")
