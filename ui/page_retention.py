"""Retention Center: a prioritised work queue of at-risk customers with the factors and rules behind each."""
import numpy as np
import pandas as pd
import streamlit as st

from churn import workspace as W

from . import theme
from .common import RULES_NOTE, require_ready, selectable_table, table_config
from .state import S, Ctx, go


def _join(items, limit=None) -> str:
    items = list(items)[:limit] if limit else list(items)
    return "; ".join(items) if items else "—"


def render(ctx: Ctx) -> None:
    theme.md(st, theme.header("Retention Center", "Who to contact first",
                              "At-risk customers in order of predicted churn probability, with the model's main factors and the "
                              "business rules that apply to each."))
    if not require_ready(ctx):
        return
    f, raw = ctx.frame, ctx.portfolio.raw
    p = f["churn_probability"].to_numpy()
    cap = ctx.action_cap
    rule_text = {r.id: r.action for r in ctx.rules}
    rule_ids = [r.id for r in ctx.rules]

    with st.container(border=True):
        a, b, c = st.columns([2, 3, 3])
        levels = a.multiselect("Risk level", ["High", "Medium"], default=["High"], key="rc_levels")
        pick = b.multiselect("Show customers matching rule", rule_ids, key="rc_rules",
                             format_func=lambda i: f"{i}: {rule_text[i]}"[:70])
        seg_cols = [None] + list(ctx.pkg.bundle.spec.categorical)
        seg = c.selectbox("Segment", seg_cols, key="rc_seg", format_func=lambda x: "All customers" if x is None else W.pretty(x))
        d, e, g = st.columns([3, 2, 2])
        seg_vals = []
        if seg:
            seg_vals = d.multiselect(f"{W.pretty(seg)} is", sorted(raw[seg].dropna().astype(str).unique()), key="rc_segvals")
        else:
            d.caption("Pick a segment column to narrow the queue.")
        sort_by = e.selectbox("Sort by", ["Churn probability", "Rules matched"], key="rc_sort")
        queue_n = g.selectbox("Queue size", [50, 100, 250, 500, 1000], index=1, key="rc_n")

    mask = f["risk_level"].isin(levels).to_numpy()
    if seg and seg_vals:
        mask &= raw[seg].astype(str).isin(seg_vals).to_numpy()
    pos_all = np.flatnonzero(mask)
    pos_all = pos_all[np.argsort(-p[pos_all], kind="stable")]
    if len(pos_all) == 0:
        theme.md(st, theme.callout("No customers match these filters."))
        return

    work = pos_all[:cap]
    acts = ctx.actions(work)
    for err in ctx.errors:
        theme.md(st, theme.callout(err, "err"))
    if acts is None:
        return
    if len(pos_all) > len(work):
        theme.md(st, theme.callout(f"Rules are worked out for the {len(work):,} highest-risk of {len(pos_all):,} matching customers, "
                                   "because explaining this model type is slower.", "warn"))

    sub = acts.loc[work]
    keep = np.ones(len(work), dtype=bool)
    if pick:
        keep &= np.array([bool(set(pick) & set(ids)) for ids in sub["action_ids"]])
    pos = work[keep]
    sub = sub.loc[pos]
    if sort_by == "Rules matched":
        order = np.lexsort((-p[pos], -sub["n_matched"].to_numpy()))
        pos, sub = pos[order], sub.iloc[order]
    pos_q, sub_q = pos[:queue_n], sub.iloc[:queue_n]

    left, right = st.columns([8, 4], gap="large")
    with right:
        n_q = len(pos_q)
        expected = float(p[pos_q].sum())
        theme.md(st, f"<div class='ci-panel'><div class='ci-label'>This queue · model estimate</div>"
                     f"<div class='ci-big' style='font-size:3rem'>{n_q:,}</div>"
                     f"<div class='ci-sub'>customers. The model expects about {expected:,.0f} of them to churn "
                     f"(sum of predicted probabilities). That is an estimate, not a count of known outcomes.</div></div>")
        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
        theme.md(st, theme.section("Rules that apply", f"Customers in this queue matching each rule."))
        counts = {rid: 0 for rid in rule_ids}
        for ids in sub_q["action_ids"]:
            for rid in ids:
                counts[rid] += 1
        no_rule = int((sub_q["n_matched"] == 0).sum())
        rows = sorted(((rule_text[r][:46], v) for r, v in counts.items()), key=lambda t: -t[1])
        if no_rule:
            rows.append(("No rule matched", no_rule))
        if rows:
            chart = pd.DataFrame({"name": [r[0] for r in rows], "value": [r[1] for r in rows], "order": range(len(rows))})
            theme.vega(st, chart, theme.spec_rule_counts(max(120, 30 * len(rows))))
        theme.md(st, "<div class='ci-sub'>Each customer shows at most three actions, so counts can exceed this when more rules match.</div>")
        theme.md(st, theme.callout(RULES_NOTE))

    with left:
        theme.md(st, theme.section("Priority queue", f"{len(pos_q):,} customers. Select a row to open the profile."))
        tbl = pd.DataFrame({
            "Customer": f["customer_id"].to_numpy()[pos_q],
            "Churn probability": np.round(p[pos_q] * 100, 2),
            "Risk": f["risk_level"].to_numpy()[pos_q],
            "Main model factors": [_join([W.pretty(x) for x in t], 3) for t in sub_q["top_factors"]],
            "Suggested actions": [_join((rule_text.get(i, i) for i in ids)) for ids in sub_q["action_ids"]],
        })
        cfg = table_config()
        cfg["Suggested actions"] = st.column_config.TextColumn("Suggested actions", width="large")
        sel = selectable_table(tbl, "rc_table", cfg, 560)
        if sel is not None and sel < len(tbl):
            S["selected_id"] = str(tbl.iloc[sel]["Customer"])
            st.button("Open customer profile", type="primary", on_click=go, args=("Customer Intelligence",), key="rc_open")
        st.download_button("Download this queue (CSV)", tbl.to_csv(index=False), "retention_queue.csv", "text/csv", key="rc_dl")
