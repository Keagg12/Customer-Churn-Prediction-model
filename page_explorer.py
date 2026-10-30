"""Customer Explorer: search, filter, sort, select a customer."""
import numpy as np
import pandas as pd
import streamlit as st

from churn import workspace as W

from . import theme
from .common import require_ready, selectable_table, table_config
from .state import S, Ctx, go

NO_RULE = "No rule matched"


def characteristic_columns(ctx: Ctx, k: int = 5) -> list:
    """Inputs shown beside each customer: the ones the model relies on most (falls back to the first inputs)."""
    imp = ctx.importance()
    cols = [c for c in imp.index if c in ctx.portfolio.raw.columns][:k]
    return cols or list(ctx.pkg.bundle.spec.used[:k])


def action_text(acts, pos: int, level: str) -> str:
    if acts is None or pos not in acts.index:
        return "—"
    t = acts.loc[pos, "next_action"]
    return t if t else NO_RULE


def render(ctx: Ctx) -> None:
    theme.md(st, theme.header("Customer Explorer", "Find and compare customers",
                              "Search by ID, filter by risk or segment, and sort. Select a row to open that customer."))
    if not require_ready(ctx):
        return
    f, raw, spec = ctx.frame, ctx.portfolio.raw, ctx.pkg.bundle.spec
    chars = characteristic_columns(ctx)

    with st.container(border=True):
        a, b, c = st.columns([2, 2, 3])
        query = a.text_input("Customer ID contains", key="ex_query", placeholder="e.g. a few characters of an ID")
        levels = b.multiselect("Risk level", ["High", "Medium", "Low"], default=["High", "Medium", "Low"], key="ex_levels")
        lo, hi = c.slider("Churn probability range (%)", 0, 100, (0, 100), 1, key="ex_range", format="%d%%")
        d, e, g, h = st.columns([2, 3, 2, 1])
        seg_cols = [None] + list(spec.categorical)
        seg_col = d.selectbox("Segment", seg_cols, key="ex_seg", format_func=lambda x: "All customers" if x is None else W.pretty(x))
        seg_vals = []
        if seg_col:
            options = sorted(raw[seg_col].dropna().astype(str).unique())
            seg_vals = e.multiselect(f"{W.pretty(seg_col)} is", options, key="ex_segvals")
        else:
            e.caption("Pick a segment column to filter by one of its values.")
        numeric_sorts = ["Churn probability"] + [W.pretty(c_) for c_ in spec.numeric]
        sort_by = g.selectbox("Sort by", numeric_sorts, key="ex_sort")
        limit = h.selectbox("Rows", [100, 250, 500], key="ex_limit")
        desc = st.radio("Order", ["Highest first", "Lowest first"], horizontal=True, key="ex_order") == "Highest first"

    p = f["churn_probability"].to_numpy()
    mask = f["risk_level"].isin(levels).to_numpy() & (p >= lo / 100.0) & (p <= hi / 100.0 + 1e-9)
    if query.strip():
        mask &= f["customer_id"].str.lower().str.contains(query.strip().lower(), regex=False).to_numpy()
    if seg_col and seg_vals:
        mask &= raw[seg_col].astype(str).isin(seg_vals).to_numpy()
    pos = np.flatnonzero(mask)
    if sort_by == "Churn probability":
        key = p[pos]
    else:
        col = next(c_ for c_ in spec.numeric if W.pretty(c_) == sort_by)
        key = pd.to_numeric(raw[col], errors="coerce").to_numpy()[pos]
    fill = -np.inf if desc else np.inf
    order = np.argsort(np.nan_to_num(key, nan=fill) * (-1 if desc else 1), kind="stable")
    pos = pos[order][:limit]

    total = int(mask.sum())
    theme.md(st, f"<div class='ci-sub' style='margin:0.6rem 0'>{total:,} of {len(f):,} customers match. "
                 f"Showing {len(pos):,}. Predicted risk comes from the model; it is not an observed outcome.</div>")
    if total == 0:
        theme.md(st, theme.callout("No customers match these filters."))
        return

    n_act = min(len(pos), ctx.action_cap)
    acts = ctx.actions(pos[:n_act])
    tbl = pd.DataFrame({"Customer": f["customer_id"].to_numpy()[pos],
                        "Churn probability": np.round(p[pos] * 100, 2),
                        "Risk": f["risk_level"].to_numpy()[pos]})
    for c_ in chars:
        tbl[W.pretty(c_)] = raw[c_].to_numpy()[pos]
    tbl["Suggested next action"] = [action_text(acts, int(q), lv) if i < n_act else "—"
                                    for i, (q, lv) in enumerate(zip(pos, tbl["Risk"]))]
    if n_act < len(pos):
        theme.md(st, theme.callout(f"Suggested actions are worked out for the first {n_act} rows only, because explaining this "
                                   "model type is slower. Narrow the filters to see more.", "warn"))
    for err in ctx.errors:
        theme.md(st, theme.callout(err, "err"))

    sel = selectable_table(tbl, "ex_table", table_config())
    if sel is not None and sel < len(tbl):
        row = tbl.iloc[sel]
        S["selected_id"] = str(row["Customer"])
        left, right = st.columns([5, 2])
        left.markdown(f"<div class='ci-panel'><div class='ci-label'>Selected customer</div>"
                      f"<div style='margin:0.3rem 0'><span class='ci-mono' style='font-size:1.1rem'>{theme.esc(row['Customer'])}</span> "
                      f"&nbsp;{theme.pill(row['Risk'])}&nbsp; <span class='ci-mono'>{row['Churn probability']:.1f}%</span></div>"
                      f"<div class='ci-sub'>Suggested next action: {theme.esc(row['Suggested next action'])}</div></div>",
                      unsafe_allow_html=True)
        right.button("Open customer profile", type="primary", on_click=go, args=("Customer Intelligence",), key="ex_open")
    st.download_button("Download this view (CSV)", tbl.to_csv(index=False), "customer_explorer_view.csv", "text/csv",
                       key="ex_dl")
