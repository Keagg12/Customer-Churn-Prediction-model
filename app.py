"""Churn Intelligence: Predict, Explain, Recommend, Retain.
Run from the project root:  streamlit run app.py

The app opens on the Overview using the files in data/ and models."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Churn Intelligence", layout="wide", initial_sidebar_state="expanded")

from ui import page_explorer, page_intelligence, page_overview, page_retention, theme  # noqa: E402
from ui.common import source_line  # noqa: E402
from ui.state import PAGES, S, build_ctx  # noqa: E402

ROUTES = {
    "Overview": page_overview.render,
    "Customer Explorer": page_explorer.render,
    "Customer Intelligence": page_intelligence.render,
    "Retention Center": page_retention.render,
}

theme.inject(st)
S.setdefault("nav", PAGES[0])
if S["nav"] not in PAGES:
    S["nav"] = PAGES[0]

with st.sidebar:
    theme.md(st, theme.brand())
    page = st.radio("Navigate", PAGES, key="nav", label_visibility="collapsed")

ctx = build_ctx()

with st.sidebar:
    line = source_line(ctx)
    note = (f"{theme.esc(line)}<br>" if line else "No customers scored yet.<br>")
    note += "Observed churn comes from labeled history. Predicted risk is a model estimate."
    theme.md(st, f"<div class='ci-side-note'>{note}</div>")

ROUTES[page](ctx)
