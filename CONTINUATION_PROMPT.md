# Paste this into a fresh Claude conversation

CONTINUE FROM HANDOFF

Current state (31 Oct 2026): the original Streamlit app is retained as a fallback; the primary converted web app is React/Vite in `frontend/` with FastAPI in `backend/`, reusing the existing `churn/` business logic. All six React routes were checked locally with real data. `python -m unittest discover tests` passes 46 tests; `npm --prefix frontend ci` and `npm --prefix frontend run build` pass. Vercel's `vite: Permission denied` was traced to committed Windows `node_modules`/`dist` files with non-executable Git modes. Those generated directories are staged for untracking, only their ignore rules were restored, and npm's esbuild install script is explicitly approved. The remote Vercel redeploy is NOT yet verified; commit/push the staged cleanup and redeploy. See `TECHNICAL_AUDIT.md` and `README.md` for details. Historical Streamlit-only notes below describe the earlier baseline and are superseded by this current state and `HANDOFF.md`.

This is an existing project. Do not restart it and do not regenerate finished files. It is the Churn Intelligence application (Predict, Explain, Recommend, Retain): Python, pandas, scikit-learn, SHAP, Streamlit.

I am attaching these files. Read them in this order: `CLAUDE.md`, `HANDOFF.md` (authoritative), then this prompt, then the code.
- `TECHNICAL_AUDIT.md` (current source-grounded audit)
- `frontend/` and `backend/` (React/FastAPI conversion)
- `CLAUDE.md`, `HANDOFF.md`, `CONTINUATION_PROMPT.md`, `README.md`, `requirements.txt`, `.gitignore`, `app.py`
- `.streamlit/config.toml`
- everything in `ui/` (theme, state, common, six page modules, `__init__.py`)
- everything in `churn/` (including the new `batch.py` and `workspace.py`)
- `config/dataset.json`, `config/rules.json`
- `tests/test_core.py`, `tests/test_workspace.py`
- any error output or screenshots from my local run (the most important input)
I am NOT attaching `train.csv` or `test.csv` (large, local). `data_descriptions.csv` is optional.

Already done, do not repeat: backend audit; `churn/batch.py` and `churn/workspace.py`; the full dashboard (Overview, Customer Explorer, Customer Intelligence, Model Insights, Retention Center, Prediction Workspace); the design system; the app shell. 42 unit tests passed in Claude's sandbox. A stand-in smoke run (fake streamlit module) exercised every page and flow on synthetic data and matched an independent number check.

Not yet verified: real Streamlit rendering and styling, table row selection, chart rendering, the SHAP path for non-linear models, speed and first-launch training time on the real 243,787-row data. Streamlit could not be installed in Claude's sandbox. Nothing has been run on the real CSVs.

Rules that still apply: labeled `train.csv` trains and evaluates; unlabeled `test.csv` is only scored and never evaluated; never hard-code or fabricate metrics or customers; keep class balancing off by default; risk bands Low <15%, Medium 15% to <30%, High >=30%; model explanations and business-rule recommendations stay separate; do not imply the rules are proven to reduce churn; neutral branding only. Design: Instrument Serif, IBM Plex Sans, IBM Plex Mono; #F5F3EE, #FFFFFF, #20231F, #73766F, accent #B64B36; no gradients, glass, emoji or icon-card grids.

Precise next action: reconcile the attached code with `HANDOFF.md`, then work through the problems in my pasted output or screenshots, one at a time, in the order they appear. Deliver complete replacement files (exact path, REPLACEMENT or NEW), update `HANDOFF.md` and this prompt after each meaningful fix, and say plainly what you ran and what you could not run. If I report no problems, ask me for screenshots of each page and the output of `python -m unittest discover tests` before changing anything.

Remaining checks: verify the Vercel redeploy from a commit that removes tracked generated assets; deploy and secure FastAPI separately; configure private persistent data/model storage and CORS; add API authentication before exposing real customer data. Browser checks were performed locally, not against production. The non-linear SHAP path also remains unverified.

Checkpoint commands: `CHECKPOINT AND HANDOFF` means finish the smallest safe operation, then give me the latest complete `HANDOFF.md` and `CONTINUATION_PROMPT.md`, the list of files to carry over, and a short summary.
