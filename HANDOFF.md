# HANDOFF (checkpoint 31 Oct 2026, after frontend navigation change). Must stand alone.

## For the next Claude
Read `CLAUDE.md`, then this file, then `CONTINUATION_PROMPT.md`, then inspect every attached file. Attached code is the truth for what exists. Do not restart or redo finished work. You have no access to earlier chats or the user's machine. Branding stays neutral: no college, team or invented company.

## Project
Standalone Explainable AI-Based Customer Churn Prediction and Personalized Marketing Recommendation application (Predict, Explain, Recommend, Retain). Application only; old PPT/synopsis are reference, not deliverables.

## Architecture
- `churn/` is the backend and contains no Streamlit. `data.py` loads tables. `schema.py` validates the user's column choices. `preprocess.py` builds feature specs and the sklearn preprocessor. `models.py` trains and evaluates four models. `scoring.py` holds risk bands, the model Bundle and scoring. `explain.py` gives per-customer contributions (exact for Logistic Regression, SHAP for others). `recommend.py` evaluates rules. `batch.py` (new) does top factors and rule matching for many customers at once. `workspace.py` (new) holds the dashboard logic: config reading, model package (bundle plus held-out evaluation) train/save/load, input checks, historical patterns, ROC/confusion/band-lift/importance helpers.
- `ui/` is the Streamlit layer. `theme.py` is the design system (CSS, HTML builders, Vega-Lite specs). `state.py` owns session state, caching and `build_ctx()`. The live route registry has Overview, Customer Explorer, Customer Intelligence, and Retention Center. `page_model.py` and `page_workspace.py` remain in the repo but are not imported/routed by `app.py`.
- `app.py` sets the page config, injects the theme, draws the sidebar navigation, calls `build_ctx()` and renders the chosen page.
- Data flow: `data/train.csv` (labeled) gives schema (via `config/dataset.json`), observed patterns and the model. `data/test.csv` (unlabeled) is only scored. With no `test.csv` the training rows are scored and marked in-sample. With no data the UI shows a setup state, never sample records.
- Model package: `models/model.joblib` stores `{"bundle": Bundle, "evaluation": Evaluation}`. `churn.scoring.load_bundle` still reads it. A file without an evaluation (written by `churn.cli`) is rejected by `workspace.load_package` and the app trains its own.
- Uploads live in `st.session_state` only (`labeled_override`, `portfolio_override`, `pkg_override`, `bands`). The saved model changes only when the user ticks "Save to models/model.joblib" (default on for `data/train.csv`, off for uploads).

## Dataset decisions and modeling constraints (verified 31 Oct 2026)
- `data/train.csv`: 243,787 rows x 21 columns, 18 business features + `CustomerID` + `Churn`. Churn counts: 44,182 positive and 199,605 negative (18.1232%). No missing cells or duplicate rows; IDs are unique.
- `data/test.csv`: 104,480 rows x 20 columns, no churn label, no missing/duplicate rows, and no overlap with training IDs. It is only scored, never evaluated.
- `TotalCharges` equals `AccountAge * MonthlyCharges` within 1e-12 on every training row, and is excluded by `config/dataset.json`.
- The data has one snapshot and no date field. Provenance is not documented. A random stratified holdout is possible; a time-based evaluation is not.
- Stored held-out evaluation is on 48,758 rows from `train.csv`; Logistic Regression selected by five-fold CV average precision. Exact candidate metrics and the confusion matrix are in `TECHNICAL_AUDIT.md`.
- Class balancing is off in the stored run. Risk bands are Low <15%, Medium 15% to <30%, High >=30%; these are operational defaults, not established optimal cut-offs.
- Six rule-based retention suggestions use input thresholds. Actions are business judgment and have not been evaluated by an experiment.

## Completed in this session
- Audit of all uploaded files: present, consistent, `churn.original` / `config.original` / `tests.original` are byte-identical to `cli.py` / `rules.json` / `test_core.py`. Original 28 tests pass in the sandbox.
- Backend additions: `churn/batch.py`, `churn/workspace.py`; 14 new tests in `tests/test_workspace.py`.
- Dashboard with four frontend pages: Overview, Customer Explorer, Customer Intelligence, and Retention Center. Model Insights and Prediction Workspace are no longer routed; their modules remain in the repository.
- Complete implementation audit and measured results recorded in `TECHNICAL_AUDIT.md`; `README.md` now matches the active page registry; stale dataset status in `CLAUDE.md` updated.
- Removed 23 ignored/generated Python bytecode files from three `__pycache__` directories. Tracked legacy data/archive/backups were retained.
- Design: Instrument Serif / IBM Plex Sans / IBM Plex Mono, palette #F5F3EE #FFFFFF #20231F #73766F, brick red #B64B36 for primary actions and selected states, semantic green/amber/red for risk only. Asymmetric ledger/column layouts, thin borders, no gradients, shadows, glass, emoji or icon cards.

## Tests actually executed (Claude's sandbox: Python 3.13, scikit-learn 1.9.1, pandas 3.0.5, no Streamlit, no SHAP)
- `python -m unittest discover tests`: 42 tests, all passed (28 original + 14 new). The fixtures are synthetic; no result is a project result.
- Stand-in smoke run (a hand-written fake `streamlit` module that records calls; scratch scripts, NOT shipped). Scenarios on synthetic data shaped like the real columns, all completed without exceptions:
  - empty `data/`: every page shows the setup state;
  - full: first launch trains and saves a model, all six pages render, second launch reuses the saved model;
  - interactions: Explorer filters and row selection, open profile, Intelligence with unknown/typed/selected ID, Retention rule filter and sort, Model Insights, Workspace upload-and-retrain, upload-and-score, apply bands, reset;
  - stale training file notice, wrong `dataset.json`, invalid `rules.json`.
- Independent number check: with the saved model, observed churn rate, customer count, High-risk count, average probability and the top-risk customer shown on Overview matched values recomputed straight from the files with the backend. Model Insights used the held-out split labels only. An upload-trained model did not change the saved model file (md5 identical).
- Generated HTML from every page was checked for balanced tags.
- Current workspace check (31 Oct 2026): `python -m unittest discover tests` ran 42 tests and all passed. `python -m streamlit run app.py` started Streamlit at `http://localhost:8502`; the bare `streamlit run app.py` command is not on the active PowerShell PATH. The selected interpreter measured both real CSVs and loaded both model artifacts; detailed output is summarized in `TECHNICAL_AUDIT.md`.

## Failed commands and blockers
- `pip install streamlit shap altair` failed in the sandbox (pypi.org refused by the egress proxy, 403). So the real app was never launched there.

## NOT verified (needs the user's VS Code environment)
- Browser rendering of the pages was not inspected: layout, sidebar CSS, Google Fonts, Vega-Lite rendering, and table selection remain unverified. The server process started with `python -m streamlit run app.py` at `http://localhost:8502`.
- Row selection in tables (`st.dataframe(on_select=...)`, needs Streamlit >= 1.35; there is a selectbox fallback for older versions) and full-width handling across Streamlit versions.
- SHAP code path (non-linear models): not run at all. The real data's selected model is Logistic Regression, which uses the exact linear path (tested).
- Speed on the real data: first-launch training on 243,787 rows (not measured), scoring 104,480 rows, Explorer/Retention responsiveness.
- Scores on `data/test.csv` have no ground-truth labels; quality metrics for that portfolio cannot be computed.

## File manifest (paths relative to repo root)
Status words: Generated = written by Claude, not yet applied by the user. Tested = what is described above.
| Path | Status | Verification |
|---|---|---|
| `app.py` | Tested | unit suite passed; Streamlit launch unavailable in active environment |
| `requirements.txt` | REPLACEMENT (Generated): streamlit>=1.35 | none needed |
| `README.md` | REPLACEMENT (Generated) | none needed |
| `TECHNICAL_AUDIT.md` | Generated | source and dataset audit; tests and stored model artifact inspected |
| `HANDOFF.md` | REPLACEMENT (this file) | none needed |
| `CONTINUATION_PROMPT.md` | NEW (Generated) | none needed |
| `.streamlit/config.toml` | NEW (Generated) | not loaded by a real Streamlit |
| `ui/__init__.py` | NEW | smoke run |
| `ui/theme.py` | NEW | smoke run; HTML tag balance checked; CSS unrendered |
| `ui/state.py` | Tested | unit suite passed; Streamlit launch unavailable in active environment |
| `ui/common.py` | Tested | unit suite passed; Streamlit launch unavailable in active environment |
| `ui/page_overview.py` | Tested | unit suite passed; Streamlit launch unavailable in active environment |
| `ui/page_explorer.py` | NEW | smoke run |
| `ui/page_intelligence.py` | NEW | smoke run |
| `ui/page_model.py` | NEW | smoke run + number check |
| `ui/page_retention.py` | NEW | smoke run |
| `ui/page_workspace.py` | NEW | smoke run (uploads simulated) |
| `churn/batch.py` | NEW | unit tests (matches single-customer functions row by row) |
| `churn/workspace.py` | NEW | unit tests |
| `tests/test_workspace.py` | NEW | 14 tests pass |
| `churn/__init__.py`, `cli.py`, `data.py`, `explain.py`, `models.py`, `preprocess.py`, `recommend.py`, `schema.py`, `scoring.py` | UNCHANGED | 28 original tests pass |
| `config/dataset.json`, `config/rules.json` | UNCHANGED | validated by tests |
| `tests/test_core.py` | UNCHANGED | passes |
| `.gitignore` | UNCHANGED (ignores data/, models/, caches) | n/a |
| `CLAUDE.md` | Generated | dataset status and current risk-band defaults |
| `data/*.csv` | user's local files, never uploaded to Claude | n/a |
| `layout_backup/`, `customer-churn.csv`, `files.zip`, `.kilo/` | not used by current app runtime; retained as tracked backups/data or tooling config because provenance/ownership is unclear | n/a |

## Module dependencies
`app.py` -> `ui.*`. `ui.state` -> `churn.workspace`, `churn.batch`, `churn.data`, `churn.recommend`, `churn.scoring`, `churn.explain`. Every `ui.page_*` -> `ui.theme`, `ui.common`, `ui.state`. `ui.common` -> `ui.state`, `ui.theme`. `churn.workspace` -> `churn.models`, `churn.scoring`, `churn.explain`, `churn.preprocess`, `churn.schema`. `churn.batch` -> `churn.recommend`. `churn/` never imports `ui/`.

## Known issues and open items
- Moderate model quality (AUC about 0.755 vs 18.1% base rate). No time-based split (no dates).
- First launch blocks on training with a spinner; two browser sessions opening at the same moment on a fresh install could both train.
- The frontend no longer exposes uploads, manual retraining, or held-out model evaluation; training still occurs automatically if no usable saved model exists.
- A bundle saved by `python -m churn.cli train` has no evaluation and is not used by the app.
- Recommendation actions are heuristics; the UI says so on every page that shows them.

## Next action
1. If desired, open the already running local server at `http://localhost:8502`; for future launches use `python -m streamlit run app.py` in the selected environment.
2. The user reports browser errors or screenshots of the four active pages. Browser rendering has not been inspected by this audit.
3. The next Claude fixes only what the real run shows (CSS selectors, table selection, chart rendering, speed), updates this file, and re-delivers complete replacement files.
4. Optional later: remove clutter after the user confirms; tune hyperparameters; update the stale text in `CLAUDE.md`.

## Manual steps for the user
The 42-test suite has passed. Launch with `python -m streamlit run app.py`; confirm data files stay local/private and keep `data/` and `models/` out of version control.
