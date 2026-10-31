# HANDOFF (checkpoint 31 Oct 2026, after React/FastAPI conversion and Vercel fix). Must stand alone.

## For the next Claude
Read `CLAUDE.md`, then this file, then `CONTINUATION_PROMPT.md`, then inspect every attached file. Attached code is the truth for what exists. Do not restart or redo finished work. You have no access to earlier chats or the user's machine. Branding stays neutral: no college, team or invented company.

## Project
Standalone Explainable AI-Based Customer Churn Prediction and Personalized Marketing Recommendation application (Predict, Explain, Recommend, Retain). Application only; old PPT/synopsis are reference, not deliverables.

## Current Web Application
- `frontend/` is the new React/Vite SPA with six routes; `backend/` is the FastAPI adapter. Existing `churn/` logic is reused for training, scoring, explanations, metrics, and retention rules.
- Original `app.py` and `ui/` remain unchanged as a Streamlit fallback; do not remove until the React deployment is verified.
- Vercel failure was caused by tracked Windows-generated `frontend/node_modules`/`dist` files; Git recorded Vite/esbuild executables as mode `100644`. Those directories are now staged for removal from Git; local copies remain. `.gitignore` only re-enables `frontend/node_modules/` and `frontend/dist/`; other user-commented ignore patterns remain commented.
- `frontend/package.json` explicitly allows esbuild's install script for npm 11. `npm --prefix frontend ci` and `npm --prefix frontend run build` pass locally after the fix. The Vercel redeploy is not yet verified; commit/push the staged cleanup and redeploy.
- FastAPI is not hosted on Vercel. Deploy it separately and set Vercel `VITE_API_BASE_URL`; configure backend `CORS_ORIGINS`. The API currently has no authentication and uses in-memory browser sessions, so protect it and provide private/persistent data/model storage before production use.
- `render.yaml` defines a free Render web service from the repository root, pinned Python requirements, `/api/health`, repository-relative CSV/config/model paths, `CHURN_AUTO_TRAIN=false`, and an unsynced `CORS_ORIGINS` value. The existing Render service is currently failing during startup with OOM at its 512 MiB limit.
- Pushed `main` is `54bf44d`. `data/*.csv` and `models/*.joblib` are tracked in that commit, so Render can read them; their visibility depends on GitHub repository visibility. Do not remove or re-ignore them without user approval; use private storage if sensitive.

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
- React/FastAPI conversion added at `frontend/` and `backend/`; legacy Streamlit files are preserved. Feature mapping and deployment steps are in `README.md`.
- Vercel Linux executable-mode failure addressed by untracking only generated frontend directories, restoring those two ignore rules, and approving esbuild's npm install script.
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
- Current workspace check (31 Oct 2026): `python -m unittest discover tests` ran 46 tests and all passed, including four synthetic FastAPI API tests. `npm --prefix frontend ci` and `npm --prefix frontend run build` passed. Playwright locally checked all six React routes, customer filtering/profile flow, desktop/mobile Overview, and API-backed recommendations. Vercel has not yet been redeployed.

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
| `TECHNICAL_AUDIT.md` | Tested | source/data audit plus React conversion and local Vercel-fix verification |
| `backend/` | Tested | FastAPI routes exercised through synthetic TestClient tests; production deployment not done |
| `backend/requirements.txt` | Generated | pinned runtime dependencies for the tested backend/model artifact |
| `backend/.env.example` | Generated | private data/config/model paths, auto-train, CORS |
| `render.yaml` | Generated | free Render Blueprint; `CORS_ORIGINS` requires Vercel origin input |
| `frontend/` | Tested | clean npm install, production build, Playwright local flows |
| `frontend/.env.example` | Generated | public API base URL template |
| `frontend/vercel.json` | Generated | React Router history fallback |
| `tests/test_api.py` | Tested | four synthetic FastAPI contract tests pass |
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
| `.gitignore` | Updated by user/agent | only `frontend/node_modules/` and `frontend/dist/` are active ignore entries; remaining lines are commented |
| `CLAUDE.md` | Generated | dataset status and current risk-band defaults |
| `data/*.csv`, `models/*.joblib` | tracked in current `main` commit `54bf44d`; available in repository checkout | measured in technical audit; commit visibility depends on GitHub repository settings |
| `layout_backup/`, `customer-churn.csv`, `files.zip`, `.kilo/` | not used by current app runtime; retained as tracked backups/data or tooling config because provenance/ownership is unclear | n/a |

## Module dependencies
`app.py` -> `ui.*`. `ui.state` -> `churn.workspace`, `churn.batch`, `churn.data`, `churn.recommend`, `churn.scoring`, `churn.explain`. Every `ui.page_*` -> `ui.theme`, `ui.common`, `ui.state`. `ui.common` -> `ui.state`, `ui.theme`. `churn.workspace` -> `churn.models`, `churn.scoring`, `churn.explain`, `churn.preprocess`, `churn.schema`. `churn.batch` -> `churn.recommend`. `churn/` never imports `ui/`.

## Known issues and open items
- Moderate model quality (AUC about 0.755 vs 18.1% base rate). No time-based split (no dates).
- First launch blocks on training with a spinner; two browser sessions opening at the same moment on a fresh install could both train.
- The frontend no longer exposes uploads, manual retraining, or held-out model evaluation; training still occurs automatically if no usable saved model exists.
- A bundle saved by `python -m churn.cli train` has no evaluation and is not used by the app.
- Recommendation actions are heuristics; the UI says so on every page that shows them.
- Vercel build fix is in pushed commit `54bf44d`; failure logs supplied so far target older commit `fceeda8`, so confirm deployments use the new SHA.
- FastAPI is unauthenticated and keeps upload/session overrides in process memory; use access control and private persistent storage before exposing real customer data.
- Render logs show the existing service still builds from root `requirements.txt` (pandas 2.2.3 / NumPy 1.26.4) rather than `backend/requirements.txt` (pandas 3.0.3 / NumPy 2.4.6 / sklearn 1.9.0 matching the model). It then OOMs during lifespan startup. Set the existing Render service build command to `pip install -r backend/requirements.txt` and `CHURN_AUTO_TRAIN=false`, redeploy, then inspect `/api/health`.

## Next action
1. In the existing Render service settings, set Build Command to `pip install -r backend/requirements.txt`; set `CHURN_AUTO_TRAIN=false` and the exact `CORS_ORIGINS` Vercel origin. The changed Blueprint also contains these settings, but the service must sync it or be edited directly.
2. Deploy and wait for `/api/health` to return `model_loaded: true`; if it reports false, inspect saved-model load notes and dependency versions before enabling training.
3. Configure Vercel `VITE_API_BASE_URL` with the Render origin and redeploy. Before public use with sensitive customer data, add API authentication and persistent private storage. Keep Streamlit as fallback until production is verified.

## Manual steps for the user
The 46-test suite has passed; frontend `npm ci` and `npm run build` have passed locally. Render/Vercel production deployment remains unverified. For local setup, follow `README.md`; Streamlit fallback starts with `python -m streamlit run app.py`.
