tests/                 test_core.py, test_workspace.py
data/  models/         local files (not committed)
# Churn Intelligence

Explainable customer churn prediction with deterministic, rule-based retention suggestions. Workflow: **Predict, Explain, Recommend, Retain.**

The current web application uses React 19, Vite, Tailwind CSS, React Router, Recharts, Lucide React, FastAPI, pandas, and scikit-learn. The original Streamlit app remains in the repository as a fallback; the React UI is under `frontend/` and its API under `backend/`.

## Features

| React page | Streamlit equivalent | Functionality |
|---|---|---|
| Overview | Overview | Observed labeled churn and predicted risk side by side, risk distribution, historical group rates, highest-risk customers. |
| Customer Explorer | Customer Explorer | Search by ID, filter risk/probability/segment, sort, choose 100/250/500 rows, view model-relevant attributes/actions, export CSV, open profile. |
| Customer Intelligence | Customer Intelligence | Exact ID lookup or top-100 picker, score, customer attributes, per-customer model contributions, separately matched rule suggestions. |
| Model Insights | Model Insights | Cross-validation/held-out candidate metrics, ROC curves, confusion matrix, calibration, risk-band outcomes, ranking check, global feature reliance, limitations. |
| Retention Center | Retention Center | Risk/segment/rule filters, priority queue, main risk factors, matching suggestions, expected churn sum, CSV export. |
| Prediction Workspace | Prediction Workspace | Status, schema mapping for uploaded training data, model training and persistence choice, upload scoring and full CSV export, session risk-band controls, reset. |

The Python calculations remain in `churn/`. The FastAPI adapter does not reimplement training, scoring, explanations, target normalization, or rule matching in JavaScript.

## Local Development

Use two terminals from the repository root. PowerShell setup for the backend:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
Copy-Item backend/.env.example backend/.env
python -m uvicorn backend.main:app --reload --env-file backend/.env --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
cd frontend
Copy-Item .env.example .env
npm ci
npm run dev
```

Open `http://localhost:5173`. Vite proxies `/api` to `http://127.0.0.1:8000` when the frontend environment file is absent or leaves `VITE_API_BASE_URL` empty. To override, set `VITE_API_BASE_URL=http://localhost:8000` in `frontend/.env`.

Run checks from the repository root:

```powershell
python -m unittest discover tests
npm --prefix frontend ci
npm --prefix frontend run build
```

FastAPI health and interactive API documentation are at `/api/health` and `/docs` on port 8000.

## Data and Model

Default runtime files are `data/train.csv` (labeled training/evaluation/history), `data/test.csv` (unlabeled scoring only), optional `data/data_descriptions.csv`, `config/dataset.json`, and `config/rules.json`. The current pushed Git commit includes `data/` and `models/`, so Render can read them when deploying this repository. For sensitive customer data, move them to authorized private/persistent storage and configure the `CHURN_*_DIR` variables instead. Without a usable model package the backend trains on `train.csv` and saves to `models/model.joblib`.

`config/dataset.json` defines the target, positive target values, ID, and excluded columns. `config/rules.json` defines six heuristic actions. Actions are not trained or proven to reduce churn. Risk bands default to Low below 15%, Medium from 15% to below 30%, High from 30% upward; these are operational defaults, not established optimal cut-offs. Class balancing is off by default.

Only labeled held-out training rows are used for reported model metrics. `data/test.csv` has no target and is never used to evaluate quality. Explanations describe what the model relies on, not causal reasons for churn. The data is one snapshot per customer, with no dates for a temporal evaluation. See [TECHNICAL_AUDIT.md](TECHNICAL_AUDIT.md) for measured dataset results, model metrics, architecture diagrams, and file classifications.

## Deployment

Deploy the static React frontend to Vercel and the Python API separately to a Python web-service host such as Render. Vercel project settings: set **Root Directory** to `frontend`, build command to `npm run build`, output directory to `dist`, and configure `VITE_API_BASE_URL` to the public API origin. `frontend/vercel.json` provides history fallback for React Router.

Do not commit `frontend/node_modules/` or `frontend/dist/`. Vercel must install dependencies on its Linux build image from `frontend/package-lock.json`; committing Windows `node_modules` can preserve non-executable file modes and cause `vite: Permission denied`. The package manifest explicitly allows esbuild's required install script under npm 11.

For Render, choose **New → Blueprint**, connect this repository and branch `main`, and apply the checked-in `render.yaml`. It creates a free Python web service from the repository root with health check `/api/health`, build command `pip install -r backend/requirements.txt`, and start command `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`. After creation, open the service's Environment settings and set `CORS_ORIGINS` to the exact Vercel origin (for example `https://your-project.vercel.app`). The Blueprint uses `./data`, `./config`, and `./models/model.joblib` from the repository by default; change `CHURN_DATA_DIR`, `CHURN_CONFIG_DIR`, and `CHURN_MODEL_PATH` when using private mounted storage. It sets `CHURN_AUTO_TRAIN=false` because model training can exceed Render Free's 512 MiB limit. The default filesystem is ephemeral, so persist any newly trained model externally or expect updates to be lost on redeploy/restart.

The root `requirements.txt` now includes `backend/requirements.txt`, so either `pip install -r requirements.txt` or the explicit backend command installs the pinned API dependencies. Pandas 3.0.3, NumPy 2.4.6, scikit-learn 1.9.0, and PyArrow 26.0.0 match the committed joblib model; PyArrow is required to deserialize pandas Arrow-backed strings. The Blueprint sets `CHURN_AUTO_TRAIN=false` so a missing/incompatible model does not trigger four-model training on Render Free's 512 MiB limit; if health reports no model, inspect the startup notes/model dependencies before retraining or using a higher-memory service. Render's filesystem is ephemeral, so model updates need persistent storage.

Do not commit real customer CSVs, model artifacts, credentials, or `.env` files. The API currently has no user authentication or authorization; do not expose it publicly with sensitive data until protected by an authentication/access-control layer. Browser environment variables prefixed `VITE_` are public and must contain no secrets.

## Repository Layout

```text
frontend/      React pages/components, Vite, Tailwind, Vercel SPA config
backend/       FastAPI endpoints, Render Blueprint, request/session orchestration
churn/         Existing Python business and machine-learning logic
ui/ app.py     Original Streamlit application, retained as fallback
config/        Dataset mapping and retention rules
data/ models/  Current committed CSV inputs and generated model packages; protect or move if sensitive
tests/         Backend unit tests and FastAPI contract tests
```

The standalone CLI remains available with `python -m churn.cli --help`. Its saved bundle does not contain held-out evaluation, so the dashboard API does not accept that artifact as a complete evaluation package.
