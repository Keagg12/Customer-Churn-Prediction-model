# HANDOFF (checkpoint 30 Oct 2026). Must stand alone.

## For the next Claude
Read `CLAUDE.md`, then inspect every attached file and reconcile it with this handoff. The attached files are the truth for what exists. Continue from "Next action". Do not restart or redo finished work. You have no access to earlier chats or the user's machine.

## Project
Working Explainable AI-Based Customer Churn Prediction and Personalized Marketing Recommendation application (Predict → Explain → Recommend). Application only; old PPT/synopsis are reference, not deliverables.

## Dataset (received and assessed 30 Oct 2026): SUITABLE, proceed
- `train.csv`: 243,787 rows, 18 features + `CustomerID` + `Churn` (1 = churned, 0 = not). Churn rate 18.1% (44,182 churners). No missing values, no duplicates, IDs unique. Used for training AND evaluation (internal stratified 80/20 split plus 5-fold CV on the 80%).
- `test.csv`: 104,480 rows, same columns but **no `Churn` label**, so it cannot be used to measure accuracy. Use it only as "new customers to score" (no ID overlap with train; feature distributions match train to <1% shift).
- `data_descriptions.csv`: column dictionary (matches the data; confirms `Churn` is the target and `CustomerID` the identifier).
- Features: AccountAge (months), MonthlyCharges, TotalCharges, SubscriptionType, PaymentMethod, PaperlessBilling, ContentType, MultiDeviceAccess, DeviceRegistered, ViewingHoursPerWeek, AverageViewingDuration, ContentDownloadsPerMonth, GenrePreference, UserRating, SupportTicketsPerMonth, Gender, WatchlistSize, ParentalControl, SubtitlesEnabled.
- Evidence of real signal, no leakage (leakage screen clean): churn falls steadily with tenure (30% in newest fifth vs 9% in oldest), viewing hours, session length, downloads; rises with support tickets, monthly charges, Basic plan.
- Caveats: data looks simulated (flat distributions; `TotalCharges` = `AccountAge` × `MonthlyCharges` exactly on every row, so it is excluded as redundant). Single snapshot per customer: there is NO "reduced purchases / trend" field, so that idea from the original brief cannot be supported. No explicit "complaints" field; `SupportTicketsPerMonth` is the closest proxy. Predictive ceiling is moderate.

## Results (real data, held-out 20% = 48,758 rows; Claude's sandbox)
| Model | ROC-AUC | Avg precision (baseline 0.181) | Brier | Precision / recall at 30% |
|---|---|---|---|---|
| Logistic Regression (selected) | 0.755 | 0.409 | 0.129 | 0.42 / 0.43 |
| Gradient Boosting | 0.751 | 0.404 | 0.129 | 0.42 / 0.42 |
| Random Forest (light) | 0.748 | 0.398 | 0.132 | 0.46 / 0.31 |
| Decision Tree | 0.710 | 0.339 | 0.135 | 0.39 / 0.32 |
Risk bands (Low <15%, Medium 15-30%, High ≥30%) on held-out data: Low churns 8%, Medium 22%, High 42%; High is 18% of customers and holds 43% of churners. Scoring `test.csv` gives the same shares (52% / 30% / 18%) and average risk 18.1%.

## Decisions (with reasons)
- **No class balancing by default** (`balance=False`): balancing made average predicted churn 44% vs actual 18% (probabilities wrong, Brier 0.203 vs 0.129) with identical ranking. Honest probabilities matter for the "X% chance of churn" display.
- **Risk bands 15% / 30%** (adjustable in app/CLI): original 30/70 would put only 0.3% of customers in High on calibrated probabilities.
- **Exclude `TotalCharges`** (redundant); config in `config/dataset.json` preselects target/positive label/ID/exclusions in the app.
- Model choice: Logistic Regression wins on cross-validated average precision and gives exact explanations; tree/boosting models are no better here. Explanations for non-linear models need `shap`.
- Rules in `config/rules.json` use real columns and the 20%/80% cut points of `train.csv` (SupportTickets ≥6, MonthlyCharges >14, AccountAge <24, ViewingHours <16, AvgViewingDuration <40, Downloads <10). Triggers are data-based; the ACTIONS are business judgement and their effectiveness is unproven. App shows at most 3 actions per customer.
- Stack: Python, pandas, scikit-learn, SHAP, Streamlit.

## Status
- **Applied by user:** earlier files were byte-identical to Claude's versions when uploaded 29 Oct. User-run test/app results have not been reported.
- **Tested in Claude's sandbox:** 28 unit/end-to-end tests pass (`python -m unittest discover tests`); full real-data run (train on `train.csv`, score `test.csv`, explain + recommend) succeeded (~100 s on one CPU core).
- **NOT run anywhere yet:** the Streamlit UI (compile-checked only) and the SHAP path for non-linear models (shap/streamlit not installable in the sandbox).

## File manifest (paths relative to repo root; Claude's copies in `app_files/`)
| Path | Change this checkpoint | Status |
|---|---|---|
| `app.py` | REPLACEMENT (dataset.json defaults, 15/30 sliders, no-balance default, metrics note) | compile-checked only |
| `churn/models.py` | REPLACEMENT (balance off, lighter RF, metrics threshold 30%) | Tested (sandbox) |
| `churn/scoring.py` | REPLACEMENT (bands 15/30) | Tested (sandbox) |
| `churn/cli.py` | REPLACEMENT (`--balance` flag, 15/30 defaults) | Tested (sandbox) |
| `churn/recommend.py` | REPLACEMENT (`max_actions`) | Tested (sandbox) |
| `config/rules.json` | REPLACEMENT (real rules) | Tested (validated against real columns) |
| `config/dataset.json` | NEW | Tested |
| `tests/test_core.py` | REPLACEMENT (28 tests) | Pass in sandbox |
| `churn/__init__.py`, `data.py`, `schema.py`, `preprocess.py`, `explain.py`, `requirements.txt`, `.gitignore` | unchanged | — |
Obsolete (delete if present): `churn/risk.py`, `churn/bundle.py`, `tests/__init__.py`. Data files and `models/` stay local (gitignored).

## Known issues / open items
- Model quality is moderate (AUC ≈0.75). Possible later improvements: tune hyperparameters, simple interaction features; only if the user wants to push performance.
- No time-based split (no dates in data).
- Explanation units differ by model; compare within a customer only.
- Joblib model files are pickles: load only your own.

## Next action
1. User applies the REPLACEMENT/NEW files, puts `train.csv` and `test.csv` anywhere (e.g. `data/`), runs `python -m unittest discover tests` (expect `Ran 28 tests ... OK`) and `streamlit run app.py`.
2. In the app: upload `train.csv` → Columns tab should be preselected (Churn, value 1, CustomerID, TotalCharges unticked) → Train (about 1-3 minutes) → pick the suggested model → Customers tab: upload `test.csv`, Score, inspect a High-risk customer (bar chart + up to 3 actions).
3. User reports any error text verbatim; fix, then mark UI/SHAP as Tested.
**Needs from the user:** the output/errors from step 1-2 (or confirmation that it works), and any wishes to change rules/actions or push model performance.
