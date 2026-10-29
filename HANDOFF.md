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
- **Applied & Verified on User Machine (30 Oct 2026):**
  - All package files organized into `churn/` (`__init__.py`, `cli.py`, `data.py`, `explain.py`, `models.py`, `preprocess.py`, `recommend.py`, `schema.py`, `scoring.py`).
  - Configuration files organized into `config/` (`dataset.json`, `rules.json`).
  - Real dataset files recovered from `C:/Users/HP/Desktop/Dataset/` and placed intact into `data/` (`train.csv` [243,787 rows], `test.csv` [104,480 rows], `data_descriptions.csv`).
  - `tests/test_core.py` in `tests/`.
  - Dependencies: all packages (pandas, scikit-learn, shap, streamlit, joblib, openpyxl) are installed and verified.
- **Verification Performed on Local Machine:**
  - `python -m unittest discover tests`: All 28 tests PASSED (Ran 28 tests in 19.100s, OK).
  - End-to-end Python pipeline test against real datasets: Loaded 243,787 training rows, trained 4 models (Logistic Regression best with cv_average_precision 0.402, ROC-AUC 0.755), scored 104,480-row test dataset, computed SHAP/linear explanations, validated and evaluated recommendation rules.
  - Streamlit server: Launched via `python -m streamlit run app.py --server.headless true` and verified HTTP 200 OK response on port 8501.
- **Clutter (Preserved, Pending User Confirmation to Remove):**
  - `layout_backup/`: holds byte-identical duplicates (`churn.original`, `config.original`, `tests.original` = cli.py, rules.json, test_core.py).
  - `customer-churn.csv`: 923 rows, 64 KB, unrelated Telco dataset (not used by the application).
  - `files.zip`: root archive of the extracted helper files.
  - `.kilo/`: leftover worktree metadata from Kilo agent manager.

## File manifest (paths relative to repo root)
| Path | Status | Verification |
|---|---|---|
| `app.py` | Present | UI verified on port 8501 |
| `churn/__init__.py` | Moved to churn/ | Tested |
| `churn/cli.py` | Present in churn/ | Tested |
| `churn/data.py` | Moved to churn/ | Tested |
| `churn/explain.py` | Present in churn/ | Tested (Linear/SHAP verified) |
| `churn/models.py` | Present in churn/ | Tested (4 models trained on real data) |
| `churn/preprocess.py` | Moved to churn/ | Tested |
| `churn/recommend.py` | Present in churn/ | Tested (Rule evaluation verified) |
| `churn/schema.py` | Moved to churn/ | Tested |
| `churn/scoring.py` | Present in churn/ | Tested (Bands 15/30 verified) |
| `config/dataset.json` | Moved to config/ | Tested |
| `config/rules.json` | Present in config/ | Tested against real columns |
| `tests/test_core.py` | Present in tests/ | 28/28 tests passing |
| `data/train.csv` | Copied from Desktop/Dataset | Verified (243,787 rows, 18 features + CustomerID + Churn) |
| `data/test.csv` | Copied from Desktop/Dataset | Verified (104,480 rows, unlabeled) |
| `data/data_descriptions.csv` | Copied from Desktop/Dataset | Verified |
| `requirements.txt` | Present | All packages installed |
| `.gitignore` | Present | Ignores data/, models/, caches |
| `CLAUDE.md` | Present | Up to date |

## Known issues / open items
- Model quality is moderate (AUC ≈0.755, baseline churn rate 18.1%). Possible later improvements: tune hyperparameters, simple interaction features, if desired.
- No time-based split (no dates in data).
- Recommendation actions are business heuristics mapped to data-driven risk triggers.

## Next action
1. Open Streamlit in the browser at `http://localhost:8501` to use the application.
2. Confirm cleanup of clutter files (`layout_backup/`, `customer-churn.csv`, `files.zip`, `.kilo/`).
3. If desired, discuss model optimization or rule adjustments.
