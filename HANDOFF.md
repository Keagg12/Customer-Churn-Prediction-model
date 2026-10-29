# HANDOFF (checkpoint 29 Oct 2026, application-only scope). Must stand alone.

## For the next Claude
Read `CLAUDE.md`, then inspect every attached file and reconcile it with this handoff. The attached files are the truth for what exists. Continue from "Next action" below. Do not restart the project or redo finished work. You have no access to earlier chats or the user's machine.

## Project
**Standalone product** (independent developer; not a college/academic project): Explainable AI-Based Customer Churn Prediction and Personalized Marketing Recommendation System, Predict → Explain → Recommend. Follow the branding rules in `CLAUDE.md`: no institutional, mentor, team or roll-number details, no invented company or results, and do not ask the user about them. Example (illustrative only): customer Rahul, 86%, drivers low usage / many complaints / reduced purchases / short membership, action: personalised retention offer.

## Dataset
**Not received; expected 30 Oct 2026.** Format, columns, labels and quality are unknown. Do not assume any. Apply the six dataset checks in `CLAUDE.md` when it arrives.

## Decisions so far
- Stack: Python, pandas, scikit-learn, SHAP, Streamlit UI. Candidate models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting (sklearn). XGBoost was removed to keep dependencies and untested paths minimal; add it back only if the real data shows a need.
- Design: schema-agnostic core. The user picks target, churn value(s), ID and feature columns from the real file (UI or CLI flags). Nothing is hard-coded to a column name.
- Evaluation: stratified (or ID-grouped) hold-out test set; model chosen by cross-validated average precision on the training part only; test metrics reported; class weights for imbalance (no SMOTE); single-column leakage screen; calibration table.
- Explanations: exact linear contributions for Logistic Regression; SHAP for other models. One-hot pieces are summed back to the original column.
- Risk bands default Low <30%, Medium 30% to <70%, High >=70%; user-adjustable; provisional.
- Recommendations: rule engine reading `config/rules.json`, edited in VS Code (no in-app rule editor). The rules list is EMPTY on purpose until real feature names are known; with no matching rule the app says so and lists the top risk-raising factors, never inventing an action. Starter rule ideas (write after seeing data): high risk + low usage → engagement/tutorial; + many complaints → support intervention; + reduced purchases → personalised discount; + short tenure → onboarding help.
- Simplification review (done): kept one module per distinct job; merged risk bands + save/load + scoring into `scoring.py`; trimmed the CLI to `inspect` and `train`; dropped the app's Rules tab and `tests/__init__.py`. Do not merge further without a concrete reason.
- Scope: application only. PPT, synopsis and other documents are reference only (see CLAUDE.md).

## Status
- **Code (all Generated, none Applied by the user yet):** core package, CLI, Streamlit app, tests. See manifest.
- **Tested in Claude's sandbox only** (Python 3.12.3, pandas 3.0.2, scikit-learn 1.8.0): 24 tests pass (`python -m unittest discover tests`), also on a clean copy, using a synthetic fixture with invented columns. This verifies mechanics, NOT any project result.
- **NOT run anywhere yet:** the SHAP path (non-linear models) and the Streamlit UI (`app.py` only compile-checked), because shap and streamlit were unavailable in the sandbox. The user's first run must confirm these.
- **No dataset yet** (expected 30 Oct 2026). No real results exist.

## File manifest (paths relative to the user's repo root; Claude's copies are in `app_files/`)
| Path | Purpose | Change | Status |
|---|---|---|---|
| `requirements.txt` | dependencies (no XGBoost) | REPLACEMENT | Generated |
| `.gitignore` | keeps data/models out of git | unchanged | Generated |
| `app.py` | Streamlit app, 4 tabs: Data, Columns, Train, Customers | REPLACEMENT | Generated; compile-checked only |
| `churn/__init__.py` | package marker | unchanged | Generated |
| `churn/data.py` | file loading (CSV/Excel), profiling | unchanged | Tested (sandbox) |
| `churn/schema.py` | target/feature mapping, validation | unchanged | Tested (sandbox) |
| `churn/preprocess.py` | type inference, imputing, scaling, one-hot | unchanged | Tested (sandbox) |
| `churn/models.py` | train, compare, evaluate, leakage screen | REPLACEMENT | Tested (sandbox) |
| `churn/explain.py` | per-customer contributions | REPLACEMENT | Linear path tested; SHAP path untested |
| `churn/scoring.py` | risk bands, save/load model, score customers | NEW | Tested (sandbox) |
| `churn/recommend.py` | rule engine | unchanged | Tested (sandbox) |
| `churn/cli.py` | `inspect` and `train` text commands | REPLACEMENT | Tested (sandbox) |
| `config/rules.json` | recommendation rules (empty list) | unchanged | Generated |
| `tests/test_core.py` | 24 tests | REPLACEMENT | Pass in sandbox |
**DELETE if present from the earlier delivery:** `churn/risk.py`, `churn/bundle.py`, `tests/__init__.py` (their contents moved into `churn/scoring.py` or were unnecessary).
`CLAUDE.md` and `HANDOFF.md` are carried separately.

## Known issues / open items
- No time-based split yet; decide after seeing the data (does it have dates / repeated customers?).
- Hyperparameters are plain defaults, untuned. Tune only after the real data shows a need.
- No SMOTE (class weights used instead); revisit only if the real class balance demands it.
- Explanation units differ by model (log-odds for linear/boosting, probability for forest/tree); compare within a customer only.
- Rules are empty; they need real feature names (edit `config/rules.json` in VS Code; the app shows warnings for unknown names).
- Models are saved as joblib pickles: load only files you made yourself.

## Next action
1. User applies the files (VS Code), deletes the three obsolete files listed in the manifest, then in the repo root: `pip install -r requirements.txt` and `python -m unittest discover tests`. Expected: `Ran 24 tests ... OK`. Then `streamlit run app.py` should open with four tabs and show only an upload prompt. Report any error text verbatim; that changes Status from Generated to Applied/Tested.
2. When the dataset arrives: `python -m churn.cli inspect <file>` (or upload in the app) and share the output or the file. Claude answers the six dataset checks in `CLAUDE.md` with a verdict (proceed / minor adjustments / reconsider). If suitable: choose columns, train, review leakage warnings and metrics against the churn-rate baseline, write real rules in `config/rules.json`, then fix whatever the real data exposes.
**Needs from the user:** the dataset (with source, column description, churn definition, usage restrictions if known) and the output of step 1.
