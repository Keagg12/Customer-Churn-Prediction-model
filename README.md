# Churn Intelligence

Explainable customer churn prediction with rule-based retention recommendations.
Workflow: **Predict, Explain, Recommend, Retain.**

Stack: Python, pandas, scikit-learn, SHAP, Streamlit.

## Run

```bash
pip install -r requirements.txt
python -m unittest discover tests
streamlit run app.py
```

Put the dataset in `data/`:

| File | Purpose |
|---|---|
| `data/train.csv` | Labeled customers (has the `Churn` column). Used for training, held-out evaluation and observed churn patterns. |
| `data/test.csv` | Customers to score. No label is needed and none is used. |
| `data/data_descriptions.csv` | Column dictionary (optional). |

On first launch the app trains and compares four models on `train.csv` and saves the result to
`models/model.joblib`. Later launches load it. Retraining only happens when you press the button in the
Prediction Workspace (or when the saved model cannot be used on this machine).

## Pages

| Page | What it shows |
|---|---|
| Overview | Observed churn (labeled history) beside predicted risk (model estimate), where churn concentrates, who needs attention first. |
| Customer Explorer | Search, filter, sort and select customers. |
| Customer Intelligence | One customer: score, attributes, the model's explanation, and the business rules that apply (kept separate). |
| Model Insights | Held-out metrics, ROC curves, confusion matrix, calibration, risk-band check, feature importance, limitations. |
| Retention Center | Priority queue with main risk factors and applicable rules. |
| Prediction Workspace | Optional uploads, validation, retraining, scoring new customers, risk bands. |

## Configuration

- `config/dataset.json`: churn column, the value that means churned, ID column, excluded columns.
- `config/rules.json`: retention rules. Triggers use real columns and data-derived thresholds. The actions are
  business judgement; their effect on churn has not been tested.
- Risk bands default to Low below 15%, Medium 15% to below 30%, High 30% and above. Adjustable in the Workspace.
- Class balancing is off by default because it inflates predicted probabilities.

## Layout

```text
app.py                 entry point and navigation
ui/                    Streamlit pages, theme and session handling
churn/                 data loading, training, scoring, explanations, rules (no Streamlit)
config/                dataset.json, rules.json
tests/                 test_core.py, test_workspace.py
data/  models/         local files (not committed)
```

## What the results do and do not mean

- Metrics are computed only on a labeled held-out split of `train.csv`. Unlabeled customers are scored, never evaluated.
- Explanations show what the model relies on. They are not proven causes of churn.
- The data is a single snapshot per customer. It has no dates, so usage trends cannot be measured.
- `SupportTicketsPerMonth` is a proxy for service friction, not a record of complaints.
- `TotalCharges` equals `AccountAge` times `MonthlyCharges`, so it is excluded.

Command line helpers (inspect a file, train and save a bundle): `python -m churn.cli --help`.
A bundle saved by the command line tool has no evaluation, so the app trains its own instead of using it.
