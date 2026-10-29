"""Streamlit app: load data -> map columns -> train -> score -> explain -> recommend.
Run from the project root:  streamlit run app.py"""
from pathlib import Path

import streamlit as st

from churn import data as D
from churn.explain import ExplainError, explain_rows, explanation_unit, top_drivers
from churn.models import calibration_table, train_and_evaluate
from churn.recommend import load_rules, recommend, validate_rules
from churn.schema import Schema, target_value_counts, validate_schema
from churn.scoring import RiskBands, band_summary, load_bundle, make_bundle, save_bundle, score_table

ROOT = Path(__file__).parent
RULES_PATH = ROOT / "config" / "rules.json"
MODEL_PATH = ROOT / "models" / "model.joblib"

st.set_page_config(page_title="Churn Prediction, Explanation and Recommendation", layout="wide")
S = st.session_state
for k in ("df", "result", "bundle", "scored"):
    S.setdefault(k, None)

st.title("Customer Churn: Predict, Explain, Recommend")
st.caption("Nothing here is pre-filled. Results appear only after you load real data and train on it.")
tab_data, tab_setup, tab_train, tab_cust = st.tabs(["1. Data", "2. Columns", "3. Train", "4. Customers"])

# ---------------------------------------------------------------- 1. Data
with tab_data:
    up = st.file_uploader("Customer data (CSV, TSV or Excel)", type=["csv", "tsv", "txt", "xlsx", "xls"])
    if up is not None and S.get("_file") != (up.name, up.size):
        try:
            S.df = D.load_table(up, filename=up.name)
            S._file = (up.name, up.size)
            S.result = S.bundle = S.scored = None
        except D.DataLoadError as exc:
            st.error(str(exc))
    if S.df is not None:
        ov = D.overview(S.df)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", ov["rows"]); c2.metric("Columns", ov["columns"]); c3.metric("Duplicate rows", ov["duplicate_rows"])
        if ov["fully_empty_columns"]:
            st.warning(f"Completely empty columns: {ov['fully_empty_columns']}")
        st.subheader("Column profile")
        st.dataframe(D.profile(S.df), use_container_width=True)
        st.subheader("First rows")
        st.dataframe(S.df.head(20), use_container_width=True)
    else:
        st.info("Upload a file to begin.")

# ---------------------------------------------------------------- 2. Columns
with tab_setup:
    if S.df is None:
        st.info("Load data first.")
    else:
        df = S.df
        cols = list(df.columns)
        target = st.selectbox("Which column says whether the customer churned?", cols, index=None,
                              placeholder="Choose the churn column")
        if target:
            vc = target_value_counts(df, target)
            st.write("Values found in that column:")
            st.dataframe(vc, hide_index=True)
            pos = st.multiselect("Which value(s) mean the customer CHURNED?", list(vc["value"]))
            idc = st.selectbox("Customer ID column (optional)", [None] + [c for c in cols if c != target])
            default_feats = [c for c in cols if c not in (target, idc)]
            feats = st.multiselect("Columns the model may use as inputs", [c for c in cols if c != target],
                                   default=default_feats)
            st.caption("Remove any column that is only known AFTER a customer leaves (for example a cancellation "
                       "date or a leaving reason); it would make results look better than they are.")
            S.schema = Schema(target=target, positive_labels=pos, feature_columns=[f for f in feats if f != idc],
                              id_column=idc)
            errs, warns = validate_schema(df, S.schema)
            for m in errs:
                st.error(m)
            for m in warns:
                st.warning(m)
            if not errs:
                st.success("Column choices look consistent. Go to the Train tab.")

# ---------------------------------------------------------------- 3. Train
with tab_train:
    schema = S.get("schema")
    if S.df is None or schema is None or validate_schema(S.df, schema)[0]:
        st.info("Finish the Data and Columns steps first.")
    else:
        c1, c2, c3 = st.columns(3)
        test_size = c1.slider("Share held out for testing", 0.1, 0.4, 0.2, 0.05)
        balance = c2.checkbox("Balance classes (give churners more weight)", value=True)
        seed = int(c3.number_input("Random seed", value=42, step=1))
        if st.button("Train and compare models", type="primary"):
            with st.spinner("Training..."):
                try:
                    S.result = train_and_evaluate(S.df, schema, test_size=test_size, seed=seed, balance=balance)
                    S.bundle = S.scored = None
                except ValueError as exc:
                    S.result = None
                    st.error(f"Cannot train: {exc}")
        res = S.result
        if res is not None:
            sp = res.split
            st.write(f"Rows used: {sp['rows_used']} (train {sp['train']}, test {sp['test']}). "
                     f"Churn rate: {sp['churn_rate_all']:.1%} overall, {sp['churn_rate_test']:.1%} in the test set.")
            for n in res.spec.notes + res.warnings:
                st.warning(n)
            if res.spec.excluded:
                st.info("Columns not used: " + "; ".join(f"{k} ({v})" for k, v in res.spec.excluded.items()))
            st.subheader("Results on the held-out test set")
            st.dataframe(res.metrics.style.format("{:.3f}"), use_container_width=True)
            st.caption(f"Suggested model (best cross-validated {res.selection_metric} on training data): "
                       f"{res.best_name}. Always compare against the churn rate: a model must beat that baseline.")
            choice = st.selectbox("Model to use", list(res.pipelines), index=list(res.pipelines).index(res.best_name))
            b1, b2 = st.columns(2)
            low = b1.slider("Low risk below", 0.05, 0.60, 0.30, 0.05)
            high = b2.slider("High risk from", 0.40, 0.95, 0.70, 0.05)
            st.caption("These cut-offs are settings you can change to match how many customers your team can contact.")
            st.write("Calibration (do predicted probabilities match what actually happened?)")
            st.dataframe(calibration_table(res.y_test, res.test_proba[choice]).style.format(
                {"mean_predicted": "{:.3f}", "observed_rate": "{:.3f}"}), use_container_width=True)
            if low < high and st.button("Use this model"):
                S.bundle = make_bundle(res, choice, RiskBands(low, high))
                S.scored = None
                st.success(f"Model '{choice}' is ready. Go to the Customers tab.")
            if S.bundle is not None and st.button("Save model to models/model.joblib"):
                MODEL_PATH.parent.mkdir(exist_ok=True)
                save_bundle(MODEL_PATH, S.bundle)
                st.success(f"Saved to {MODEL_PATH}")
    if MODEL_PATH.exists() and S.bundle is None and st.button("Load saved model (models/model.joblib)"):
        S.bundle = load_bundle(MODEL_PATH)
        st.success("Loaded. Go to the Customers tab.")

# ---------------------------------------------------------------- 4. Customers
with tab_cust:
    b = S.bundle
    if b is None:
        st.info("Train and choose a model (or load a saved one) first.")
    else:
        src = S.df
        other = st.file_uploader("Score a different file (optional; must contain the same input columns)",
                                 type=["csv", "tsv", "txt", "xlsx", "xls"], key="score_file")
        if other is not None:
            try:
                src = D.load_table(other, filename=other.name)
            except D.DataLoadError as exc:
                st.error(str(exc)); src = None
        if src is not None and st.button("Score customers"):
            try:
                S.scored = (src, score_table(b, src))
            except ValueError as exc:
                st.error(str(exc))
        if S.scored is not None:
            src, sc = S.scored
            st.subheader("Risk overview")
            st.dataframe(band_summary(sc["churn_probability"], b.bands), hide_index=True)
            level = st.multiselect("Show risk levels", ["High", "Medium", "Low"], default=["High"])
            view = sc[sc["risk_level"].isin(level)].sort_values("churn_probability", ascending=False)
            st.dataframe(view, use_container_width=True)
            st.download_button("Download all scores (CSV)", sc.to_csv(index=False), "churn_scores.csv", "text/csv")
            if len(view):
                pick = st.selectbox("Inspect a customer (row position in the file)", list(view.index),
                                    format_func=lambda i: f"row {i}: {view.loc[i].to_dict()}")
                row_df = src.loc[[pick]]
                rowsc = sc.loc[pick]
                st.metric("Churn probability", f"{rowsc['churn_probability']:.1%}", rowsc["risk_level"])
                try:
                    contrib = explain_rows(b.pipeline, b.spec, row_df, b.background).iloc[0]
                    unit = explanation_unit(b.pipeline.named_steps["clf"])
                    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(10)
                    st.write(f"Why: contribution of each input to this customer's score ({unit}). "
                             "Positive pushes risk up. This shows what the model relies on, not proven causes.")
                    st.bar_chart(top)
                    try:
                        rules = load_rules(RULES_PATH) if RULES_PATH.exists() else []
                    except (ValueError, KeyError, TypeError) as exc:
                        rules = []
                        st.error(f"config/rules.json is not valid: {exc}")
                    for m in validate_rules(rules, b.spec.used):
                        st.warning(m)
                    if not rules:
                        st.caption("No rules defined yet. Add rules to config/rules.json to get recommended actions.")
                    rec = recommend(rules, rowsc["risk_level"], top_drivers(contrib, 5), src.loc[pick])
                    st.subheader("Recommendation")
                    st.write(rec["summary"])
                    for a in rec["actions"]:
                        st.success(f"{a['action']}  \n_because: {a['because']}_")
                except ExplainError as exc:
                    st.error(str(exc))
