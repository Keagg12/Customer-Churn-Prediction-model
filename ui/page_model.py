"""Model Insights: only held-out, labeled evaluation. Nothing here is computed on unlabeled customers."""
import numpy as np
import pandas as pd
import streamlit as st

from churn import workspace as W
from churn.models import calibration_table

from . import theme
from .common import MODEL_NOTE, setup_state
from .state import Ctx

LIMITATIONS = [
    ("Moderate ceiling.", "The model ranks customers better than chance but is far from certain. Many customers it scores as "
                          "high risk will stay, and some it scores as low risk will leave."),
    ("A snapshot, not a history.", "Each customer appears once. The data holds no dates, so the model cannot see a customer's "
                                   "usage falling over time, and it was not tested on a later period."),
    ("Explanations are not causes.", "They show which inputs the model leans on for a score. A different action on that input "
                                     "is not guaranteed to change the outcome."),
    ("Support tickets are a proxy.", "Tickets per month hints at service friction. It is not a record of complaints."),
    ("Rules are untested.", "Recommended actions are business judgement. Nothing in this data measures whether they reduce churn."),
]


def render(ctx: Ctx) -> None:
    theme.md(st, theme.header("Model Insights", "How well the model works",
                              "Every figure on this page is measured on labeled customers the model did not train on."))
    if ctx.pkg is None:
        setup_state(ctx, "No trained model yet")
        return
    ev, bundle = ctx.pkg.evaluation, ctx.pkg.bundle
    sp = ev.split
    chosen = bundle.model_name
    m = ev.metrics

    theme.md(st, theme.callout(
        f"Trained {ev.trained_at[:10]} on {ev.train_source.get('name', 'unknown source')}. {sp['train']:,} customers trained the models, "
        f"{sp['test']:,} held out for the figures below. Churn rate: {sp['churn_rate_all']:.1%} overall, "
        f"{sp['churn_rate_test']:.1%} in the held-out part. Class balancing: {'on' if ev.balance else 'off'}."))
    for w in ev.warnings:
        theme.md(st, theme.callout(w, "warn"))
    if ev.excluded:
        theme.md(st, theme.callout("Inputs not used: " + "; ".join(f"{W.pretty(k)} ({v})" for k, v in ev.excluded.items())))
    if ev.balance:
        theme.md(st, theme.callout("Class balancing inflates predicted probabilities. Do not read them as chances of churn.", "warn"))

    # ---- headline: the chosen model against the baseline
    p_sel = ev.test_proba[chosen]
    auc, ap = float(m.loc[chosen, "roc_auc"]), float(m.loc[chosen, "average_precision"])
    base = sp["churn_rate_test"]
    head = [
        theme.big_number(f"Selected model · {chosen}", f"{auc:.3f}", "ROC-AUC on held-out customers. 0.5 is chance, 1.0 is perfect.", "4.2rem"),
        theme.big_number("Average precision", f"{ap:.3f}", f"Compare with {base:.3f}, the share who churned. A model must clearly beat that baseline.", "2.6rem"),
        theme.big_number("Brier score", f"{float(m.loc[chosen, 'brier']):.3f}", "Lower is better. Measures how honest the probabilities are.", "2.6rem"),
    ]
    theme.md(st, theme.ledger(head, "5fr 4fr 3fr"))
    theme.md(st, f"<div class='ci-sub' style='margin-top:0.5rem'>{theme.esc(MODEL_NOTE)}</div>")
    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)

    left, right = st.columns([7, 5], gap="large")
    with left:
        theme.md(st, theme.section("Model comparison", f"Held-out metrics. “Selected” is chosen by cross-validated {ev.selection_metric.replace('_', ' ')} on training data only."))
        show = m.copy()
        show.insert(0, "role", ["selected" if i == chosen else ("best in training" if i == ev.best_name else "") for i in show.index])
        cols = [c for c in show.columns if c != "role"]
        names = {"roc_auc": "ROC-AUC", "average_precision": "Avg precision", "brier": "Brier", "accuracy": "Accuracy",
                 "precision": "Precision", "recall": "Recall", "f1": "F1", f"cv_{ev.selection_metric}": "CV " + ev.selection_metric.replace("_", " ")}
        view = show.rename(columns=names)
        view.index.name = "Model"
        fmt = {names.get(c, c): "{:.3f}" for c in cols}
        theme.md(st, theme.html_table(view, highlight=chosen, fmt=fmt))
        theme.md(st, f"<div class='ci-sub' style='margin-top:0.5rem'>Precision, recall, accuracy and F1 use a churn cut-off of {sp['threshold']:.0%}. "
                     "ROC-AUC, average precision and Brier do not depend on a cut-off.</div>")

        st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
        theme.md(st, theme.section("ROC curves", "How much churn each model catches as it flags more customers. Further up and to the left is better."))
        roc = W.roc_points(ev.y_test, ev.test_proba)
        roc["selected"] = roc["model"] == chosen
        theme.vega(st, roc, theme.spec_roc())

    with right:
        theme.md(st, theme.section("What the model relies on", "Share of the model's total attention per input, averaged over scored customers."))
        imp = ctx.importance()
        if len(imp):
            top = imp.head(10)
            chart = pd.DataFrame({"name": [W.pretty(i) for i in top.index], "value": top.to_numpy(), "order": range(len(top))})
            theme.vega(st, chart, theme.spec_bars_horizontal("Share of attention", "%", 320))
            theme.md(st, "<div class='ci-sub'>Describes the model, not causes of churn. Computed on the customers currently scored, without using churn labels.</div>")
        else:
            theme.md(st, theme.callout("Feature importance is not available for this model here."))
        st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)
        theme.md(st, theme.section("Confusion matrix", "Held-out customers at the High-risk cut-off."))
        cut = ctx.bands.high_min
        theme.md(st, theme.confusion_matrix(W.confusion_at(ev.y_test, p_sel, cut), cut))

    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
    a, b = st.columns(2, gap="large")
    with a:
        theme.md(st, theme.section("Do the risk bands hold up?", "Held-out customers grouped by the current bands."))
        bl = W.band_lift(ev.y_test, p_sel, ctx.bands)
        t = pd.DataFrame({"Customers": bl["customers"].map("{:,}".format).to_numpy(),
                          "Share of customers": (bl["share_of_customers"] * 100).map("{:.0f}%".format).to_numpy(),
                          "Actually churned": (bl["observed_churn_rate"] * 100).map("{:.1f}%".format).to_numpy(),
                          "Share of all churners": (bl["share_of_churners"] * 100).map("{:.0f}%".format).to_numpy()},
                         index=bl["risk_level"].to_numpy())
        t.index.name = "Band"
        theme.md(st, theme.html_table(t))
        frac = st.slider("Contact the highest-scored share of customers (%)", 1, 50, 10, 1, key="mi_frac")
        tf = W.top_fraction_check(ev.y_test, p_sel, frac / 100)
        theme.md(st, f"<div class='ci-sub' style='margin-top:0.4rem'>In held-out data the top {frac}% by score ({tf['customers']:,} customers) churned at "
                     f"{tf['observed_churn_rate']:.1%} and held {tf['share_of_churners']:.0%} of all churners. "
                     f"This checks how well the model ranks customers. It does not show that contacting them prevents churn.</div>")
    with b:
        theme.md(st, theme.section("Are the probabilities honest?", "Held-out customers split into ten equal groups by score."))
        cal = calibration_table(ev.y_test, p_sel)
        theme.vega(st, cal, theme.spec_calibration())
        gap = abs(float(np.mean(p_sel)) - float(np.mean(ev.y_test)))
        theme.md(st, f"<div class='ci-sub'>Points near the dashed line mean predicted and observed churn agree. The average prediction is "
                     f"{float(np.mean(p_sel)):.1%} against {float(np.mean(ev.y_test)):.1%} observed (a gap of {gap * 100:.1f} points).</div>")

    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
    theme.md(st, theme.section("Limitations", "Read these before acting on a score."))
    theme.md(st, theme.insight_list(LIMITATIONS))
