"""Mechanics tests. The data below is a synthetic TEST FIXTURE with arbitrary column names made up
for these tests only. It says nothing about the real dataset, and no result here is a project result."""
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from churn import data as D
from churn.cli import main as cli_main
from churn.explain import aggregate_by_origin, explain_rows, positive_class_values, top_drivers
from churn.models import calibration_table, metrics_at, train_and_evaluate
from churn.preprocess import build_preprocessor, feature_origins, infer_feature_spec, prepare_frame
from churn.recommend import load_rules, parse_rules, recommend, validate_rules
from churn.schema import Schema, make_target, norm_value, validate_schema
from churn.scoring import RiskBands, band_summary, load_bundle, make_bundle, save_bundle, score_table


def fixture(n=600, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    grp = rng.choice(["g1", "g2", "g3"], size=n)
    logit = 1.2 * a - 0.8 * b + (grp == "g3") * 1.0 - 0.5
    y = rng.random(n) < 1 / (1 + np.exp(-logit))
    df = pd.DataFrame({"cust": [f"id{i}" for i in range(n)], "f_a": a, "f_b": b, "f_grp": grp,
                       "outcome": np.where(y, "left", "stayed")})
    df.loc[rng.choice(n, 30, replace=False), "f_a"] = np.nan
    df.loc[rng.choice(n, 20, replace=False), "f_grp"] = None
    return df


SCHEMA = Schema(target="outcome", positive_labels=["left"], feature_columns=["f_a", "f_b", "f_grp"], id_column="cust")


class LoadTests(unittest.TestCase):
    def test_csv_delimiters_and_encoding(self):
        for sep in [",", ";", "\t", "|"]:
            raw = f"a{sep}b{sep}c\n1{sep}x{sep}2\n3{sep}y{sep}4\n".encode("utf-8")
            df = D.load_table(io.BytesIO(raw), filename="t.csv")
            self.assertEqual(list(df.columns), ["a", "b", "c"], sep)
        df = D.load_table(io.BytesIO("n,v\ncafé,1\n".encode("cp1252")), filename="t.csv")
        self.assertEqual(df.iloc[0, 0], "café")

    def test_excel_and_errors(self):
        buf = io.BytesIO()
        pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_excel(buf, index=False)
        df = D.load_table(buf, filename="t.xlsx")
        self.assertEqual(df.shape, (2, 2))
        with self.assertRaises(D.DataLoadError):
            D.load_table(io.BytesIO(b"   "), filename="t.csv")

    def test_profile(self):
        p = D.profile(fixture())
        self.assertEqual(set(p.columns), {"column", "dtype", "missing", "missing_pct", "unique", "examples"})
        self.assertEqual(int(p.set_index("column").loc["f_a", "missing"]), 30)


class SchemaTests(unittest.TestCase):
    def test_norm_value_and_target_types(self):
        self.assertEqual(norm_value(1), norm_value(1.0))
        self.assertEqual(norm_value(True), norm_value("True"))
        df = pd.DataFrame({"t": [1, 0, 1.0, np.nan], "x": [1, 2, 3, 4]})
        y = make_target(df, Schema("t", ["1"], ["x"]))
        self.assertEqual(y.tolist()[:3], [1, 0, 1])
        self.assertTrue(np.isnan(y.iloc[3]))

    def test_validation_messages(self):
        df = fixture()
        self.assertTrue(validate_schema(df, Schema("nope", ["left"], ["f_a"]))[0])
        e, _ = validate_schema(df, Schema("outcome", ["missing_value"], ["f_a"]))
        self.assertTrue(any("never appear" in m for m in e))
        e, _ = validate_schema(df, Schema("outcome", ["left", "stayed"], ["f_a"]))
        self.assertTrue(any("no non-churn" in m for m in e))
        e, _ = validate_schema(df, Schema("outcome", ["left"], ["outcome", "f_a"]))
        self.assertTrue(any("cannot also be a feature" in m for m in e))
        self.assertEqual(validate_schema(df, SCHEMA)[0], [])


class PreprocessTests(unittest.TestCase):
    def test_spec_types_and_exclusions(self):
        df = pd.DataFrame({"n": [1.0, 2, 3, 4], "t": ["1", "2", "3", "4"], "c": list("abab"),
                           "k": [5, 5, 5, 5], "e": [np.nan] * 4, "d": pd.date_range("2024-01-01", periods=4),
                           "u": [f"v{i}" for i in range(4)]})
        spec = infer_feature_spec(df, list(df.columns), max_categories=3)
        self.assertEqual(sorted(spec.numeric), ["n", "t"])
        self.assertEqual(spec.categorical, ["c"])
        self.assertEqual(set(spec.excluded), {"k", "e", "d", "u"})

    def test_origins_match_output_width_and_unseen_category(self):
        df = fixture()
        spec = infer_feature_spec(df, ["f_a", "f_b", "f_grp"])
        prep = build_preprocessor(spec).fit(prepare_frame(df, spec))
        X = prep.transform(prepare_frame(df, spec))
        self.assertEqual(X.shape[1], len(feature_origins(prep, spec)))
        new = df.head(3).copy()
        new["f_grp"] = "never_seen"
        self.assertFalse(np.isnan(prep.transform(prepare_frame(new, spec))).any())

    def test_missing_column_message(self):
        spec = infer_feature_spec(fixture(), ["f_a"])
        with self.assertRaises(ValueError):
            prepare_frame(pd.DataFrame({"zzz": [1]}), spec)


class TrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = fixture()
        cls.res = train_and_evaluate(cls.df, SCHEMA, model_names=["Logistic Regression", "Decision Tree", "Random Forest"])

    def test_metrics_and_split(self):
        m = self.res.metrics
        self.assertEqual(set(m.index), {"Logistic Regression", "Decision Tree", "Random Forest"})
        self.assertGreater(m.loc["Logistic Regression", "roc_auc"], 0.7)   # fixture has built-in signal
        s = self.res.split
        self.assertEqual(s["train"] + s["test"], s["rows_used"])
        self.assertAlmostEqual(s["churn_rate_train"], s["churn_rate_test"], delta=0.03)
        self.assertIn(self.res.best_name, m.index)

    def test_default_probabilities_are_calibrated(self):
        # without class balancing, average predicted risk should match the real churn rate
        p = self.res.test_proba["Logistic Regression"]
        self.assertAlmostEqual(float(p.mean()), float(self.res.y_test.mean()), delta=0.05)

    def test_missing_target_rows_are_dropped(self):
        d = self.df.copy()
        d.loc[:19, "outcome"] = None
        r = train_and_evaluate(d, SCHEMA, model_names=["Logistic Regression"])
        self.assertEqual(r.split["rows_used"], len(d) - 20)

    def test_too_few_examples_is_a_clear_error(self):
        d = self.df.head(40).copy()
        d["outcome"] = "stayed"
        d.loc[:1, "outcome"] = "left"
        with self.assertRaises(ValueError):
            train_and_evaluate(d, SCHEMA)

    def test_grouped_split_keeps_ids_together(self):
        d = pd.concat([self.df, self.df.assign(f_a=self.df.f_a + 0.01)], ignore_index=True)
        r = train_and_evaluate(d, SCHEMA, model_names=["Logistic Regression"])
        self.assertTrue(r.split["grouped_by_id"])

    def test_leakage_screen_flags_outcome_copy(self):
        d = self.df.copy()
        d["leaky"] = (d["outcome"] == "left").astype(int) * 5 + np.random.default_rng(1).normal(0, 0.01, len(d))
        sch = Schema("outcome", ["left"], ["f_a", "f_b", "leaky"], "cust")
        r = train_and_evaluate(d, sch, model_names=["Logistic Regression"])
        self.assertIn("leaky", [c for c, _ in r.leakage])
        self.assertTrue(any("leakage" in w.lower() for w in r.warnings))

    def test_metrics_helpers(self):
        m = metrics_at([0, 1, 1, 0], [0.1, 0.9, 0.6, 0.4])
        self.assertEqual(m["accuracy"], 1.0)
        cal = calibration_table(self.res.y_test, self.res.test_proba["Logistic Regression"], bins=5)
        self.assertEqual(int(cal["customers"].sum()), len(self.res.y_test))


class RiskTests(unittest.TestCase):
    def test_bands(self):
        self.assertEqual((RiskBands().low_max, RiskBands().high_min), (0.15, 0.30))
        b = RiskBands(0.30, 0.70)
        self.assertEqual(list(b.assign([0.0, 0.299, 0.30, 0.699, 0.70, 1.0])),
                         ["Low", "Low", "Medium", "Medium", "High", "High"])
        self.assertEqual(int(band_summary([0.1, 0.5, 0.9, 0.95], b).set_index("risk_level").loc["High", "customers"]), 2)
        with self.assertRaises(ValueError):
            RiskBands(0.8, 0.3)


class ExplainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = fixture()
        cls.res = train_and_evaluate(cls.df, SCHEMA, model_names=["Logistic Regression"])

    def test_linear_contributions_add_up_to_the_score(self):
        pipe = self.res.pipelines["Logistic Regression"]
        rows = self.df.head(25)
        c = explain_rows(pipe, self.res.spec, rows, self.res.background)
        self.assertEqual(set(c.columns), {"f_a", "f_b", "f_grp"})   # one-hot pieces summed back
        prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
        Xb = prep.transform(prepare_frame(self.res.background, self.res.spec))
        base = clf.intercept_[0] + (Xb.mean(axis=0) * clf.coef_[0]).sum()
        logit = pipe.decision_function(prepare_frame(rows, self.res.spec))
        np.testing.assert_allclose(c.sum(axis=1).values + base, logit, rtol=1e-6, atol=1e-8)

    def test_shape_normalisation_and_aggregation(self):
        a = np.arange(12, dtype=float).reshape(2, 3, 2)
        np.testing.assert_array_equal(positive_class_values(a), a[:, :, 1])
        np.testing.assert_array_equal(positive_class_values([a[:, :, 0], a[:, :, 1]]), a[:, :, 1])
        agg = aggregate_by_origin(np.array([[1.0, 2.0, 3.0]]), ["x", "y", "x"])
        self.assertEqual(agg.loc[0, "x"], 4.0)

    def test_top_drivers_only_positive_sorted(self):
        d = top_drivers(pd.Series({"a": 0.1, "b": -0.9, "c": 0.5}), k=5)
        self.assertEqual([f for f, _ in d], ["c", "a"])


class RuleTests(unittest.TestCase):
    ROW = pd.Series({"f_a": 2.0, "f_grp": "g3"})
    DRV = [("f_a", 0.9), ("f_grp", 0.3)]

    def rules(self):
        return parse_rules({"rules": [
            {"id": "r1", "action": "Do X", "risk_levels": ["High"], "driver": "f_a", "priority": 1,
             "conditions": [{"feature": "f_a", "op": ">", "value": 1}]},
            {"id": "r2", "action": "Do Y", "driver": "f_grp", "conditions": [{"feature": "f_grp", "op": "in", "value": ["g3"]}]},
            {"id": "r3", "action": "Never", "risk_levels": ["Low"]}]})

    def test_matching_order_and_fallback(self):
        out = recommend(self.rules(), "High", self.DRV, self.ROW)
        self.assertEqual([a["rule_id"] for a in out["actions"]], ["r1", "r2"])
        none = recommend([], "High", self.DRV, self.ROW)
        self.assertFalse(none["matched"])
        self.assertIn("No configured rule matched", none["summary"])

    def test_max_actions_caps_output(self):
        rules = parse_rules({"rules": [{"id": f"r{i}", "action": f"A{i}", "priority": i} for i in range(5)]})
        out = recommend(rules, "High", self.DRV, self.ROW, max_actions=2)
        self.assertEqual([a["rule_id"] for a in out["actions"]], ["r4", "r3"])
        self.assertIn("3 more rule(s)", out["summary"])

    def test_missing_value_and_level_filtering(self):
        out = recommend(self.rules(), "Medium", self.DRV, pd.Series({"f_a": np.nan, "f_grp": "g3"}))
        self.assertEqual([a["rule_id"] for a in out["actions"]], ["r2"])

    def test_validation(self):
        bad = parse_rules({"rules": [{"id": "x", "action": "", "driver": "nope", "risk_levels": ["Huge"],
                                     "conditions": [{"feature": "zzz", "op": "~", "value": 1}]}]})
        self.assertGreaterEqual(len(validate_rules(bad, ["f_a"])), 5)
        self.assertEqual(validate_rules(self.rules(), ["f_a", "f_grp"]), [])


class ProjectConfigTests(unittest.TestCase):
    """The shipped rules and dataset config must only name real columns (from data_descriptions.csv)."""
    COLUMNS = ["AccountAge", "MonthlyCharges", "SubscriptionType", "PaymentMethod", "PaperlessBilling",
               "ContentType", "MultiDeviceAccess", "DeviceRegistered", "ViewingHoursPerWeek",
               "AverageViewingDuration", "ContentDownloadsPerMonth", "GenrePreference", "UserRating",
               "SupportTicketsPerMonth", "Gender", "WatchlistSize", "ParentalControl", "SubtitlesEnabled"]
    CONFIG = Path(__file__).parent.parent / "config"

    def test_shipped_rules_are_valid(self):
        rules = load_rules(self.CONFIG / "rules.json")
        self.assertGreater(len(rules), 0)
        self.assertEqual(validate_rules(rules, self.COLUMNS), [])

    def test_dataset_config_names_real_columns(self):
        cfg = json.loads((self.CONFIG / "dataset.json").read_text(encoding="utf-8"))
        allcols = self.COLUMNS + ["TotalCharges", "CustomerID", "Churn"]
        for key in ("target", "id_column"):
            self.assertIn(cfg[key], allcols)
        self.assertTrue(set(cfg["exclude"]) <= set(allcols))


class EndToEndTests(unittest.TestCase):
    def test_cli_train_then_score_explain_recommend(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            fixture().to_csv(t / "d.csv", index=False)
            (t / "rules.json").write_text(json.dumps({"rules": [{"id": "r", "action": "Act", "driver": "f_a"}]}))
            self.assertEqual(cli_main(["inspect", str(t / "d.csv")]), 0)
            self.assertEqual(cli_main(["train", str(t / "d.csv"), "--target", "outcome", "--positive", "left",
                                       "--id", "cust", "--out", str(t / "m.joblib")]), 0)
            b = load_bundle(t / "m.joblib")
            df = pd.read_csv(t / "d.csv")
            sc = score_table(b, df)
            self.assertEqual(len(sc), 600)
            self.assertTrue(sc["churn_probability"].between(0, 1).all())
            self.assertTrue(set(sc["risk_level"]) <= {"Low", "Medium", "High"})
            if b.model_name == "Logistic Regression":   # explanation without SHAP installed
                contrib = explain_rows(b.pipeline, b.spec, df.iloc[[3]], b.background).iloc[0]
                drivers = top_drivers(contrib, 5)
                rec = recommend(load_rules(t / "rules.json"), sc.iloc[3]["risk_level"], drivers, df.iloc[3])
                self.assertIn("summary", rec)

    def test_bundle_roundtrip_with_custom_bands(self):
        res = train_and_evaluate(fixture(), SCHEMA, model_names=["Logistic Regression"])
        with tempfile.TemporaryDirectory() as t:
            save_bundle(Path(t) / "m.joblib", make_bundle(res, bands=RiskBands(0.2, 0.6)))
            b = load_bundle(Path(t) / "m.joblib")
        self.assertEqual((b.bands.low_max, b.bands.high_min), (0.2, 0.6))

    def test_cli_reports_bad_target_cleanly(self):
        with tempfile.TemporaryDirectory() as t:
            fixture().to_csv(Path(t) / "d.csv", index=False)
            self.assertEqual(cli_main(["train", str(Path(t) / "d.csv"), "--target", "zzz", "--positive", "1"]), 2)


if __name__ == "__main__":
    unittest.main()
