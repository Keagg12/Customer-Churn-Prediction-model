"""Tests for the dashboard logic (churn/workspace.py, churn/batch.py).
The table below is a synthetic FIXTURE that borrows the real column names so the shipped rules can run on it.
It says nothing about the real dataset, and no result here is a project result."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from churn.batch import recommend_batch, top_factor_mask, top_factor_names
from churn.explain import explain_rows, top_drivers
from churn.recommend import load_rules, recommend
from churn.scoring import RiskBands
from churn import workspace as W

CONFIG = Path(__file__).parent.parent / "config"


def fixture(n=1500, seed=3):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "CustomerID": [f"C{i:05d}" for i in range(n)],
        "AccountAge": rng.integers(1, 120, n),
        "MonthlyCharges": rng.uniform(4, 20, n).round(2),
        "SubscriptionType": rng.choice(["Basic", "Standard", "Premium"], n),
        "ViewingHoursPerWeek": rng.uniform(1, 40, n).round(1),
        "AverageViewingDuration": rng.uniform(5, 160, n).round(1),
        "ContentDownloadsPerMonth": rng.integers(0, 50, n),
        "SupportTicketsPerMonth": rng.integers(0, 10, n),
        "UserRating": rng.uniform(1, 5, n).round(2),
    })
    d["TotalCharges"] = d["AccountAge"] * d["MonthlyCharges"]
    logit = (-1.0 - 0.02 * d["AccountAge"] + 0.25 * d["SupportTicketsPerMonth"] + 0.06 * d["MonthlyCharges"]
             - 0.03 * d["ViewingHoursPerWeek"] + (d["SubscriptionType"] == "Basic") * 0.6)
    d["Churn"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return d


CFG = {"target": "Churn", "positive_labels": ["1"], "id_column": "CustomerID", "exclude": ["TotalCharges"]}


class ConfigTests(unittest.TestCase):
    def test_schema_from_config(self):
        df = fixture(50)
        s = W.schema_from_config(df, CFG)
        self.assertEqual((s.target, s.id_column), ("Churn", "CustomerID"))
        self.assertNotIn("TotalCharges", s.feature_columns)
        self.assertNotIn("Churn", s.feature_columns)
        self.assertIsNone(W.schema_from_config(df.drop(columns="Churn"), CFG))
        self.assertIsNone(W.schema_from_config(df, {}))

    def test_shipped_dataset_config_reads(self):
        cfg = W.load_dataset_config(CONFIG / "dataset.json")
        self.assertEqual(cfg["target"], "Churn")
        self.assertEqual(W.load_dataset_config(CONFIG / "nope.json"), {})

    def test_pretty(self):
        self.assertEqual(W.pretty("SupportTicketsPerMonth"), "Support tickets per month")
        self.assertEqual(W.pretty("AccountAge"), "Account age")
        self.assertEqual(W.pretty("snake_case_name"), "Snake case name")

    def test_descriptions(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "d.csv"
            p.write_text("Column_name,Column_type,Data_type,Description\nA,Feature,integer,Age\n", encoding="utf-8")
            self.assertEqual(W.load_descriptions(p), {"A": "Age"})
            self.assertEqual(W.load_descriptions(Path(t) / "none.csv"), {})


class PatternTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = fixture()
        cls.schema = W.schema_from_config(cls.df, CFG)
        cls.pat = W.churn_patterns(cls.df, cls.schema)

    def test_rates_match_the_data(self):
        g = self.pat[self.pat.feature == "SubscriptionType"].set_index("group")
        for name in ("Basic", "Standard", "Premium"):
            sub = self.df[self.df.SubscriptionType == name]
            self.assertAlmostEqual(g.loc[name, "churn_rate"], sub.Churn.mean(), places=9)
            self.assertEqual(g.loc[name, "customers"], len(sub))
        base = self.df.Churn.mean()
        self.assertAlmostEqual(g.loc["Basic", "lift"], g.loc["Basic", "churn_rate"] / base, places=9)

    def test_every_group_set_covers_all_customers_and_id_is_skipped(self):
        for f, grp in self.pat.groupby("feature"):
            self.assertEqual(int(grp.customers.sum()), len(self.df), f)
        self.assertNotIn("CustomerID", set(self.pat.feature))
        self.assertNotIn("TotalCharges", set(self.pat.feature))

    def test_strength_orders_by_spread_and_sentences(self):
        s = W.pattern_strength(self.pat)
        self.assertTrue((s.spread.diff().dropna() <= 1e-12).all())
        self.assertIn(s.feature.iloc[0], {"AccountAge", "SupportTicketsPerMonth", "ViewingHoursPerWeek",
                                          "MonthlyCharges", "SubscriptionType"})
        text = W.insight_sentences(s, float(self.df.Churn.mean()), k=2)
        self.assertEqual(len(text), 2)
        self.assertIn("overall", text[0])


class EvaluationViewTests(unittest.TestCase):
    def test_views(self):
        rng = np.random.default_rng(0)
        p = rng.random(2000)
        y = (rng.random(2000) < p).astype(int)
        b = RiskBands(0.3, 0.6)
        lift = W.band_lift(y, p, b).set_index("risk_level")
        self.assertEqual(int(lift.customers.sum()), 2000)
        self.assertAlmostEqual(lift.share_of_churners.sum(), 1.0, places=9)
        self.assertGreater(lift.loc["High", "observed_churn_rate"], lift.loc["Low", "observed_churn_rate"])
        cm = W.confusion_at(y, p, 0.6)
        self.assertEqual(sum(cm.values()), 2000)
        self.assertEqual(cm["tp"] + cm["fn"], int(y.sum()))
        top = W.top_fraction_check(y, p, 0.1)
        self.assertEqual(top["customers"], 200)
        self.assertGreater(top["observed_churn_rate"], y.mean())
        roc = W.roc_points(y, {"m": p})
        self.assertEqual(set(roc.model), {"m"})
        self.assertTrue(roc.fpr.between(0, 1).all() and roc.tpr.between(0, 1).all())
        hist = W.probability_histogram(p, b)
        self.assertEqual(int(hist.customers.sum()), 2000)
        self.assertEqual(set(hist.risk_level), {"Low", "Medium", "High"})


class BatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = fixture(900)
        cls.schema = W.schema_from_config(cls.df, CFG)
        cls.pkg = W.train_package(cls.df, cls.schema, {"name": "fixture"}, model_names=["Logistic Regression"])
        cls.rules = load_rules(CONFIG / "rules.json")

    def test_batch_matches_single_customer_functions(self):
        b = self.pkg.bundle
        rows = self.df.iloc[:300]
        contrib = explain_rows(b.pipeline, b.spec, rows, b.background)
        proba = W.predict_proba(b, rows)
        levels = b.bands.assign(proba)
        out = recommend_batch(self.rules, levels, contrib, rows)
        self.assertEqual(len(out), 300)
        matched_any = 0
        for i in range(len(rows)):
            drivers = top_drivers(contrib.iloc[i], 5)
            single = recommend(self.rules, levels[i], drivers, rows.iloc[i])
            self.assertEqual(out["action_ids"].iloc[i], [a["rule_id"] for a in single["actions"]], i)
            self.assertEqual(out["top_factors"].iloc[i], [f for f, _ in drivers[:3]], i)
            matched_any += bool(single["matched"])
        self.assertGreater(matched_any, 0)      # the comparison is not vacuous

    def test_top_factor_helpers(self):
        c = pd.DataFrame({"a": [0.5, -1.0], "b": [0.2, -0.1], "c": [-0.3, 0.4]})
        self.assertEqual(top_factor_names(c, 2), [["a", "b"], ["c"]])
        m = top_factor_mask(c, 1)
        self.assertEqual(m.values.tolist(), [[True, False, False], [False, False, True]])

    def test_length_mismatch_is_an_error(self):
        with self.assertRaises(ValueError):
            recommend_batch(self.rules, ["High"], pd.DataFrame({"a": [1.0, 2.0]}), self.df.iloc[:1])

    def test_global_importance_sums_to_one(self):
        imp = W.global_importance(self.pkg.bundle, self.df)
        self.assertAlmostEqual(float(imp.sum()), 1.0, places=9)
        self.assertTrue((imp.diff().dropna() <= 1e-12).all())


class PackageTests(unittest.TestCase):
    def test_save_load_stale_and_checks(self):
        df = fixture(700)
        schema = W.schema_from_config(df, CFG)
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            csv = t / "train.csv"
            df.to_csv(csv, index=False)
            src = W.file_fingerprint(csv)
            pkg = W.train_package(df, schema, src, model_names=["Logistic Regression"])
            self.assertEqual(pkg.evaluation.best_name, "Logistic Regression")
            self.assertEqual(set(pkg.evaluation.test_proba), {"Logistic Regression"})
            self.assertEqual(len(pkg.evaluation.y_test), len(pkg.evaluation.test_proba["Logistic Regression"]))
            self.assertEqual((pkg.bundle.bands.low_max, pkg.bundle.bands.high_min), (0.15, 0.30))
            self.assertFalse(pkg.evaluation.balance)
            path = t / "models" / "m.joblib"
            W.save_package(path, pkg)
            back, why = W.load_package(path)
            self.assertEqual(why, "")
            self.assertEqual(back.key, pkg.key)
            # the command line loader still reads the same file
            from churn.scoring import load_bundle
            self.assertEqual(load_bundle(path).model_name, "Logistic Regression")
            self.assertFalse(W.is_stale(back, csv))
            df.head(600).to_csv(csv, index=False)
            self.assertTrue(W.is_stale(back, csv))
            self.assertIsNone(W.load_package(t / "missing.joblib")[0])
            bad = t / "bad.joblib"
            bad.write_bytes(b"not a pickle")
            self.assertIsNone(W.load_package(bad)[0])
            # a bundle written without an evaluation (command line tool) is reported, not trusted
            from churn.scoring import save_bundle
            save_bundle(t / "cli.joblib", pkg.bundle)
            self.assertIn("no evaluation", W.load_package(t / "cli.joblib")[1])

            checks = W.input_checks(pkg.bundle, df.drop(columns=["Churn"]))
            self.assertEqual(checks["missing_columns"], [])
            self.assertEqual(W.input_checks(pkg.bundle, df[["CustomerID"]])["missing_columns"] != [], True)
            odd = df.head(5).copy()
            odd.loc[0, "SubscriptionType"] = "Platinum"
            odd.loc[1, "MonthlyCharges"] = np.nan
            odd["AccountAge"] = odd["AccountAge"].astype(object)
            odd.loc[2, "AccountAge"] = "n/a"
            c2 = W.input_checks(pkg.bundle, odd)
            self.assertEqual(c2["unseen_categories"], {"SubscriptionType": ["Platinum"]})
            self.assertEqual(c2["missing_values"].get("MonthlyCharges"), 1)
            self.assertEqual(c2["non_numeric"].get("AccountAge"), 1)

    def test_customer_ids(self):
        df = pd.DataFrame({"id": [7, 8]})
        self.assertEqual(W.customer_ids(df, "id").tolist(), ["7", "8"])
        self.assertEqual(W.customer_ids(df, None).tolist(), ["row 1", "row 2"])


if __name__ == "__main__":
    unittest.main()
