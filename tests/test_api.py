"""API contract tests on synthetic fixtures; these are not project model results."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.service import DashboardService
from churn import workspace as W


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        (cls.root / "data").mkdir()
        (cls.root / "config").mkdir()
        (cls.root / "models").mkdir()
        config = {"target": "churn", "positive_labels": ["1"], "id_column": "customer_id", "exclude": []}
        (cls.root / "config" / "dataset.json").write_text(json.dumps(config), encoding="utf-8")
        (cls.root / "config" / "rules.json").write_text(json.dumps({"rules": [
            {"id": "support", "action": "Contact the customer", "risk_levels": ["Medium", "High"],
             "driver": "tickets", "conditions": [{"feature": "tickets", "op": ">=", "value": 4}], "priority": 2}
        ]}), encoding="utf-8")
        rng = np.random.default_rng(42)
        rows = 500
        age = rng.integers(1, 120, rows)
        tickets = rng.integers(0, 8, rows)
        segment = rng.choice(["basic", "plus"], rows)
        probability = 1 / (1 + np.exp(-(-1 + 0.02 * tickets - 0.008 * age + (segment == "basic") * 0.3)))
        churn = (rng.random(rows) < probability).astype(int)
        cls.frame = pd.DataFrame({"customer_id": [f"T{i:04d}" for i in range(rows)], "age": age,
                                  "tickets": tickets, "segment": segment, "churn": churn})
        cls.frame.to_csv(cls.root / "data" / "train.csv", index=False)
        cls.frame.drop(columns="churn").head(35).to_csv(cls.root / "data" / "test.csv", index=False)
        schema = W.schema_from_config(cls.frame, config)
        package = W.train_package(cls.frame, schema, W.file_fingerprint(cls.root / "data" / "train.csv"),
                                  model_names=["Logistic Regression"])
        W.save_package(cls.root / "models" / "model.joblib", package)
        cls.service = DashboardService(cls.root, auto_train=False)
        cls.client_context = TestClient(create_app(cls.service))
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)
        cls.temp.cleanup()

    def test_auto_training_is_disabled_by_default(self):
        service = DashboardService(self.root, auto_train=None)
        self.assertFalse(service.auto_train)

    def test_dashboard_routes_return_real_fixture_data(self):
        headers = {"X-Session-ID": "api-test-session"}
        for path in ("/api/health", "/api/bootstrap", "/api/overview", "/api/customers?limit=5",
                     "/api/retention?risk_levels=High&limit=5", "/api/model-insights", "/api/workspace/status"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path, headers=headers).status_code, 200)
        rows = self.client.get("/api/customers?limit=1", headers=headers).json()["items"]
        detail = self.client.get(f"/api/customers/{rows[0]['customer_id']}", headers=headers)
        self.assertEqual(detail.status_code, 200)
        self.assertGreaterEqual(detail.json()["probability"], 0)
        self.assertIn("explanation", detail.json())

    def test_filters_bands_and_sessions_are_isolated(self):
        first = {"X-Session-ID": "session-one"}
        second = {"X-Session-ID": "session-two"}
        changed = self.client.post("/api/workspace/bands", headers=first,
                                   json={"low_max": 0.2, "high_min": 0.6})
        self.assertEqual(changed.status_code, 200)
        self.assertEqual(self.client.get("/api/bootstrap", headers=first).json()["bands"]["high_min"], 0.6)
        self.assertEqual(self.client.get("/api/bootstrap", headers=second).json()["bands"]["high_min"], 0.3)
        invalid = self.client.get("/api/customers?risk_levels=Urgent", headers=first)
        self.assertEqual(invalid.status_code, 422)

    def test_upload_score_and_full_csv_download(self):
        headers = {"X-Session-ID": "score-upload-session"}
        raw = self.frame.drop(columns="churn").head(7).to_csv(index=False).encode()
        response = self.client.post("/api/workspace/score-upload", headers=headers,
                                    files={"file": ("scoring.csv", raw, "text/csv")})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["rows"], 7)
        download = self.client.get("/api/workspace/scores.csv", headers=headers)
        self.assertEqual(download.status_code, 200)
        self.assertIn("churn_probability", download.text.splitlines()[0])
        self.assertEqual(len(download.text.splitlines()), 8)

    def test_upload_training_keeps_model_session_only_by_default(self):
        headers = {"X-Session-ID": "train-upload-session"}
        raw = self.frame.to_csv(index=False).encode()
        uploaded = self.client.post("/api/workspace/train-upload", headers=headers,
                                    files={"file": ("labeled.csv", raw, "text/csv")})
        self.assertEqual(uploaded.status_code, 200, uploaded.text)
        result = self.client.post("/api/workspace/train", headers=headers, json={
            "target": "churn",
            "positive_labels": ["1"],
            "feature_columns": ["age", "tickets", "segment"],
            "id_column": "customer_id",
            "test_size": 0.2,
            "seed": 42,
            "balance": False,
            "persist": False,
            "model_name": "Logistic Regression",
        })
        self.assertEqual(result.status_code, 200, result.text)
        self.assertFalse(result.json()["saved"])
        self.assertEqual(self.service.pkg.bundle.model_name, "Logistic Regression")
        status = self.client.get("/api/workspace/status", headers=headers).json()
        self.assertEqual(status["model_name"], "Logistic Regression")
        self.assertTrue(status["session_has_train_upload"])


if __name__ == "__main__":
    unittest.main()
