import tempfile
import unittest
from pathlib import Path

import dashboard


class DashboardTimeframeContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="dashboard-timeframe-")
        self.root = Path(self.tmp.name)
        self.originals = {
            "BACKTEST_ROOT": dashboard.BACKTEST_ROOT,
            "LOGS_ROOT": dashboard.LOGS_ROOT,
            "SAVED_MODELS_ROOT": dashboard.SAVED_MODELS_ROOT,
            "get_dashboard_python": dashboard.get_dashboard_python,
        }
        dashboard.BACKTEST_ROOT = self.root / "backtest"
        dashboard.LOGS_ROOT = self.root / "logs"
        dashboard.SAVED_MODELS_ROOT = self.root / "saved_models"
        dashboard.get_dashboard_python = lambda: "python.exe"
        dashboard.app.config.update(TESTING=True)
        self.client = dashboard.app.test_client()

    def tearDown(self):
        for name, value in self.originals.items():
            setattr(dashboard, name, value)
        self.tmp.cleanup()

    def write_file(self, path, content="x"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_saved_models_endpoint_is_strict_to_selected_timeframe(self):
        self.write_file(dashboard.SAVED_MODELS_ROOT / "XAUUSD" / "M1" / "selected_catboost_model")
        self.write_file(dashboard.SAVED_MODELS_ROOT / "XAUUSD" / "M5" / "selected_xgboost_model")

        response = self.client.get("/api/saved-models?symbol=XAUUSD&timeframe=M1")
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["context"]["timeframe"], "M1")
        self.assertEqual(len(payload["models"]), 1)
        self.assertEqual(payload["models"][0]["value"], "XAUUSD/M1/selected_catboost_model")
        self.assertIn("XAUUSD\\M1", payload["source_dir"])

    def test_signal_logs_endpoint_reads_only_selected_timeframe_folder(self):
        self.write_file(
            dashboard.LOGS_ROOT / "XAUUSD" / "M1" / "signals_20260603.log",
            "M1 signal\n",
        )
        self.write_file(
            dashboard.LOGS_ROOT / "XAUUSD" / "M5" / "signals_20260603.log",
            "M5 signal\n",
        )

        response = self.client.get("/api/logs/signals?symbol=XAUUSD&timeframe=M1")
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["lines"], ["M1 signal"])
        self.assertIn("XAUUSD\\M1", payload["source_dir"])

    def test_control_command_uses_selected_timeframe(self):
        self.write_file(dashboard.SAVED_MODELS_ROOT / "XAUUSD" / "M1" / "selected_catboost_model")

        cmd, ctx = dashboard.build_control_command(
            {
                "action": "paper",
                "symbol": "XAUUSD",
                "timeframe": "M1",
                "model": "XAUUSD/M1/selected_catboost_model",
                "strategy_mode": "ml",
            }
        )

        self.assertEqual(ctx["timeframe"], "M1")
        self.assertIn("--timeframe", cmd)
        self.assertEqual(cmd[cmd.index("--timeframe") + 1], "M1")

    def test_control_command_rejects_model_from_different_timeframe(self):
        self.write_file(dashboard.SAVED_MODELS_ROOT / "XAUUSD" / "M5" / "selected_catboost_model")

        with self.assertRaisesRegex(ValueError, "selected timeframe"):
            dashboard.build_control_command(
                {
                    "action": "paper",
                    "symbol": "XAUUSD",
                    "timeframe": "M1",
                    "model": "XAUUSD/M5/selected_catboost_model",
                    "strategy_mode": "ml",
                }
            )


if __name__ == "__main__":
    unittest.main()
