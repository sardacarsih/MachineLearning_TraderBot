import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

from config.mt5_config import MT5Config, apply_mt5_config_from_yaml, mt5_config
from config.settings import config
from mt5 import account_manager as account_manager_module
from mt5 import connector as connector_module
from mt5 import order_executor as order_executor_module


class MT5ConfigSingletonTests(unittest.TestCase):
    def setUp(self):
        self._original = {
            cfg_field.name: getattr(mt5_config, cfg_field.name)
            for cfg_field in fields(MT5Config)
        }
        self._original_max_risk_per_trade = config.risk.max_risk_per_trade
        self._original_high_confidence_threshold = config.risk.high_confidence_threshold
        self._original_high_confidence_lot_multiplier = config.risk.high_confidence_lot_multiplier

    def tearDown(self):
        for name, value in self._original.items():
            setattr(mt5_config, name, value)
        config.risk.max_risk_per_trade = self._original_max_risk_per_trade
        config.risk.high_confidence_threshold = self._original_high_confidence_threshold
        config.risk.high_confidence_lot_multiplier = self._original_high_confidence_lot_multiplier

    def test_yaml_is_applied_to_shared_singleton(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "mt5.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "mt5:",
                        "  login: 123456",
                        '  password: "secret"',
                        '  server: "Demo-Server"',
                        '  terminal_path: "C:/MT5/terminal64.exe"',
                        "  magic_number: 777",
                        "  paper_trading: false",
                        "  trading_enabled: true",
                        "  safety_max_lot: 0.5",
                        '  account_currency: "IDR"',
                        '  risk_quote_currency: "USD"',
                        '  usd_idr_symbol_candidates: ["USDIDR", "USDIDRm"]',
                        "  usd_idr_fallback_rate: 16400",
                        "  margin_reserve_pct: 0.2",
                        "risk:",
                        "  max_risk_per_trade: 0.005",
                        "  high_confidence_threshold: 0.72",
                        "  high_confidence_lot_multiplier: 2.5",
                    ]
                ),
                encoding="utf-8",
            )

            applied = apply_mt5_config_from_yaml(str(config_path))

        self.assertIs(applied, mt5_config)
        self.assertIs(order_executor_module.mt5_config, mt5_config)
        self.assertIs(account_manager_module.mt5_config, mt5_config)
        self.assertIs(connector_module.mt5_config, mt5_config)
        self.assertFalse(order_executor_module.mt5_config.paper_trading)
        self.assertTrue(account_manager_module.mt5_config.trading_enabled)
        self.assertEqual(connector_module.mt5_config.login, 123456)
        self.assertEqual(mt5_config.magic_number, 777)
        self.assertEqual(mt5_config.safety_max_lot, 0.5)
        self.assertEqual(mt5_config.account_currency, "IDR")
        self.assertEqual(mt5_config.risk_quote_currency, "USD")
        self.assertEqual(mt5_config.usd_idr_symbol_candidates, ["USDIDR", "USDIDRm"])
        self.assertEqual(mt5_config.usd_idr_fallback_rate, 16400)
        self.assertEqual(mt5_config.margin_reserve_pct, 0.2)
        self.assertEqual(config.risk.max_risk_per_trade, 0.005)
        self.assertEqual(config.risk.high_confidence_threshold, 0.72)
        self.assertEqual(config.risk.high_confidence_lot_multiplier, 2.5)


if __name__ == "__main__":
    unittest.main()
