from copy import deepcopy
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
        self._original_model_confidence_threshold = config.model.confidence_threshold
        self._original_high_confidence_threshold = config.risk.high_confidence_threshold
        self._original_high_confidence_lot_multiplier = config.risk.high_confidence_lot_multiplier
        self._original_consecutive_loss_cooldown_enabled = config.risk.consecutive_loss_cooldown_enabled
        self._original_consecutive_loss_cooldown_count = config.risk.consecutive_loss_cooldown_count
        self._original_consecutive_loss_cooldown_hours = config.risk.consecutive_loss_cooldown_hours
        self._original_symbol = config.symbol.symbol
        self._original_timeframe = config.symbol.timeframe
        self._original_mt5_timeframe = config.symbol.mt5_timeframe
        self._original_confidence = deepcopy(config.confidence)

    def tearDown(self):
        for name, value in self._original.items():
            setattr(mt5_config, name, value)
        config.risk.max_risk_per_trade = self._original_max_risk_per_trade
        config.model.confidence_threshold = self._original_model_confidence_threshold
        config.risk.high_confidence_threshold = self._original_high_confidence_threshold
        config.risk.high_confidence_lot_multiplier = self._original_high_confidence_lot_multiplier
        config.risk.consecutive_loss_cooldown_enabled = self._original_consecutive_loss_cooldown_enabled
        config.risk.consecutive_loss_cooldown_count = self._original_consecutive_loss_cooldown_count
        config.risk.consecutive_loss_cooldown_hours = self._original_consecutive_loss_cooldown_hours
        config.symbol.symbol = self._original_symbol
        config.symbol.timeframe = self._original_timeframe
        config.symbol.mt5_timeframe = self._original_mt5_timeframe
        config.confidence = self._original_confidence

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
                        "  consecutive_loss_cooldown_enabled: false",
                        "  consecutive_loss_cooldown_count: 4",
                        "  consecutive_loss_cooldown_hours: 6.5",
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
        self.assertFalse(config.risk.consecutive_loss_cooldown_enabled)
        self.assertEqual(config.risk.consecutive_loss_cooldown_count, 4)
        self.assertEqual(config.risk.consecutive_loss_cooldown_hours, 6.5)

    def test_consecutive_loss_cooldown_enabled_defaults_to_true_without_yaml_field(self):
        config.risk.consecutive_loss_cooldown_enabled = True

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "mt5.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "mt5:",
                        "  login: 123456",
                        "risk:",
                        "  max_risk_per_trade: 0.005",
                    ]
                ),
                encoding="utf-8",
            )

            apply_mt5_config_from_yaml(str(config_path))

        self.assertTrue(config.risk.consecutive_loss_cooldown_enabled)

    def test_confidence_resolver_uses_symbol_timeframe_precedence(self):
        config.model.confidence_threshold = 0.50
        config.risk.high_confidence_threshold = 0.70
        config.risk.high_confidence_lot_multiplier = 2.0
        config.confidence.load_from_mapping({
            "default": {
                "signal_threshold": 0.51,
                "high_confidence_threshold": 0.71,
                "high_confidence_lot_multiplier": 1.1,
            },
            "by_timeframe": {
                "M5": {
                    "signal_threshold": 0.52,
                    "high_confidence_threshold": 0.72,
                    "high_confidence_lot_multiplier": 1.2,
                },
            },
            "by_symbol": {
                "XAUUSD": {
                    "signal_threshold": 0.53,
                    "high_confidence_threshold": 0.73,
                    "high_confidence_lot_multiplier": 1.3,
                },
            },
            "by_symbol_timeframe": {
                "XAUUSD": {
                    "M5": {
                        "signal_threshold": 0.54,
                        "high_confidence_threshold": 0.74,
                        "high_confidence_lot_multiplier": 1.4,
                    },
                },
            },
        })

        effective = config.resolve_confidence("xauusd", "5")

        self.assertEqual(effective.signal_threshold, 0.54)
        self.assertEqual(effective.high_confidence_threshold, 0.74)
        self.assertEqual(effective.high_confidence_lot_multiplier, 1.4)

    def test_confidence_resolver_falls_back_by_level(self):
        config.model.confidence_threshold = 0.50
        config.risk.high_confidence_threshold = 0.70
        config.risk.high_confidence_lot_multiplier = 2.0
        config.confidence.load_from_mapping({
            "default": {"signal_threshold": 0.51},
            "by_timeframe": {"M15": {"signal_threshold": 0.52}},
            "by_symbol": {"XAGUSD": {"high_confidence_threshold": 0.73}},
        })

        timeframe_only = config.resolve_confidence("GBPJPY", "M15")
        symbol_only = config.resolve_confidence("XAGUSD", "M1")
        global_only = config.resolve_confidence("EURUSD", "H1")

        self.assertEqual(timeframe_only.signal_threshold, 0.52)
        self.assertEqual(timeframe_only.high_confidence_threshold, 0.70)
        self.assertEqual(symbol_only.signal_threshold, 0.51)
        self.assertEqual(symbol_only.high_confidence_threshold, 0.73)
        self.assertEqual(global_only.signal_threshold, 0.51)
        self.assertEqual(global_only.high_confidence_lot_multiplier, 2.0)

    def test_confidence_resolver_uses_global_config_without_overrides(self):
        config.model.confidence_threshold = 0.57
        config.risk.high_confidence_threshold = 0.77
        config.risk.high_confidence_lot_multiplier = 1.8
        config.confidence.clear()

        effective = config.resolve_confidence("EURUSD", "H1")

        self.assertEqual(effective.signal_threshold, 0.57)
        self.assertEqual(effective.high_confidence_threshold, 0.77)
        self.assertEqual(effective.high_confidence_lot_multiplier, 1.8)

    def test_yaml_applies_confidence_overrides(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "mt5.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "mt5:",
                        "  login: 123456",
                        "confidence:",
                        "  by_symbol_timeframe:",
                        "    XAUUSD:",
                        "      M5:",
                        "        signal_threshold: 0.61",
                        "        high_confidence_threshold: 0.76",
                        "        high_confidence_lot_multiplier: 1.7",
                    ]
                ),
                encoding="utf-8",
            )

            apply_mt5_config_from_yaml(str(config_path))

        effective = config.resolve_confidence("XAUUSD", "M5")
        self.assertEqual(effective.signal_threshold, 0.61)
        self.assertEqual(effective.high_confidence_threshold, 0.76)
        self.assertEqual(effective.high_confidence_lot_multiplier, 1.7)

    def test_yaml_rejects_invalid_confidence_timeframe(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "mt5.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "mt5:",
                        "  login: 123456",
                        "confidence:",
                        "  by_timeframe:",
                        "    BAD:",
                        "      signal_threshold: 0.61",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Unsupported timeframe"):
                apply_mt5_config_from_yaml(str(config_path))


if __name__ == "__main__":
    unittest.main()
