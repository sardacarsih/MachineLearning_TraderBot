import unittest
from dataclasses import fields
from types import SimpleNamespace

from config.mt5_config import MT5Config, mt5_config
from config.settings import config
from mt5.currency import CurrencyConverter
from strategy.risk_manager import RiskManager


class FakeConnector:
    def __init__(self, connected=True, account_currency="IDR", ticks=None):
        self.connected = connected
        self._account = SimpleNamespace(currency=account_currency)
        self._ticks = ticks or {}
        self._mt5 = SimpleNamespace(symbol_info_tick=lambda symbol: self._ticks.get(symbol))

    def is_connected(self):
        return self.connected

    def get_account_info(self):
        return self._account


class CurrencyRiskTests(unittest.TestCase):
    def setUp(self):
        self._mt5_original = {cfg_field.name: getattr(mt5_config, cfg_field.name) for cfg_field in fields(MT5Config)}
        self._risk_pct = config.risk.max_risk_per_trade
        self._contract_size = config.symbol.contract_size
        self._min_lot = config.symbol.min_lot
        self._max_lot = config.symbol.max_lot
        self._lot_step = config.symbol.lot_step
        self._high_confidence_threshold = config.risk.high_confidence_threshold
        self._high_confidence_lot_multiplier = config.risk.high_confidence_lot_multiplier

        mt5_config.account_currency = "AUTO"
        mt5_config.risk_quote_currency = "USD"
        mt5_config.usd_idr_symbol_candidates = ["USDIDR", "USDIDRm", "USDIDR."]
        mt5_config.usd_idr_rate_mode = "broker"
        mt5_config.usd_idr_manual_rate = 15000.0
        mt5_config.usd_idr_fallback_rate = 16400.0
        config.risk.max_risk_per_trade = 0.01
        config.symbol.contract_size = 100.0
        config.symbol.min_lot = 0.01
        config.symbol.max_lot = 100.0
        config.symbol.lot_step = 0.01
        config.risk.high_confidence_threshold = 0.70
        config.risk.high_confidence_lot_multiplier = 2.0

    def tearDown(self):
        for name, value in self._mt5_original.items():
            setattr(mt5_config, name, value)
        config.risk.max_risk_per_trade = self._risk_pct
        config.symbol.contract_size = self._contract_size
        config.symbol.min_lot = self._min_lot
        config.symbol.max_lot = self._max_lot
        config.symbol.lot_step = self._lot_step
        config.risk.high_confidence_threshold = self._high_confidence_threshold
        config.risk.high_confidence_lot_multiplier = self._high_confidence_lot_multiplier

    def test_idr_to_usd_conversion_uses_tick_mid(self):
        converter = CurrencyConverter(
            FakeConnector(ticks={"USDIDR": SimpleNamespace(bid=16390.0, ask=16410.0)})
        )

        amount, account_ccy, quote_ccy, rate = converter.convert_risk_amount(100000.0)

        self.assertEqual(account_ccy, "IDR")
        self.assertEqual(quote_ccy, "USD")
        self.assertAlmostEqual(rate, 16400.0)
        self.assertAlmostEqual(amount, 6.09756, places=5)

    def test_usd_account_stays_unchanged(self):
        converter = CurrencyConverter(FakeConnector(account_currency="USD"))

        amount, account_ccy, quote_ccy, rate = converter.convert_risk_amount(100.0)

        self.assertEqual(account_ccy, "USD")
        self.assertEqual(quote_ccy, "USD")
        self.assertIsNone(rate)
        self.assertEqual(amount, 100.0)

    def test_fallback_rate_is_used_when_quote_unavailable(self):
        converter = CurrencyConverter(FakeConnector(ticks={}))

        amount, _, _, rate = converter.convert_risk_amount(100000.0)

        self.assertEqual(rate, 16400.0)
        self.assertAlmostEqual(amount, 6.09756, places=5)

    def test_manual_rate_overrides_broker_quote(self):
        mt5_config.usd_idr_rate_mode = "manual"
        mt5_config.usd_idr_manual_rate = 15000.0
        converter = CurrencyConverter(
            FakeConnector(ticks={"USDIDR": SimpleNamespace(bid=17800.0, ask=17900.0)})
        )

        amount, account_ccy, quote_ccy, rate = converter.convert_risk_amount(100000.0)

        self.assertEqual(account_ccy, "IDR")
        self.assertEqual(quote_ccy, "USD")
        self.assertEqual(rate, 15000.0)
        self.assertAlmostEqual(amount, 6.66667, places=5)

    def test_idr_balance_sizes_xauusd_from_usd_risk_budget(self):
        risk_manager = RiskManager()
        converter = CurrencyConverter(FakeConnector(ticks={"USDIDR": SimpleNamespace(bid=16400.0, ask=16400.0)}))
        risk_amount_quote, account_ccy, quote_ccy, rate = converter.convert_risk_amount(100000.0)

        lot = risk_manager.calculate_position_size(
            10_000_000.0,
            7.089,
            risk_amount_quote=risk_amount_quote,
            account_currency=account_ccy,
            risk_quote_currency=quote_ccy,
            conversion_rate=rate,
        )

        self.assertEqual(lot, 0.01)
        self.assertLess(lot, 100.0)

    def test_position_size_respects_symbol_max_lot(self):
        risk_manager = RiskManager()
        config.symbol.max_lot = 1.0

        lot = risk_manager.calculate_position_size(10_000_000.0, 7.089)

        self.assertEqual(lot, 1.0)

    def test_confidence_lot_multiplier_is_strictly_above_threshold(self):
        risk_manager = RiskManager()

        lot = risk_manager.apply_confidence_lot_multiplier(0.10, 0.70)

        self.assertEqual(lot, 0.10)

    def test_confidence_lot_multiplier_doubles_high_confidence_lot(self):
        risk_manager = RiskManager()

        lot = risk_manager.apply_confidence_lot_multiplier(0.10, 0.71)

        self.assertEqual(lot, 0.20)

    def test_confidence_lot_multiplier_respects_step_and_max_lot(self):
        risk_manager = RiskManager()
        config.symbol.max_lot = 0.21
        config.symbol.lot_step = 0.05

        lot = risk_manager.apply_confidence_lot_multiplier(0.13, 0.71)

        self.assertEqual(lot, 0.21)


if __name__ == "__main__":
    unittest.main()
