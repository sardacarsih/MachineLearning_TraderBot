from copy import deepcopy
import unittest
from dataclasses import fields
from datetime import datetime, timedelta
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
        self._confidence_config = deepcopy(config.confidence)
        self._symbol_name = config.symbol.symbol
        self._timeframe = config.symbol.timeframe
        self._mt5_timeframe = config.symbol.mt5_timeframe
        self._consecutive_loss_cooldown_enabled = config.risk.consecutive_loss_cooldown_enabled
        self._consecutive_loss_cooldown_count = config.risk.consecutive_loss_cooldown_count
        self._consecutive_loss_cooldown_hours = config.risk.consecutive_loss_cooldown_hours

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
        config.confidence.clear()
        config.symbol.symbol = "XAUUSD"
        config.symbol.timeframe = "M5"
        config.symbol.mt5_timeframe = 5
        config.risk.consecutive_loss_cooldown_enabled = True
        config.risk.consecutive_loss_cooldown_count = 3
        config.risk.consecutive_loss_cooldown_hours = 4.0

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
        config.confidence = self._confidence_config
        config.symbol.symbol = self._symbol_name
        config.symbol.timeframe = self._timeframe
        config.symbol.mt5_timeframe = self._mt5_timeframe
        config.risk.consecutive_loss_cooldown_enabled = self._consecutive_loss_cooldown_enabled
        config.risk.consecutive_loss_cooldown_count = self._consecutive_loss_cooldown_count
        config.risk.consecutive_loss_cooldown_hours = self._consecutive_loss_cooldown_hours

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

    def test_confidence_lot_multiplier_uses_symbol_timeframe_override(self):
        risk_manager = RiskManager()
        config.confidence.load_from_mapping({
            "by_symbol_timeframe": {
                "XAUUSD": {
                    "M5": {
                        "high_confidence_threshold": 0.80,
                        "high_confidence_lot_multiplier": 3.0,
                    },
                },
            },
        })

        below_override = risk_manager.apply_confidence_lot_multiplier(0.10, 0.79)
        above_override = risk_manager.apply_confidence_lot_multiplier(0.10, 0.81)

        self.assertEqual(below_override, 0.10)
        self.assertEqual(above_override, 0.30)

    def test_three_net_losses_same_magic_starts_four_hour_cooldown(self):
        risk_manager = RiskManager()
        now = datetime(2026, 6, 4, 8, 0, 0)

        risk_manager.update_trade_result(-1.0, magic_number=777, trade_id="a", closed_at=now)
        risk_manager.update_trade_result(-2.0, magic_number=777, trade_id="b", closed_at=now)
        risk_manager.update_trade_result(-3.0, magic_number=777, trade_id="c", closed_at=now)

        self.assertFalse(risk_manager.can_trade(10_000.0, magic_number=777, now=now + timedelta(hours=3)))
        self.assertIn("Consecutive-loss cooldown active", risk_manager.last_block_reason)
        self.assertTrue(risk_manager.can_trade(10_000.0, magic_number=777, now=now + timedelta(hours=4, minutes=1)))

    def test_disabled_consecutive_loss_cooldown_does_not_block_trading(self):
        risk_manager = RiskManager()
        config.risk.consecutive_loss_cooldown_enabled = False
        now = datetime(2026, 6, 4, 8, 0, 0)

        risk_manager.update_trade_result(-1.0, magic_number=777, trade_id="a", closed_at=now)
        risk_manager.update_trade_result(-2.0, magic_number=777, trade_id="b", closed_at=now)
        risk_manager.update_trade_result(-3.0, magic_number=777, trade_id="c", closed_at=now)

        self.assertTrue(risk_manager.can_trade(10_000.0, magic_number=777, now=now + timedelta(minutes=1)))
        self.assertEqual(risk_manager.daily_pnl, -6.0)
        self.assertEqual(risk_manager.daily_trade_count, 3)
        self.assertEqual(risk_manager._cooldown_state(777)["consecutive_losses"], 3)

    def test_disabled_consecutive_loss_cooldown_still_syncs_closed_trade_pnl(self):
        risk_manager = RiskManager()
        config.risk.consecutive_loss_cooldown_enabled = False
        config.risk.consecutive_loss_cooldown_count = 2
        now = datetime(2026, 6, 4, 8, 0, 0)
        trades = [
            {"ticket": 1, "order": 1, "position_id": 1, "time": "2026-06-04 08:00:00", "entry": "OUT", "profit": -1, "commission": 0, "swap": 0, "magic": 777},
            {"ticket": 2, "order": 2, "position_id": 2, "time": "2026-06-04 08:01:00", "entry": "OUT", "profit": -2, "commission": -0.5, "swap": 0, "magic": 777},
        ]

        self.assertEqual(risk_manager.sync_closed_trades(trades, magic_number=777), 2)
        self.assertEqual(risk_manager.daily_pnl, -3.5)
        self.assertEqual(risk_manager.daily_trade_count, 2)
        self.assertTrue(risk_manager.can_trade(10_000.0, magic_number=777, now=now))

    def test_win_or_breakeven_resets_magic_loss_streak(self):
        risk_manager = RiskManager()
        now = datetime(2026, 6, 4, 8, 0, 0)

        risk_manager.update_trade_result(-1.0, magic_number=777, trade_id="a", closed_at=now)
        risk_manager.update_trade_result(0.0, magic_number=777, trade_id="b", closed_at=now)
        risk_manager.update_trade_result(-1.0, magic_number=777, trade_id="c", closed_at=now)
        risk_manager.update_trade_result(-1.0, magic_number=777, trade_id="d", closed_at=now)

        self.assertTrue(risk_manager.can_trade(10_000.0, magic_number=777, now=now + timedelta(minutes=1)))

    def test_loss_streaks_are_isolated_by_magic_number(self):
        risk_manager = RiskManager()
        now = datetime(2026, 6, 4, 8, 0, 0)

        risk_manager.update_trade_result(-1.0, magic_number=111, trade_id="a", closed_at=now)
        risk_manager.update_trade_result(-1.0, magic_number=111, trade_id="b", closed_at=now)
        risk_manager.update_trade_result(-1.0, magic_number=222, trade_id="c", closed_at=now)

        self.assertTrue(risk_manager.can_trade(10_000.0, magic_number=111, now=now + timedelta(minutes=1)))
        self.assertTrue(risk_manager.can_trade(10_000.0, magic_number=222, now=now + timedelta(minutes=1)))

    def test_sync_closed_trades_uses_net_pnl_and_deduplicates_deals(self):
        risk_manager = RiskManager()
        config.risk.consecutive_loss_cooldown_count = 2
        now = datetime(2026, 6, 4, 8, 0, 0)
        trades = [
            {"ticket": 1, "order": 1, "position_id": 1, "time": "2026-06-04 08:00:00", "entry": "IN", "profit": -50, "commission": 0, "swap": 0, "magic": 777},
            {"ticket": 2, "order": 2, "position_id": 2, "time": "2026-06-04 08:01:00", "entry": "OUT", "profit": -1, "commission": 0, "swap": 0, "magic": 888},
            {"ticket": 3, "order": 3, "position_id": 3, "time": "2026-06-04 08:02:00", "entry": "OUT", "profit": 1, "commission": -2, "swap": 0, "magic": 777},
            {"ticket": 4, "order": 4, "position_id": 4, "time": "2026-06-04 08:03:00", "entry": "OUT", "profit": -1, "commission": 0, "swap": 0, "magic": 777},
        ]

        self.assertEqual(risk_manager.sync_closed_trades(trades, magic_number=777), 2)
        self.assertEqual(risk_manager.sync_closed_trades(trades, magic_number=777), 0)
        self.assertFalse(risk_manager.can_trade(10_000.0, magic_number=777, now=now))
        self.assertTrue(risk_manager.can_trade(10_000.0, magic_number=888, now=now))


if __name__ == "__main__":
    unittest.main()
