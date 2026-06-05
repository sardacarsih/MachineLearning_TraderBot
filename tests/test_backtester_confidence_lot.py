from copy import deepcopy
import unittest

import pandas as pd

from backtest.backtester import Backtester
from config.settings import config


class BacktesterConfidenceLotTests(unittest.TestCase):
    def setUp(self):
        self._risk_pct = config.risk.max_risk_per_trade
        self._atr_sl_multiplier = config.risk.atr_sl_multiplier
        self._high_confidence_threshold = config.risk.high_confidence_threshold
        self._high_confidence_lot_multiplier = config.risk.high_confidence_lot_multiplier
        self._confidence_config = deepcopy(config.confidence)
        self._max_open_positions = config.risk.max_open_positions
        self._spread_filter_enabled = config.filters.spread_filter_enabled
        self._volatility_filter_enabled = config.filters.volatility_filter_enabled
        self._ranging_filter_enabled = config.filters.ranging_filter_enabled
        self._point = config.symbol.point
        self._digits = config.symbol.digits
        self._contract_size = config.symbol.contract_size
        self._min_lot = config.symbol.min_lot
        self._max_lot = config.symbol.max_lot
        self._lot_step = config.symbol.lot_step

        config.risk.max_risk_per_trade = 0.01
        config.risk.atr_sl_multiplier = 1.0
        config.risk.high_confidence_threshold = 0.70
        config.risk.high_confidence_lot_multiplier = 2.0
        config.confidence.clear()
        config.risk.max_open_positions = 1
        config.filters.spread_filter_enabled = True
        config.filters.volatility_filter_enabled = True
        config.filters.ranging_filter_enabled = True
        config.symbol.point = 0.01
        config.symbol.digits = 2
        config.symbol.contract_size = 100.0
        config.symbol.min_lot = 0.01
        config.symbol.max_lot = 100.0
        config.symbol.lot_step = 0.01

    def tearDown(self):
        config.risk.max_risk_per_trade = self._risk_pct
        config.risk.atr_sl_multiplier = self._atr_sl_multiplier
        config.risk.high_confidence_threshold = self._high_confidence_threshold
        config.risk.high_confidence_lot_multiplier = self._high_confidence_lot_multiplier
        config.confidence = self._confidence_config
        config.risk.max_open_positions = self._max_open_positions
        config.filters.spread_filter_enabled = self._spread_filter_enabled
        config.filters.volatility_filter_enabled = self._volatility_filter_enabled
        config.filters.ranging_filter_enabled = self._ranging_filter_enabled
        config.symbol.point = self._point
        config.symbol.digits = self._digits
        config.symbol.contract_size = self._contract_size
        config.symbol.min_lot = self._min_lot
        config.symbol.max_lot = self._max_lot
        config.symbol.lot_step = self._lot_step

    def make_df(self):
        return pd.DataFrame(
            [
                {
                    "time": pd.Timestamp("2026-01-01 00:00:00"),
                    "open": 100.0,
                    "high": 100.5,
                    "low": 99.5,
                    "close": 100.0,
                    "atr": 1.0,
                    "spread": 1.0,
                    "adx": 30.0,
                },
                {
                    "time": pd.Timestamp("2026-01-01 00:05:00"),
                    "open": 100.0,
                    "high": 100.5,
                    "low": 99.5,
                    "close": 100.0,
                    "atr": 1.0,
                    "spread": 1.0,
                    "adx": 30.0,
                },
            ]
        )

    def run_single_trade(self, confidence):
        backtester = Backtester(
            initial_balance=1000.0,
            commission_per_lot=0.0,
            slippage_points=0.0,
            strategy_mode="ml",
        )
        result = backtester.run(
            self.make_df(),
            ["BUY", "NO_TRADE"],
            confidences=[confidence, 0.0],
        )
        return result.trade_log[0]["volume"]

    def test_high_confidence_backtest_trade_uses_double_lot(self):
        normal_lot = self.run_single_trade(0.70)
        high_confidence_lot = self.run_single_trade(0.71)

        self.assertEqual(normal_lot, 0.10)
        self.assertEqual(high_confidence_lot, 0.20)


if __name__ == "__main__":
    unittest.main()
