import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from loguru import logger

from utils.live_dashboard import (
    RichLiveDashboard,
    derive_timeframe_trends,
    format_money,
    normalize_position,
    pnl_style,
    status_style,
)
from utils.logger import TradingLogger, setup_logging
from scripts.live_trade import MAX_ORDER_COMMENT_LEN, format_order_comment


class LiveDashboardFormattingTests(unittest.TestCase):
    def test_status_colors_cover_trading_states(self):
        self.assertEqual(status_style("CONNECTED"), "bold green")
        self.assertEqual(status_style("BUY"), "bold green")
        self.assertEqual(status_style("SELL"), "bold red")
        self.assertEqual(status_style("NO_TRADE"), "bold yellow")
        self.assertEqual(status_style("WAITING"), "bold cyan")

    def test_pnl_and_money_formatting(self):
        self.assertEqual(format_money(1234.5), "1,234.50")
        self.assertEqual(format_money("bad"), "N/A")
        self.assertEqual(pnl_style(10), "green")
        self.assertEqual(pnl_style(-1), "red")
        self.assertEqual(pnl_style(0), "white")

    def test_position_normalization_supports_dict_and_mt5_like_objects(self):
        paper_position = {
            "ticket": 100,
            "type": "BUY",
            "volume": 0.1,
            "open_price": 2000.0,
            "sl": 1990.0,
            "tp": 2020.0,
        }
        live_position = SimpleNamespace(
            ticket=101,
            type=1,
            volume=0.2,
            price_open=2010.0,
            sl=2000.0,
            tp=2030.0,
            profit=-2.5,
        )

        self.assertEqual(normalize_position(paper_position)["type"], "BUY")
        normalized_live = normalize_position(live_position)
        self.assertEqual(normalized_live["type"], "SELL")
        self.assertEqual(normalized_live["open_price"], 2010.0)
        self.assertEqual(normalized_live["profit"], -2.5)

    def test_trend_derivation_uses_runtime_base_timeframe_and_htf_features(self):
        trends = derive_timeframe_trends(
            {
                "ema_alignment_bullish": 1,
                "trend_strength": 28.5,
                "htf_m15_breakout_position": 0.8,
                "htf_m15_breakout_high": 0,
                "htf_m15_breakout_low": 0,
                "htf_h1_ema_alignment_bullish": 0,
                "htf_h1_ema_alignment_bearish": 1,
                "htf_h1_dist_ema_200": -0.42,
                "htf_h4_atr_regime_ratio": 1.4,
                "htf_h4_atr_regime_high": 1,
                "htf_h4_atr_regime_low": 0,
            },
            base_timeframe="M1",
        )

        self.assertEqual(trends["M1"]["label"], "BULL")
        self.assertNotIn("M5", trends)
        self.assertEqual(trends["M15"]["label"], "BULL")
        self.assertEqual(trends["H1"]["label"], "BEAR")
        self.assertEqual(trends["H4"]["label"], "VOL HIGH")


class RichLiveDashboardStateTests(unittest.TestCase):
    def test_runtime_state_updates_for_key_signal_modes(self):
        dashboard = RichLiveDashboard("XAUUSD", "M5", "LIVE", "hybrid", htf_enabled=True)
        self.assertTrue(dashboard.state.htf_enabled)

        dashboard.update_connection(True, heartbeat_ok=True)
        dashboard.update_account(
            {
                "balance": 10_000,
                "equity": 10_025,
                "free_margin": 9_500,
                "margin_level": 250,
                "daily_pnl": 25,
                "trade_allowed": True,
            }
        )
        dashboard.update_market(SimpleNamespace(bid=2330.12, ask=2330.25), spread_points=13.0, cached_atr=1.2345)
        dashboard.update_candle("2026-06-03 00:30:00+00:00")
        dashboard.update_trends({"ema_alignment_bearish": 1, "htf_h1_ema_alignment_bullish": 1})
        self.assertEqual(dashboard.state.timeframe_trends["M5"]["label"], "BEAR")
        self.assertEqual(dashboard.state.timeframe_trends["H1"]["label"], "BULL")
        dashboard.update_positions([
            {"ticket": 1, "type": "BUY", "volume": 0.1, "open_price": 2320.0, "sl": 2310.0, "tp": 2340.0}
        ])

        for action in ("BUY", "SELL", "NO_TRADE"):
            dashboard.update_signal({"action": action, "confidence": 0.55, "probabilities": [0.2, 0.55, 0.25]})
            self.assertEqual(dashboard.state.last_signal["action"], action)
            self.assertIsNotNone(dashboard.render())

        dashboard.update_connection(False, heartbeat_ok=False)
        self.assertEqual(dashboard.state.connection, "DISCONNECTED")
        self.assertEqual(dashboard.state.heartbeat, "FAILED")

    def test_signal_panel_renders_signal_time_and_timeframe_metadata(self):
        dashboard = RichLiveDashboard("XAUUSD", "M15", "LIVE", "ml")
        dashboard.update_candle("2026-06-03 12:00:00")
        dashboard.update_trends({"ema_alignment_bullish": 1, "htf_m15_breakout_low": 1})
        dashboard.update_signal(
            {
                "action": "SELL",
                "confidence": 0.65,
                "probabilities": [0.2, 0.15, 0.65],
                "signal_time": "2026-06-03 12:00:00",
                "timeframe": "M15",
            }
        )

        self.assertEqual(dashboard.state.last_signal["signal_time"], "2026-06-03 19:00:00 WIB")
        self.assertEqual(dashboard.state.last_signal["timeframe"], "M15")
        self.assertEqual(dashboard.state.timeframe_trends["M15"]["label"], "BULL")
        self.assertNotEqual(dashboard.state.timeframe_trends["M15"]["label"], "BEAR BRK")
        self.assertIsNotNone(dashboard.render())

    def test_order_comment_preserves_timeframe_prefix(self):
        comment = format_order_comment("M15", "hybrid", "VeryLongModelNameThatWouldExceedBrokerLimit")

        self.assertLessEqual(len(comment), MAX_ORDER_COMMENT_LEN)
        self.assertTrue(comment.startswith("M15_HYBRID_"))


class LoggerQuietConsoleTests(unittest.TestCase):
    def tearDown(self):
        logger.remove()
        TradingLogger._initialized = False
        TradingLogger._console_handler_id = None
        TradingLogger._file_handler_ids = []

    def test_quiet_console_keeps_file_handlers_without_stdout_handler(self):
        log_dir = Path(tempfile.mkdtemp(prefix="rich-dashboard-logs-"))
        setup_logging(log_dir=str(log_dir), debug=False, quiet_console=True)

        self.assertIsNone(TradingLogger._console_handler_id)
        self.assertEqual(len(TradingLogger._file_handler_ids), 4)

        logger.info("main event")
        TradingLogger.signal_log("signal event")
        TradingLogger.trade_log("trade event")
        logger.error("error event")
        logger.complete()

        self.assertTrue(list(log_dir.glob("main_*.log")))
        self.assertTrue(list(log_dir.glob("signals_*.log")))
        self.assertTrue(list(log_dir.glob("trades_*.log")))
        self.assertTrue(list(log_dir.glob("errors_*.log")))


if __name__ == "__main__":
    unittest.main()
