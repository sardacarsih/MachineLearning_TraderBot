import tempfile
import unittest
from pathlib import Path

from utils.live_trade_analysis import build_analysis, parse_trade_logs


class LiveTradeAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="live-analysis-")
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write_log(self, name, content):
        path = self.root / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_trade_log_parser_supports_open_failed_skipped_and_close_events(self):
        path = self.write_log(
            "trades_20260604.log",
            "\n".join(
                [
                    "2026-06-04 09:55:00.865 | INFO     | [TRADE OPENED] Ticket: 3218627623 | USTEC_x100 SELL 0.01 lots at 30407.98 (Requested: 30407.98, Slippage: 0.0 pts) | SL: 30451.12, TP: 30349.67",
                    "2026-06-04 09:55:00.900 | INFO     | [SIGNAL TRADE LINK] Ticket: 3218627623 | Action: SELL | Confidence: 0.6200 | SignalTime: 2026-06-04 09:50:00+00:00 | Entry: 30407.98000 | SL: 30451.12000 | TP: 30349.67000",
                    "2026-06-04 09:56:00.000 | INFO     | [TRADE FAILED] XAUUSD SELL 0.06 lots | Reason: Invalid filling type",
                    "2026-06-04 09:57:00.000 | INFO     | [TRADE SKIPPED] XAUUSD BUY | Reason: Max open positions blocked trade for XAUUSD: 3 open >= limit 3",
                    "2026-06-04 10:00:00.000 | INFO     | [TRADE CLOSED] Ticket: 3218627623 | USTEC_x100 closed at 30349.67 (Entry: 30407.98, Profit: 12.50)",
                    "2026-06-04 10:05:00.000 | INFO     | [PAPER TRADE CLOSE] Ticket: 100000 | XAUUSD BUY closed at 4451.72 (Entry: 4448.74) | Gross PnL: 2.98, Net PnL: 2.78 (Comm: 0.20) | ExitReason: TP",
                ]
            ),
        )

        parsed = parse_trade_logs([path])

        self.assertEqual(len(parsed.opened), 1)
        self.assertEqual(len(parsed.signal_trade_links), 1)
        self.assertEqual(len(parsed.failed), 1)
        self.assertEqual(len(parsed.skipped), 1)
        self.assertEqual(len(parsed.closed), 2)
        self.assertEqual(parsed.signal_trade_links[0]["confidence"], 0.62)
        self.assertEqual(parsed.closed[0]["pnl"], 12.50)
        self.assertEqual(parsed.closed[1]["pnl"], 2.78)
        self.assertEqual(parsed.closed[1]["close_reason"], "TP")
        self.assertEqual(parsed.unparsed_trade_lines, 0)

    def test_build_analysis_summarizes_signals_execution_pnl_and_backtest(self):
        self.write_log(
            "signals_20260604.log",
            "\n".join(
                [
                    "2026-06-04 07:40:06.360 | INFO     | Signal Generated: NO_TRADE | Confidence: 0.4011 | Probas: [NO_TRADE: 0.2734, BUY: 0.3255, SELL: 0.4011]",
                    "2026-06-04 07:45:06.360 | INFO     | Signal Generated: BUY | Confidence: 0.6011 | Probas: [NO_TRADE: 0.1, BUY: 0.6, SELL: 0.3]",
                ]
            ),
        )
        self.write_log(
            "trades_20260604.log",
            "\n".join(
                [
                    "2026-06-04 07:46:00.000 | INFO     | [TRADE OPENED] Ticket: 1 | XAUUSD BUY 0.01 lots at 4448.74 (Requested: 4448.74, Slippage: 0.0 pts) | SL: 4443.04, TP: 4451.72",
                    "2026-06-04 07:50:00.000 | INFO     | [TRADE CLOSED] Ticket: 1 | XAUUSD closed at 4451.72 (Entry: 4448.74, Profit: 2.98)",
                    "2026-06-04 07:51:00.000 | INFO     | [TRADE SKIPPED] XAUUSD BUY | Reason: No lot >= broker min 0.01 passed margin/order checks",
                ]
            ),
        )
        backtest = self.write_log(
            "performance_report.txt",
            "\n".join(
                [
                    "Net Profit:                 $100.00",
                    "Total Trades:               10",
                    "Winning Trades:             6 (60.00%)",
                    "Profit Factor:              1.50",
                    "Trade Expectancy:           $10.00",
                    "Max Drawdown (%):           5.00%",
                ]
            ),
        )

        analysis = build_analysis(
            symbol="XAUUSD",
            timeframe="M5",
            log_dir=self.root,
            start_date="20260604",
            end_date="20260604",
            backtest_report_path=backtest,
            paper_db_path=self.root / "missing.db",
        )

        self.assertEqual(analysis["signals"]["total"], 2)
        self.assertEqual(analysis["signals"]["buy"], 1)
        self.assertEqual(analysis["execution"]["opened"], 1)
        self.assertEqual(analysis["execution"]["skipped"], 1)
        self.assertEqual(analysis["performance"]["total_closed"], 1)
        self.assertEqual(analysis["performance"]["net_pnl"], 2.98)
        self.assertEqual(analysis["backtest"]["profit_factor"], 1.5)
        self.assertEqual(analysis["decision"]["status"], "PAUSE_REVIEW")

    def test_build_analysis_summarizes_confidence_by_tp_sl_outcome(self):
        self.write_log(
            "trades_20260604.log",
            "\n".join(
                [
                    "2026-06-04 07:46:00.000 | INFO     | [TRADE OPENED] Ticket: 1 | XAUUSD BUY 0.01 lots at 4448.74 (Requested: 4448.74, Slippage: 0.0 pts) | SL: 4443.04, TP: 4451.72",
                    "2026-06-04 07:46:00.001 | INFO     | [SIGNAL TRADE LINK] Ticket: 1 | Action: BUY | Confidence: 0.6200 | SignalTime: 2026-06-04 07:45:00+00:00 | Entry: 4448.74000 | SL: 4443.04000 | TP: 4451.72000",
                    "2026-06-04 07:55:00.000 | INFO     | [TRADE OPENED] Ticket: 2 | XAUUSD BUY 0.01 lots at 4444.00 (Requested: 4444.00, Slippage: 0.0 pts) | SL: 4440.00, TP: 4450.00",
                    "2026-06-04 07:55:00.001 | INFO     | [SIGNAL TRADE LINK] Ticket: 2 | Action: BUY | Confidence: 0.4800 | SignalTime: 2026-06-04 07:54:00+00:00 | Entry: 4444.00000 | SL: 4440.00000 | TP: 4450.00000",
                ]
            ),
        )

        analysis = build_analysis(
            symbol="XAUUSD",
            timeframe="M1",
            log_dir=self.root,
            start_date="20260604",
            end_date="20260604",
            paper_db_path=self.root / "missing.db",
            mt5_deals=[
                {
                    "ticket": 101,
                    "order": 1,
                    "position_id": 1,
                    "time": "2026-06-04 07:50:00",
                    "symbol": "XAUUSD",
                    "type": "SELL",
                    "entry": "OUT",
                    "profit": 2.0,
                    "commission": 0.0,
                    "swap": 0.0,
                    "reason_label": "TP",
                    "reason": 0,
                },
                {
                    "ticket": 102,
                    "order": 2,
                    "position_id": 2,
                    "time": "2026-06-04 08:00:00",
                    "symbol": "XAUUSD",
                    "type": "SELL",
                    "entry": "OUT",
                    "profit": -1.0,
                    "commission": 0.0,
                    "swap": 0.0,
                    "reason_label": "SL",
                    "reason": 0,
                },
            ],
        )

        outcomes = analysis["confidence_outcomes"]
        self.assertEqual(outcomes["tp_count"], 1)
        self.assertEqual(outcomes["sl_count"], 1)
        self.assertEqual(outcomes["win_count"], 1)
        self.assertEqual(outcomes["loss_count"], 1)
        self.assertEqual(outcomes["avg_confidence_tp"], 0.62)
        self.assertEqual(outcomes["avg_confidence_sl"], 0.48)
        self.assertEqual(outcomes["avg_confidence_win"], 0.62)
        self.assertEqual(outcomes["avg_confidence_loss"], 0.48)
        thresholds = {item["threshold"]: item for item in outcomes["thresholds"]}
        self.assertEqual(thresholds[0.5]["trades"], 1)
        self.assertEqual(thresholds[0.5]["closed"], 1)
        self.assertEqual(thresholds[0.5]["wins"], 1)
        self.assertEqual(thresholds[0.5]["losses"], 0)
        self.assertEqual(thresholds[0.5]["winrate_pct"], 100.0)
        self.assertEqual(thresholds[0.6]["trades"], 1)
        self.assertEqual(thresholds[0.6]["net_pnl"], 2.0)


if __name__ == "__main__":
    unittest.main()
