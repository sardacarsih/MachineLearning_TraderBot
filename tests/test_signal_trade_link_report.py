import tempfile
import unittest
from pathlib import Path

from scripts.report_signal_trade_links import (
    build_arg_parser,
    build_link_details,
    parse_signal_trade_links,
    summarize_details_by_day,
    summarize_details,
    total_daily_summary,
    total_summary,
)


class SignalTradeLinkReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="signal-link-report-")
        self.root = Path(self.tmp.name)
        self.logs = self.root / "logs"

    def tearDown(self):
        self.tmp.cleanup()

    def write_log(self, relative_path, content):
        path = self.logs / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_parser_deduplicates_main_and_trade_logs(self):
        line = (
            "2026-06-05 10:15:00.431 | INFO     | [SIGNAL TRADE LINK] "
            "Ticket: 4163787993 | Action: SELL | Confidence: 0.6022 | "
            "SignalTime: 2026-06-05 03:10:00+00:00 | Entry: 30204.06000 | "
            "SL: 30239.51000 | TP: 30150.89000"
        )
        self.write_log("USTEC_X100/M5/main_20260605.log", line)
        self.write_log("USTEC_X100/M5/trades_20260605.log", line)

        links = parse_signal_trade_links(self.logs, start_date="20260605", end_date="20260605")

        self.assertEqual(len(links), 1)
        self.assertEqual(links[0].ticket, 4163787993)
        self.assertEqual(links[0].symbol, "USTEC_X100")
        self.assertEqual(links[0].timeframe, "M5")
        self.assertEqual(links[0].confidence, 0.6022)

    def test_details_match_close_deals_by_position_id_and_flag_unclosed(self):
        self.write_log(
            "XAUUSD/M1/main_20260605.log",
            "\n".join(
                [
                    "2026-06-05 05:17:02.480 | INFO     | [SIGNAL TRADE LINK] Ticket: 1 | Action: BUY | Confidence: 0.5056 | SignalTime: 2026-06-04 22:16:00+00:00 | Entry: 4475.38600 | SL: 4473.70500 | TP: 4477.90800",
                    "2026-06-05 05:24:07.601 | INFO     | [SIGNAL TRADE LINK] Ticket: 2 | Action: BUY | Confidence: 0.5065 | SignalTime: 2026-06-04 22:23:00+00:00 | Entry: 4478.50200 | SL: 4476.61200 | TP: 4481.33700",
                    "2026-06-05 05:31:02.006 | INFO     | [SIGNAL TRADE LINK] Ticket: 3 | Action: BUY | Confidence: 0.5076 | SignalTime: 2026-06-04 22:30:00+00:00 | Entry: 4480.58700 | SL: 4478.59100 | TP: 4483.58100",
                    "2026-06-05 05:35:02.006 | INFO     | [SIGNAL TRADE LINK] Ticket: 4 | Action: SELL | Confidence: 0.6076 | SignalTime: 2026-06-04 22:34:00+00:00 | Entry: 30204.06000 | SL: 30239.51000 | TP: 30150.89000",
                ]
            ),
        )
        links = parse_signal_trade_links(self.logs)
        deals = [
            {"ticket": 101, "order": 1, "position_id": 1, "entry": 0, "profit": 0.0, "commission": 0.0, "swap": 0.0},
            {"ticket": 102, "order": 10, "position_id": 1, "entry": 1, "price": 4477.386, "profit": 5.0, "commission": -0.2, "swap": 0.0, "time": "2026-06-05 05:18:00"},
            {"ticket": 201, "order": 2, "position_id": 2, "entry": 0, "profit": 0.0, "commission": 0.0, "swap": 0.0},
            {"ticket": 202, "order": 20, "position_id": 2, "entry": 1, "price": 4477.502, "profit": -3.0, "commission": 0.0, "swap": 0.0, "time": "2026-06-05 05:25:00"},
            {"ticket": 401, "order": 4, "position_id": 4, "entry": 0, "profit": 0.0, "commission": 0.0, "swap": 0.0},
            {"ticket": 402, "order": 40, "position_id": 4, "entry": 1, "price": 30200.060, "profit": 6.0, "commission": 0.0, "swap": 0.0, "time": "2026-06-05 05:36:00"},
        ]

        details = build_link_details(links, deals)
        by_ticket = {row["ticket"]: row for row in details}

        self.assertTrue(by_ticket[1]["closed"])
        self.assertEqual(by_ticket[1]["close_price"], 4477.386)
        self.assertAlmostEqual(by_ticket[1]["points"], 2.0)
        self.assertEqual(by_ticket[1]["net_pl"], 4.8)
        self.assertEqual(by_ticket[1]["outcome"], "WIN")
        self.assertTrue(by_ticket[2]["closed"])
        self.assertEqual(by_ticket[2]["close_price"], 4477.502)
        self.assertAlmostEqual(by_ticket[2]["points"], -1.0)
        self.assertEqual(by_ticket[2]["net_pl"], -3.0)
        self.assertEqual(by_ticket[2]["outcome"], "LOSS")
        self.assertFalse(by_ticket[3]["closed"])
        self.assertIsNone(by_ticket[3]["close_price"])
        self.assertEqual(by_ticket[3]["points"], 0.0)
        self.assertEqual(by_ticket[3]["net_pl"], 0.0)
        self.assertEqual(by_ticket[3]["outcome"], "OPEN/UNMATCHED")
        self.assertTrue(by_ticket[4]["closed"])
        self.assertEqual(by_ticket[4]["close_price"], 30200.060)
        self.assertAlmostEqual(by_ticket[4]["points"], 4.0)
        self.assertEqual(by_ticket[4]["net_pl"], 6.0)

    def test_summary_winrate_uses_closed_trades_and_total_sums_matched_pnl(self):
        details = [
            {"symbol": "XAUUSD", "timeframe": "M1", "closed": True, "points": 2.0, "net_pl": 4.8},
            {"symbol": "XAUUSD", "timeframe": "M1", "closed": True, "points": -1.0, "net_pl": -3.0},
            {"symbol": "XAUUSD", "timeframe": "M1", "closed": False, "points": 0.0, "net_pl": 0.0},
            {"symbol": "USTEC_X100", "timeframe": "M5", "closed": True, "points": 4.0, "net_pl": 10.0},
        ]

        summary = summarize_details(details)
        by_key = {(row["symbol"], row["timeframe"]): row for row in summary}
        total = total_summary(summary)

        self.assertEqual(by_key[("XAUUSD", "M1")]["links"], 3)
        self.assertEqual(by_key[("XAUUSD", "M1")]["closed"], 2)
        self.assertEqual(by_key[("XAUUSD", "M1")]["wins"], 1)
        self.assertEqual(by_key[("XAUUSD", "M1")]["losses"], 1)
        self.assertEqual(by_key[("XAUUSD", "M1")]["winrate_pct"], 50.0)
        self.assertAlmostEqual(by_key[("XAUUSD", "M1")]["points"], 1.0)
        self.assertAlmostEqual(by_key[("XAUUSD", "M1")]["net_pl"], 1.8)
        self.assertEqual(total["links"], 4)
        self.assertEqual(total["closed"], 3)
        self.assertEqual(total["wins"], 2)
        self.assertEqual(total["losses"], 1)
        self.assertAlmostEqual(total["winrate_pct"], 66.6666666667)
        self.assertAlmostEqual(total["points"], 5.0)
        self.assertAlmostEqual(total["net_pl"], 11.8)

    def test_daily_summary_uses_close_date_for_closed_and_link_date_for_unclosed(self):
        details = [
            {
                "symbol": "XAUUSD",
                "timeframe": "M1",
                "closed": True,
                "points": 2.0,
                "net_pl": 4.8,
                "close_time": "2026-06-05 05:18:00",
                "link_time": "2026-06-04 22:16:00.000",
                "report_date": "2026-06-05",
            },
            {
                "symbol": "XAUUSD",
                "timeframe": "M1",
                "closed": True,
                "points": -1.0,
                "net_pl": -3.0,
                "close_time": "2026-06-05 05:25:00",
                "link_time": "2026-06-05 05:24:00.000",
                "report_date": "2026-06-05",
            },
            {
                "symbol": "USTEC_X100",
                "timeframe": "M5",
                "closed": False,
                "points": 0.0,
                "net_pl": 0.0,
                "close_time": "",
                "link_time": "2026-06-06 10:00:00.000",
                "report_date": "2026-06-06",
            },
        ]

        daily = summarize_details_by_day(details)
        by_date = {row["date"]: row for row in daily}
        total = total_daily_summary(daily)

        self.assertEqual(by_date["2026-06-05"]["links"], 2)
        self.assertEqual(by_date["2026-06-05"]["closed"], 2)
        self.assertEqual(by_date["2026-06-05"]["wins"], 1)
        self.assertEqual(by_date["2026-06-05"]["losses"], 1)
        self.assertEqual(by_date["2026-06-05"]["winrate_pct"], 50.0)
        self.assertAlmostEqual(by_date["2026-06-05"]["points"], 1.0)
        self.assertAlmostEqual(by_date["2026-06-05"]["net_pl"], 1.8)
        self.assertEqual(by_date["2026-06-06"]["links"], 1)
        self.assertEqual(by_date["2026-06-06"]["closed"], 0)
        self.assertEqual(by_date["2026-06-06"]["points"], 0.0)
        self.assertEqual(by_date["2026-06-06"]["net_pl"], 0.0)
        self.assertEqual(total["links"], 3)
        self.assertEqual(total["closed"], 2)
        self.assertAlmostEqual(total["points"], 1.0)
        self.assertAlmostEqual(total["net_pl"], 1.8)

    def test_default_history_days_is_weekly(self):
        args = build_arg_parser().parse_args([])

        self.assertEqual(args.history_days, 7)


if __name__ == "__main__":
    unittest.main()
