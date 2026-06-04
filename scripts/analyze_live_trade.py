"""
CLI for live trade analysis from local logs, optional MT5 history, and paper DB.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.mt5_config import apply_mt5_config_from_yaml
from config.settings import config, normalize_timeframe
from mt5.account_manager import AccountManager
from mt5.connector import MT5Connector
from utils.live_trade_analysis import (
    analysis_to_json,
    build_analysis,
    format_analysis_report,
)


def _default_log_dir(symbol: str, timeframe: str) -> Path:
    return Path(config.paths.logs_dir) / symbol.upper() / normalize_timeframe(timeframe)


def _default_backtest_report(symbol: str, timeframe: str) -> Path:
    base = Path(config.paths.backtest_dir) / symbol.upper() / normalize_timeframe(timeframe)
    for candidate in (base / "hybrid" / "performance_report.txt", base / "ml" / "performance_report.txt", base / "performance_report.txt"):
        if candidate.exists():
            return candidate
    return base / "performance_report.txt"


def fetch_mt5_history(days: int, config_path: str | None = None) -> list[dict]:
    if config_path:
        apply_mt5_config_from_yaml(config_path)

    connector = MT5Connector()
    if not connector.connect():
        raise RuntimeError("Could not connect to MT5 to fetch deal history.")
    try:
        account_manager = AccountManager(connector)
        return account_manager.get_trade_history(days=days)
    finally:
        connector.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Analyze live/paper trade results from logs and optional MT5 history.")
    parser.add_argument("--symbol", default=config.symbol.symbol, help="Symbol context, e.g. XAUUSD or USTEC_X100")
    parser.add_argument("--timeframe", default=config.symbol.timeframe, help="Timeframe context, e.g. M1, M5, M15")
    parser.add_argument("--start-date", default=None, help="Inclusive log date filter in YYYYMMDD")
    parser.add_argument("--end-date", default=None, help="Inclusive log date filter in YYYYMMDD")
    parser.add_argument("--log-dir", default=None, help="Override log directory containing trades_*.log and signals_*.log")
    parser.add_argument("--backtest-report", default=None, help="Override backtest performance_report.txt path")
    parser.add_argument("--paper-db", default=None, help="Override paper_trading.db path")
    parser.add_argument("--mt5-history", action="store_true", help="Connect to MT5 and include realized deal history")
    parser.add_argument("--history-days", type=int, default=1, help="MT5 history lookback in days")
    parser.add_argument("--config", default=None, help="Credentials YAML for MT5 history connection")
    parser.add_argument("--output", default=None, help="Write text report to this path")
    parser.add_argument("--json-output", default=None, help="Write machine-readable JSON report to this path")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    timeframe = normalize_timeframe(args.timeframe)
    log_dir = Path(args.log_dir) if args.log_dir else _default_log_dir(symbol, timeframe)
    backtest_report = Path(args.backtest_report) if args.backtest_report else _default_backtest_report(symbol, timeframe)
    paper_db = Path(args.paper_db) if args.paper_db else log_dir / "paper_trading.db"

    mt5_deals = []
    if args.mt5_history:
        mt5_deals = fetch_mt5_history(days=args.history_days, config_path=args.config)

    analysis = build_analysis(
        symbol=symbol,
        timeframe=timeframe,
        log_dir=log_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        backtest_report_path=backtest_report,
        paper_db_path=paper_db,
        mt5_deals=mt5_deals,
    )
    report = format_analysis_report(analysis)
    print(report)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")

    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(analysis_to_json(analysis), encoding="utf-8")


if __name__ == "__main__":
    main()
