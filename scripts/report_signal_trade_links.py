"""
Report MT5 results for trades that have a [SIGNAL TRADE LINK] log line.

The script is read-only against MT5: it reads account info and deal history,
matches closed deals back to linked tickets, and writes TXT/CSV reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.mt5_config import apply_mt5_config_from_yaml, mt5_config
from mt5.connector import MT5Connector


SIGNAL_TRADE_LINK_RE = re.compile(
    r"^(?P<link_time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?) \| .*?"
    r"\[SIGNAL TRADE LINK\] Ticket: (?P<ticket>\d+) \| "
    r"Action: (?P<action>BUY|SELL) \| Confidence: (?P<confidence>[-\d.]+) \| "
    r"SignalTime: (?P<signal_time>[^|]+?)"
    r"(?: \| Entry: (?P<entry>[-\d.]+) \| SL: (?P<sl>[-\d.]+) \| TP: (?P<tp>[-\d.]+))?"
    r"\s*$"
)

CLOSE_ENTRY_VALUES = {1, 2, "OUT", "REV"}


@dataclass(frozen=True)
class SignalTradeLink:
    ticket: int
    symbol: str
    timeframe: str
    action: str
    confidence: float
    signal_time: str
    link_time: str
    entry: float | None
    sl: float | None
    tp: float | None
    source_file: str


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _date_from_log_name(path: Path) -> str | None:
    match = re.search(r"_(\d{8})\.log$", path.name)
    return match.group(1) if match else None


def default_start_date(history_to: datetime, history_days: int) -> str:
    return (history_to - timedelta(days=history_days)).strftime("%Y%m%d")


def default_end_date(history_to: datetime) -> str:
    return history_to.strftime("%Y%m%d")


def _infer_symbol_timeframe(path: Path, logs_dir: Path) -> tuple[str, str]:
    try:
        relative = path.relative_to(logs_dir)
    except ValueError:
        relative = path

    parts = relative.parts
    if len(parts) >= 3:
        return parts[0], parts[1]
    if len(parts) == 2:
        return parts[0], ""
    return "", ""


def iter_log_files(logs_dir: Path, start_date: str | None = None, end_date: str | None = None) -> list[Path]:
    files = []
    for pattern in ("main_*.log", "trades_*.log"):
        for path in logs_dir.rglob(pattern):
            date_key = _date_from_log_name(path)
            if start_date and date_key and date_key < start_date:
                continue
            if end_date and date_key and date_key > end_date:
                continue
            files.append(path)
    return sorted(files)


def parse_signal_trade_links(
    logs_dir: Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[SignalTradeLink]:
    """Parse and deduplicate signal-trade links from main/trade logs."""
    links_by_ticket: dict[int, SignalTradeLink] = {}
    for path in iter_log_files(logs_dir, start_date=start_date, end_date=end_date):
        symbol, timeframe = _infer_symbol_timeframe(path, logs_dir)
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            match = SIGNAL_TRADE_LINK_RE.search(line)
            if not match:
                continue

            item = match.groupdict()
            ticket = int(item["ticket"])
            if ticket in links_by_ticket:
                continue

            links_by_ticket[ticket] = SignalTradeLink(
                ticket=ticket,
                symbol=symbol,
                timeframe=timeframe,
                action=item["action"],
                confidence=float(item["confidence"]),
                signal_time=item["signal_time"].strip(),
                link_time=item["link_time"],
                entry=_to_float(item.get("entry")),
                sl=_to_float(item.get("sl")),
                tp=_to_float(item.get("tp")),
                source_file=str(path),
            )
    return sorted(links_by_ticket.values(), key=lambda item: item.link_time)


def _is_close_entry(value: Any) -> bool:
    if isinstance(value, str):
        return value.upper() in CLOSE_ENTRY_VALUES
    return value in CLOSE_ENTRY_VALUES


def _deal_net(deal: dict[str, Any]) -> float:
    return float(deal.get("net", float(deal.get("profit", 0.0)) + float(deal.get("commission", 0.0)) + float(deal.get("swap", 0.0))))


def _deal_matches_ticket(deal: dict[str, Any], ticket: int) -> bool:
    return any(int(deal.get(key) or 0) == ticket for key in ("position_id", "order", "ticket"))


def _deal_time_key(deal: dict[str, Any]) -> datetime:
    value = deal.get("time")
    if isinstance(value, datetime):
        return value
    if value:
        try:
            return datetime.fromisoformat(str(value))
        except ValueError:
            pass
    return datetime.min


def _last_close_deal(close_deals: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not close_deals:
        return None
    indexed = enumerate(close_deals)
    return max(indexed, key=lambda item: (_deal_time_key(item[1]), item[0]))[1]


def _trade_points(action: str, entry: float | None, close_price: float | None) -> float:
    if entry is None or close_price is None:
        return 0.0
    if action == "BUY":
        return close_price - entry
    if action == "SELL":
        return entry - close_price
    return 0.0


def build_link_details(links: Iterable[SignalTradeLink], deals: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    deal_rows = list(deals)
    details = []
    for link in links:
        matched = [deal for deal in deal_rows if _deal_matches_ticket(deal, link.ticket)]
        close_deals = [deal for deal in matched if _is_close_entry(deal.get("entry"))]
        net_pl = sum(_deal_net(deal) for deal in close_deals)
        last_close_deal = _last_close_deal(close_deals)
        close_price = _to_float(last_close_deal.get("price")) if last_close_deal else None
        points = _trade_points(link.action, link.entry, close_price) if close_deals else 0.0
        close_times = [str(deal.get("time", "")) for deal in close_deals if deal.get("time")]
        close_time = ";".join(close_times)
        closed = bool(close_deals)
        outcome = "OPEN/UNMATCHED"
        if closed:
            if net_pl > 0:
                outcome = "WIN"
            elif net_pl < 0:
                outcome = "LOSS"
            else:
                outcome = "BREAKEVEN"

        details.append({
            "ticket": link.ticket,
            "symbol": link.symbol,
            "timeframe": link.timeframe,
            "action": link.action,
            "confidence": link.confidence,
            "signal_time": link.signal_time,
            "link_time": link.link_time,
            "entry": link.entry,
            "sl": link.sl,
            "tp": link.tp,
            "closed": closed,
            "close_time": close_time,
            "close_price": close_price,
            "points": points,
            "net_pl": net_pl if closed else 0.0,
            "outcome": outcome,
            "matched_deals": len(matched),
            "close_deals": len(close_deals),
            "source_file": link.source_file,
            "report_date": report_date(close_time if closed else link.link_time),
        })
    return details


def report_date(value: str) -> str:
    if not value:
        return ""
    first = value.split(";", 1)[0].strip()
    return first[:10]


def summarize_details(details: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {
        "symbol": "",
        "timeframe": "",
        "links": 0,
        "closed": 0,
        "wins": 0,
        "losses": 0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "points": 0.0,
        "net_pl": 0.0,
    })

    for row in details:
        key = (str(row["symbol"]), str(row["timeframe"]))
        summary = grouped[key]
        summary["symbol"] = key[0]
        summary["timeframe"] = key[1]
        summary["links"] += 1
        summary["points"] += float(row.get("points", 0.0))
        if row["closed"]:
            net_pl = float(row["net_pl"])
            summary["closed"] += 1
            summary["net_pl"] += net_pl
            if net_pl > 0:
                summary["wins"] += 1
                summary["gross_profit"] += net_pl
            elif net_pl < 0:
                summary["losses"] += 1
                summary["gross_loss"] += net_pl

    rows = []
    for summary in grouped.values():
        closed = int(summary["closed"])
        summary["winrate_pct"] = (summary["wins"] / closed * 100.0) if closed else 0.0
        rows.append(summary)
    return sorted(rows, key=lambda item: (item["symbol"], item["timeframe"]))


def summarize_details_by_day(details: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "date": "",
        "links": 0,
        "closed": 0,
        "wins": 0,
        "losses": 0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "points": 0.0,
        "net_pl": 0.0,
    })

    for row in details:
        date_key = str(row.get("report_date") or report_date(str(row.get("link_time", ""))))
        summary = grouped[date_key]
        summary["date"] = date_key
        summary["links"] += 1
        summary["points"] += float(row.get("points", 0.0))
        if row["closed"]:
            net_pl = float(row["net_pl"])
            summary["closed"] += 1
            summary["net_pl"] += net_pl
            if net_pl > 0:
                summary["wins"] += 1
                summary["gross_profit"] += net_pl
            elif net_pl < 0:
                summary["losses"] += 1
                summary["gross_loss"] += net_pl

    rows = []
    for summary in grouped.values():
        closed = int(summary["closed"])
        summary["winrate_pct"] = (summary["wins"] / closed * 100.0) if closed else 0.0
        rows.append(summary)
    return sorted(rows, key=lambda item: item["date"])


def total_summary(summary_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    total = {
        "symbol": "TOTAL",
        "timeframe": "",
        "links": 0,
        "closed": 0,
        "wins": 0,
        "losses": 0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "points": 0.0,
        "net_pl": 0.0,
        "winrate_pct": 0.0,
    }
    for row in summary_rows:
        for key in ("links", "closed", "wins", "losses"):
            total[key] += int(row[key])
        for key in ("gross_profit", "gross_loss", "points", "net_pl"):
            total[key] += float(row[key])
    total["winrate_pct"] = (total["wins"] / total["closed"] * 100.0) if total["closed"] else 0.0
    return total


def total_daily_summary(summary_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    total = {
        "date": "TOTAL",
        "links": 0,
        "closed": 0,
        "wins": 0,
        "losses": 0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "points": 0.0,
        "net_pl": 0.0,
        "winrate_pct": 0.0,
    }
    for row in summary_rows:
        for key in ("links", "closed", "wins", "losses"):
            total[key] += int(row[key])
        for key in ("gross_profit", "gross_loss", "points", "net_pl"):
            total[key] += float(row[key])
    total["winrate_pct"] = (total["wins"] / total["closed"] * 100.0) if total["closed"] else 0.0
    return total


def format_report(
    account: dict[str, Any],
    history_from: datetime,
    history_to: datetime,
    summary_rows: list[dict[str, Any]],
    daily_rows: list[dict[str, Any]],
    details: list[dict[str, Any]],
) -> str:
    total = total_summary(summary_rows)
    daily_total = total_daily_summary(daily_rows)
    lines = [
        "=" * 92,
        "Signal Trade Link MT5 Report",
        "=" * 92,
        "",
        "Account:",
        f"  Login:        {account.get('login')}",
        f"  Server:       {account.get('server')}",
        f"  Balance:      {float(account.get('balance', 0.0)):.2f}",
        f"  Equity:       {float(account.get('equity', 0.0)):.2f}",
        f"  Currency:     {account.get('currency') or 'N/A'}",
        "",
        "History Range:",
        f"  From:         {history_from:%Y-%m-%d %H:%M:%S}",
        f"  To:           {history_to:%Y-%m-%d %H:%M:%S}",
        "",
        "Summary:",
        "  Symbol       TF   Link Closed  W/L     Winrate       Points       Net PL",
    ]

    for row in summary_rows:
        lines.append(
            f"  {row['symbol']:<12} {row['timeframe']:<4} "
            f"{row['links']:>4} {row['closed']:>6} "
            f"{row['wins']:>2}/{row['losses']:<2} "
            f"{row['winrate_pct']:>9.2f}% "
            f"{row['points']:>12.2f} "
            f"{row['net_pl']:>12.2f}"
        )

    lines.extend([
        "  " + "-" * 79,
        f"  {'TOTAL':<12} {'':<4} "
        f"{total['links']:>4} {total['closed']:>6} "
        f"{total['wins']:>2}/{total['losses']:<2} "
        f"{total['winrate_pct']:>9.2f}% "
        f"{total['points']:>12.2f} "
        f"{total['net_pl']:>12.2f}",
        "",
        "Daily Breakdown:",
        "  Date          Link Closed  W/L     Winrate       Points       Net PL",
    ])

    for row in daily_rows:
        lines.append(
            f"  {row['date']:<11} "
            f"{row['links']:>4} {row['closed']:>6} "
            f"{row['wins']:>2}/{row['losses']:<2} "
            f"{row['winrate_pct']:>9.2f}% "
            f"{row['points']:>12.2f} "
            f"{row['net_pl']:>12.2f}"
        )

    lines.extend([
        "  " + "-" * 69,
        f"  {'TOTAL':<11} "
        f"{daily_total['links']:>4} {daily_total['closed']:>6} "
        f"{daily_total['wins']:>2}/{daily_total['losses']:<2} "
        f"{daily_total['winrate_pct']:>9.2f}% "
        f"{daily_total['points']:>12.2f} "
        f"{daily_total['net_pl']:>12.2f}",
    ])

    unclosed = [row for row in details if not row["closed"]]
    unmatched = [row for row in details if int(row["matched_deals"]) == 0]
    if unclosed or unmatched:
        lines.extend(["", "Warnings:"])
        if unclosed:
            lines.append(f"  - {len(unclosed)} linked trade(s) have no close deal in the selected MT5 history range.")
        if unmatched:
            lines.append(f"  - {len(unmatched)} linked ticket(s) were not found in MT5 deals.")

    return "\n".join(lines)


def write_summary_csv(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    fieldnames = ["symbol", "timeframe", "links", "closed", "wins", "losses", "winrate_pct", "gross_profit", "gross_loss", "points", "net_pl"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
        writer.writerow({key: total_summary(summary_rows).get(key) for key in fieldnames})


def write_daily_csv(path: Path, daily_rows: list[dict[str, Any]]) -> None:
    fieldnames = ["date", "links", "closed", "wins", "losses", "winrate_pct", "gross_profit", "gross_loss", "points", "net_pl"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in daily_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
        writer.writerow({key: total_daily_summary(daily_rows).get(key) for key in fieldnames})


def write_details_csv(path: Path, details: list[dict[str, Any]]) -> None:
    fieldnames = [
        "ticket",
        "symbol",
        "timeframe",
        "action",
        "confidence",
        "signal_time",
        "link_time",
        "entry",
        "sl",
        "tp",
        "closed",
        "close_time",
        "close_price",
        "points",
        "net_pl",
        "outcome",
        "report_date",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in details:
            writer.writerow({key: row.get(key) for key in fieldnames})


def mt5_deals_to_dicts(deals: Iterable[Any]) -> list[dict[str, Any]]:
    import MetaTrader5 as mt5

    rows = []
    for deal in deals:
        rows.append({
            "ticket": int(deal.ticket),
            "order": int(deal.order),
            "position_id": int(getattr(deal, "position_id", 0) or 0),
            "time": datetime.fromtimestamp(deal.time).strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": deal.symbol,
            "type": "BUY" if deal.type == mt5.DEAL_TYPE_BUY else "SELL",
            "entry": int(getattr(deal, "entry", -1)),
            "volume": float(deal.volume),
            "price": float(deal.price),
            "profit": float(deal.profit),
            "commission": float(deal.commission),
            "swap": float(deal.swap),
            "net": float(deal.profit) + float(deal.commission) + float(deal.swap),
            "magic": int(getattr(deal, "magic", 0)),
            "comment": getattr(deal, "comment", ""),
        })
    return rows


def fetch_mt5_context(config_path: str, history_days: int) -> tuple[dict[str, Any], datetime, datetime, list[dict[str, Any]]]:
    apply_mt5_config_from_yaml(config_path)
    connector = MT5Connector()
    if not connector.connect():
        raise RuntimeError("Could not connect to MT5.")

    try:
        import MetaTrader5 as mt5

        account_info = mt5.account_info()
        if account_info is None:
            raise RuntimeError(f"Could not read MT5 account info. Error: {mt5.last_error()}")

        history_to = datetime.now()
        history_from = history_to - timedelta(days=history_days)
        deals = mt5.history_deals_get(history_from, history_to)
        if deals is None:
            raise RuntimeError(f"Could not read MT5 deal history. Error: {mt5.last_error()}")

        account = {
            "login": int(account_info.login),
            "server": mt5_config.server,
            "balance": float(account_info.balance),
            "equity": float(account_info.equity),
            "currency": getattr(account_info, "currency", ""),
        }
        return account, history_from, history_to, mt5_deals_to_dicts(deals)
    finally:
        connector.disconnect()


def default_output_paths(login: Any, timestamp: str) -> tuple[Path, Path, Path, Path]:
    return (
        Path("reports") / f"signal_trade_links_{login}_{timestamp}.txt",
        Path("reports") / f"signal_trade_links_{login}_{timestamp}.csv",
        Path("reports") / f"signal_trade_links_daily_{login}_{timestamp}.csv",
        Path("reports") / f"signal_trade_links_details_{login}_{timestamp}.csv",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report MT5 PnL for trades with [SIGNAL TRADE LINK].")
    parser.add_argument("--config", default="credentials.yaml", help="MT5 credentials YAML path.")
    parser.add_argument("--history-days", type=int, default=7, help="MT5 history lookback in days.")
    parser.add_argument("--start-date", default=None, help="Inclusive log date filter in YYYYMMDD.")
    parser.add_argument("--end-date", default=None, help="Inclusive log date filter in YYYYMMDD.")
    parser.add_argument("--logs-dir", default="logs", help="Root logs directory.")
    parser.add_argument("--output", default=None, help="TXT report output path.")
    parser.add_argument("--csv-output", default=None, help="Summary CSV output path.")
    parser.add_argument("--daily-csv", default=None, help="Daily breakdown CSV output path.")
    parser.add_argument("--details-csv", default=None, help="Per-ticket details CSV output path.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary JSON to stdout.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    account, history_from, history_to, deals = fetch_mt5_context(args.config, args.history_days)
    start_date = args.start_date or default_start_date(history_to, args.history_days)
    end_date = args.end_date or default_end_date(history_to)
    links = parse_signal_trade_links(Path(args.logs_dir), start_date=start_date, end_date=end_date)
    details = build_link_details(links, deals)
    summary_rows = summarize_details(details)
    daily_rows = summarize_details_by_day(details)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_txt, default_csv, default_daily, default_details = default_output_paths(account["login"], timestamp)
    output_path = Path(args.output) if args.output else default_txt
    csv_path = Path(args.csv_output) if args.csv_output else default_csv
    daily_path = Path(args.daily_csv) if args.daily_csv else default_daily
    details_path = Path(args.details_csv) if args.details_csv else default_details

    report = format_report(account, history_from, history_to, summary_rows, daily_rows, details)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    write_summary_csv(csv_path, summary_rows)
    write_daily_csv(daily_path, daily_rows)
    write_details_csv(details_path, details)

    print(report)
    print("")
    print(f"TXT report:     {output_path}")
    print(f"Summary CSV:    {csv_path}")
    print(f"Daily CSV:      {daily_path}")
    print(f"Details CSV:    {details_path}")

    if args.json:
        print(json.dumps({
            "account": account,
            "history_from": history_from.isoformat(sep=" "),
            "history_to": history_to.isoformat(sep=" "),
            "summary": summary_rows,
            "daily": daily_rows,
            "total": total_summary(summary_rows),
        }, indent=2))


if __name__ == "__main__":
    main()
