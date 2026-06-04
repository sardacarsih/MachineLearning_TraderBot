"""
Live trade log and history analysis.

The analyzer is intentionally usable without an MT5 terminal. Local trade and
signal logs are always parsed; broker deal history can be supplied separately
when MT5 is available.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from config.settings import config, normalize_timeframe


SIGNAL_RE = re.compile(
    r"^(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?) \| .*?"
    r"Signal Generated: (?P<action>BUY|SELL|NO_TRADE) \| "
    r"Confidence: (?P<confidence>[-\d.]+)"
)
OPEN_RE = re.compile(
    r"^(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?) \| .*?"
    r"\[TRADE OPENED\] Ticket: (?P<ticket>\d+) \| (?P<symbol>\S+) "
    r"(?P<side>BUY|SELL) (?P<volume>[-\d.]+) lots at (?P<price>[-\d.]+) "
    r"\(Requested: (?P<requested>[-\d.]+), Slippage: (?P<slippage>[-\d.]+) pts\) "
    r"\| SL: (?P<sl>[-\d.]+), TP: (?P<tp>[-\d.]+)"
)
SIGNAL_TRADE_LINK_RE = re.compile(
    r"^(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?) \| .*?"
    r"\[SIGNAL TRADE LINK\] Ticket: (?P<ticket>\d+) \| "
    r"Action: (?P<action>BUY|SELL) \| Confidence: (?P<confidence>[-\d.]+) \| "
    r"SignalTime: (?P<signal_time>[^|]+)"
    r"(?: \| Entry: (?P<entry>[-\d.]+) \| SL: (?P<sl>[-\d.]+) \| TP: (?P<tp>[-\d.]+))?"
)
FAILED_RE = re.compile(
    r"^(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?) \| .*?"
    r"\[TRADE FAILED\] (?P<symbol>\S+) (?P<side>BUY|SELL) (?P<volume>[-\d.]+) lots "
    r"\| Reason: (?P<reason>.+)$"
)
SKIPPED_RE = re.compile(
    r"^(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?) \| .*?"
    r"\[TRADE SKIPPED\] (?P<symbol>\S+) (?P<side>BUY|SELL) "
    r"\| Reason: (?P<reason>.+)$"
)
LIVE_CLOSE_RE = re.compile(
    r"^(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?) \| .*?"
    r"\[TRADE CLOSED\] Ticket: (?P<ticket>\d+) \| (?P<symbol>\S+) closed at "
    r"(?P<price>[-\d.]+) \(Entry: (?P<entry>[-\d.]+), Profit: (?P<pnl>[-\d.]+)\)"
)
PAPER_CLOSE_RE = re.compile(
    r"^(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?) \| .*?"
    r"\[PAPER TRADE CLOSE\] Ticket: (?P<ticket>\d+) \| (?P<symbol>\S+) "
    r"(?P<side>BUY|SELL) closed at (?P<price>[-\d.]+) \(Entry: (?P<entry>[-\d.]+)\) "
    r"\| Gross PnL: (?P<gross_pnl>[-\d.]+), Net PnL: (?P<pnl>[-\d.]+)"
    r"(?: \(Comm: [-\d.]+\))?(?: \| ExitReason: (?P<close_reason>TP|SL|WIN|LOSS|MANUAL|UNKNOWN))?"
)


@dataclass
class ParsedLogs:
    signals: list[dict[str, Any]] = field(default_factory=list)
    signal_trade_links: list[dict[str, Any]] = field(default_factory=list)
    opened: list[dict[str, Any]] = field(default_factory=list)
    closed: list[dict[str, Any]] = field(default_factory=list)
    failed: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    unparsed_trade_lines: int = 0


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).replace("$", "").replace("%", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return default


def _date_from_path(path: Path) -> str | None:
    match = re.search(r"_(\d{8})\.log$", path.name)
    return match.group(1) if match else None


def select_log_files(log_dir: Path, prefix: str, start_date: str | None = None, end_date: str | None = None) -> list[Path]:
    files = []
    for path in sorted(log_dir.glob(f"{prefix}_*.log")):
        date_key = _date_from_path(path)
        if start_date and date_key and date_key < start_date:
            continue
        if end_date and date_key and date_key > end_date:
            continue
        files.append(path)
    return files


def parse_signal_logs(paths: Iterable[Path]) -> list[dict[str, Any]]:
    signals = []
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            match = SIGNAL_RE.search(line)
            if not match:
                continue
            item = match.groupdict()
            item["time"] = _parse_time(item["time"])
            item["confidence"] = _to_float(item["confidence"])
            signals.append(item)
    return signals


def parse_trade_logs(paths: Iterable[Path]) -> ParsedLogs:
    parsed = ParsedLogs()
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue

            for pattern, bucket in (
                (OPEN_RE, parsed.opened),
                (SIGNAL_TRADE_LINK_RE, parsed.signal_trade_links),
                (FAILED_RE, parsed.failed),
                (SKIPPED_RE, parsed.skipped),
                (LIVE_CLOSE_RE, parsed.closed),
                (PAPER_CLOSE_RE, parsed.closed),
            ):
                match = pattern.search(line)
                if match:
                    item = match.groupdict()
                    item["time"] = _parse_time(item["time"])
                    for key in ("volume", "price", "requested", "slippage", "sl", "tp", "entry", "pnl", "gross_pnl", "confidence"):
                        if key in item and item[key] is not None:
                            item[key] = _to_float(item[key])
                    bucket.append(item)
                    break
            else:
                if "[TRADE" in line:
                    parsed.unparsed_trade_lines += 1
    return parsed


def parse_backtest_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    content = path.read_text(encoding="utf-8", errors="ignore")

    def find(pattern: str) -> float | None:
        match = re.search(pattern, content)
        return _to_float(match.group(1)) if match else None

    return {
        "path": str(path),
        "net_profit": find(r"Net Profit:\s+(\$?[-\d.,]+)"),
        "total_trades": find(r"Total Trades:\s+(\d+)"),
        "winrate_pct": find(r"Winning Trades:\s+\d+\s+\(([\d.,]+)%\)"),
        "profit_factor": find(r"Profit Factor:\s+([-\d.,]+)"),
        "expectancy": find(r"Trade Expectancy:\s+(\$?[-\d.,]+)"),
        "max_drawdown_pct": find(r"Max Drawdown \(%\):\s+([\d.,]+)%"),
    }


def read_paper_state(db_path: Path) -> dict[str, Any] | None:
    if not db_path.exists():
        return None
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        account = conn.execute("SELECT balance, equity, last_updated FROM account WHERE account_key = 'main'").fetchone()
        open_positions = conn.execute("SELECT COUNT(*) AS count FROM positions").fetchone()["count"]
    return {
        "db_path": str(db_path),
        "balance": float(account["balance"]) if account else None,
        "equity": float(account["equity"]) if account else None,
        "last_updated": account["last_updated"] if account else None,
        "open_positions": int(open_positions),
    }


def summarize_closed_trades(closed: list[dict[str, Any]]) -> dict[str, Any]:
    pnl_values = [_to_float(item.get("pnl")) for item in closed]
    wins = [value for value in pnl_values if value > 0]
    losses = [value for value in pnl_values if value < 0]
    gross_profit = sum(wins)
    gross_loss = sum(losses)
    net_pnl = sum(pnl_values)
    total = len(pnl_values)
    cumulative = []
    running = 0.0
    for value in pnl_values:
        running += value
        cumulative.append(running)

    peak = 0.0
    max_drawdown = 0.0
    for value in cumulative:
        peak = max(peak, value)
        max_drawdown = max(max_drawdown, peak - value)

    return {
        "total_closed": total,
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "winrate_pct": (len(wins) / total * 100.0) if total else 0.0,
        "net_pnl": net_pnl,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": (gross_profit / abs(gross_loss)) if gross_loss else (math.inf if gross_profit > 0 else 0.0),
        "average_win": (gross_profit / len(wins)) if wins else 0.0,
        "average_loss": (gross_loss / len(losses)) if losses else 0.0,
        "expectancy": (net_pnl / total) if total else 0.0,
        "max_drawdown": max_drawdown,
        "equity_curve": cumulative,
    }


def _ticket_key(value: Any) -> str:
    return str(value or "").strip()


def _avg(values: list[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def classify_trade_outcome(closed_trade: dict[str, Any] | None) -> str:
    if not closed_trade:
        return "OPEN/UNKNOWN"

    reason = str(closed_trade.get("close_reason") or closed_trade.get("reason_label") or "").upper()
    if reason in {"TP", "SL"}:
        return reason

    pnl = _to_float(closed_trade.get("pnl"))
    if pnl > 0:
        return "WIN"
    if pnl < 0:
        return "LOSS"
    return "BREAKEVEN"


def summarize_confidence_thresholds(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
    rows = []
    for threshold in thresholds:
        matched = [
            item for item in trades
            if item.get("confidence") is not None and _to_float(item.get("confidence")) >= threshold
        ]
        closed = [item for item in matched if item.get("outcome") != "OPEN/UNKNOWN"]
        wins = [item for item in closed if item.get("outcome") in {"TP", "WIN"}]
        losses = [item for item in closed if item.get("outcome") in {"SL", "LOSS"}]
        net_pnl = sum(_to_float(item.get("pnl")) for item in closed)
        rows.append({
            "threshold": threshold,
            "trades": len(matched),
            "closed": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "open_unknown": len(matched) - len(closed),
            "winrate_pct": (len(wins) / len(closed) * 100.0) if closed else 0.0,
            "net_pnl": net_pnl,
            "avg_confidence": _avg([_to_float(item.get("confidence")) for item in matched]),
        })
    return rows


def summarize_confidence_outcomes(
    opened: list[dict[str, Any]],
    closed: list[dict[str, Any]],
    signal_trade_links: list[dict[str, Any]],
    signals: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    links_by_ticket = {_ticket_key(item.get("ticket")): item for item in signal_trade_links}
    closed_by_ticket = {_ticket_key(item.get("ticket")): item for item in closed}
    sorted_signals = sorted(signals or [], key=lambda item: item.get("time") or datetime.min)

    def fallback_signal_for(opened_trade: dict[str, Any]) -> dict[str, Any]:
        open_time = opened_trade.get("time")
        side = opened_trade.get("side")
        if not isinstance(open_time, datetime) or side not in {"BUY", "SELL"}:
            return {}
        candidates = [
            item for item in sorted_signals
            if item.get("action") == side
            and isinstance(item.get("time"), datetime)
            and item["time"] <= open_time
            and open_time - item["time"] <= timedelta(minutes=10)
        ]
        if not candidates:
            return {}
        signal = candidates[-1]
        return {
            "action": signal.get("action"),
            "confidence": signal.get("confidence"),
            "signal_time": signal.get("time"),
        }

    trades = []
    for opened_trade in opened:
        ticket = _ticket_key(opened_trade.get("ticket"))
        link = links_by_ticket.get(ticket) or fallback_signal_for(opened_trade)
        closed_trade = closed_by_ticket.get(ticket)
        outcome = classify_trade_outcome(closed_trade)
        confidence = link.get("confidence")
        if confidence is not None:
            confidence = _to_float(confidence)

        trades.append({
            "ticket": ticket,
            "symbol": opened_trade.get("symbol"),
            "action": link.get("action") or opened_trade.get("side"),
            "signal_time": link.get("signal_time"),
            "open_time": opened_trade.get("time"),
            "close_time": closed_trade.get("time") if closed_trade else None,
            "confidence": confidence,
            "entry": link.get("entry", opened_trade.get("price")),
            "sl": link.get("sl", opened_trade.get("sl")),
            "tp": link.get("tp", opened_trade.get("tp")),
            "pnl": closed_trade.get("pnl") if closed_trade else None,
            "outcome": outcome,
            "source": closed_trade.get("source") if closed_trade else None,
        })

    def confidences(*outcomes: str) -> list[float]:
        labels = set(outcomes)
        return [
            _to_float(item["confidence"])
            for item in trades
            if item.get("confidence") is not None and item.get("outcome") in labels
        ]

    tp_conf = confidences("TP")
    sl_conf = confidences("SL")
    win_conf = confidences("TP", "WIN")
    loss_conf = confidences("SL", "LOSS")
    open_unknown_conf = confidences("OPEN/UNKNOWN")

    return {
        "linked_trades": len(signal_trade_links),
        "tracked_trades": len(trades),
        "confidence_matched_trades": sum(1 for item in trades if item.get("confidence") is not None),
        "tp_count": sum(1 for item in trades if item["outcome"] == "TP"),
        "sl_count": sum(1 for item in trades if item["outcome"] == "SL"),
        "win_count": sum(1 for item in trades if item["outcome"] in {"TP", "WIN"}),
        "loss_count": sum(1 for item in trades if item["outcome"] in {"SL", "LOSS"}),
        "open_unknown_count": sum(1 for item in trades if item["outcome"] == "OPEN/UNKNOWN"),
        "avg_confidence_tp": _avg(tp_conf),
        "avg_confidence_sl": _avg(sl_conf),
        "avg_confidence_win": _avg(win_conf),
        "avg_confidence_loss": _avg(loss_conf),
        "avg_confidence_open_unknown": _avg(open_unknown_conf),
        "thresholds": summarize_confidence_thresholds(trades),
        "trades": trades,
    }


def reason_bucket(reason: str) -> str:
    text = str(reason or "").lower()
    if "margin" in text or "money" in text or "balance" in text:
        return "margin/balance"
    if "max open positions" in text:
        return "max open positions"
    if "averaging" in text:
        return "averaging rule"
    if "filling" in text:
        return "filling type"
    if "cooldown" in text:
        return "cooldown"
    if "spread" in text:
        return "spread"
    if "filter" in text:
        return "filter"
    if "lot" in text or "volume" in text:
        return "lot/volume"
    return str(reason or "unknown").split(".")[0][:80]


def build_analysis(
    symbol: str,
    timeframe: str,
    log_dir: Path,
    start_date: str | None = None,
    end_date: str | None = None,
    backtest_report_path: Path | None = None,
    paper_db_path: Path | None = None,
    mt5_deals: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    trade_files = select_log_files(log_dir, "trades", start_date, end_date)
    signal_files = select_log_files(log_dir, "signals", start_date, end_date)
    parsed = parse_trade_logs(trade_files)
    parsed.signals = parse_signal_logs(signal_files)

    if mt5_deals:
        for deal in mt5_deals:
            if str(deal.get("entry", "")).upper() not in {"OUT", "REV"}:
                continue
            close_ticket = deal.get("position_id") or deal.get("order") or deal.get("ticket")
            parsed.closed.append({
                "time": deal.get("time"),
                "ticket": close_ticket,
                "deal_ticket": deal.get("ticket"),
                "order": deal.get("order"),
                "symbol": deal.get("symbol"),
                "side": deal.get("type"),
                "pnl": _to_float(deal.get("profit")) + _to_float(deal.get("commission")) + _to_float(deal.get("swap")),
                "close_reason": deal.get("reason_label"),
                "reason": deal.get("reason"),
                "source": "mt5_history",
            })

    signal_counts = Counter(item["action"] for item in parsed.signals)
    failed_reasons = Counter(reason_bucket(item.get("reason")) for item in parsed.failed)
    skipped_reasons = Counter(reason_bucket(item.get("reason")) for item in parsed.skipped)
    slippages = [_to_float(item.get("slippage")) for item in parsed.opened if item.get("slippage") is not None]
    closed_summary = summarize_closed_trades(parsed.closed)
    confidence_outcomes = summarize_confidence_outcomes(
        parsed.opened,
        parsed.closed,
        parsed.signal_trade_links,
        parsed.signals,
    )
    total_attempts = len(parsed.opened) + len(parsed.failed) + len(parsed.skipped)
    failure_rate = ((len(parsed.failed) + len(parsed.skipped)) / total_attempts * 100.0) if total_attempts else 0.0

    backtest = parse_backtest_report(backtest_report_path) if backtest_report_path else None
    paper_state = read_paper_state(paper_db_path) if paper_db_path else None

    decision = "WAIT"
    reasons = []
    pf = closed_summary["profit_factor"]
    if closed_summary["total_closed"] == 0:
        reasons.append("No closed trades/PnL source found; realized performance is incomplete.")
    if failure_rate >= 30.0:
        reasons.append(f"High failed/skipped rate: {failure_rate:.1f}%.")
    if slippages and max(slippages) > config.risk.max_slippage:
        reasons.append(f"Max slippage {max(slippages):.1f} pts exceeds configured {config.risk.max_slippage} pts.")
    if closed_summary["total_closed"] and (pf <= 1.0 or closed_summary["expectancy"] <= 0.0):
        reasons.append("Realized profit factor or expectancy is not positive.")
    if backtest and closed_summary["total_closed"] and backtest.get("profit_factor") is not None and pf < backtest["profit_factor"] * 0.75:
        reasons.append("Live profit factor is materially below backtest.")

    no_closed_reason = "No closed trades/PnL source found; realized performance is incomplete."
    blocking_reasons = [reason for reason in reasons if reason != no_closed_reason]

    if not reasons and closed_summary["total_closed"]:
        decision = "CONTINUE"
        reasons.append("Profit factor, expectancy, execution failures, and slippage are within operating thresholds.")
    elif no_closed_reason in reasons and len(parsed.opened) > 0 and not blocking_reasons:
        decision = "MONITOR"
    else:
        decision = "PAUSE_REVIEW"

    return {
        "context": {
            "symbol": symbol.upper(),
            "timeframe": normalize_timeframe(timeframe),
            "log_dir": str(log_dir),
            "start_date": start_date,
            "end_date": end_date,
        },
        "sources": {
            "trade_files": [str(path) for path in trade_files],
            "signal_files": [str(path) for path in signal_files],
            "backtest_report": str(backtest_report_path) if backtest_report_path and backtest_report_path.exists() else None,
            "paper_db": str(paper_db_path) if paper_db_path and paper_db_path.exists() else None,
            "mt5_history_deals": len(mt5_deals or []),
        },
        "signals": {
            "total": len(parsed.signals),
            "buy": signal_counts.get("BUY", 0),
            "sell": signal_counts.get("SELL", 0),
            "no_trade": signal_counts.get("NO_TRADE", 0),
            "average_confidence": (
                sum(item["confidence"] for item in parsed.signals) / len(parsed.signals)
                if parsed.signals else 0.0
            ),
        },
        "execution": {
            "opened": len(parsed.opened),
            "closed": len(parsed.closed),
            "signal_trade_links": len(parsed.signal_trade_links),
            "failed": len(parsed.failed),
            "skipped": len(parsed.skipped),
            "failure_rate_pct": failure_rate,
            "avg_slippage_pts": (sum(slippages) / len(slippages)) if slippages else 0.0,
            "max_slippage_pts": max(slippages) if slippages else 0.0,
            "failed_reasons": dict(failed_reasons),
            "skipped_reasons": dict(skipped_reasons),
            "unparsed_trade_lines": parsed.unparsed_trade_lines,
        },
        "performance": closed_summary,
        "confidence_outcomes": confidence_outcomes,
        "backtest": backtest,
        "paper_state": paper_state,
        "decision": {
            "status": decision,
            "reasons": reasons,
        },
    }


def format_analysis_report(analysis: dict[str, Any]) -> str:
    ctx = analysis["context"]
    signals = analysis["signals"]
    execution = analysis["execution"]
    perf = analysis["performance"]
    confidence = analysis.get("confidence_outcomes", {})
    decision = analysis["decision"]
    backtest = analysis.get("backtest")
    paper_state = analysis.get("paper_state")

    pf = perf["profit_factor"]
    pf_text = "inf" if math.isinf(pf) else f"{pf:.2f}"
    lines = [
        "=" * 72,
        f"Live Trade Analysis - {ctx['symbol']} {ctx['timeframe']}",
        "=" * 72,
        "",
        "Signal Quality:",
        f"  Total Signals:              {signals['total']}",
        f"  BUY / SELL / NO_TRADE:      {signals['buy']} / {signals['sell']} / {signals['no_trade']}",
        f"  Average Confidence:         {signals['average_confidence']:.4f}",
        "",
        "Execution Quality:",
        f"  Opened / Closed:            {execution['opened']} / {execution['closed']}",
        f"  Signal Trade Links:         {execution.get('signal_trade_links', 0)}",
        f"  Failed / Skipped:           {execution['failed']} / {execution['skipped']}",
        f"  Failure Rate:               {execution['failure_rate_pct']:.2f}%",
        f"  Avg / Max Slippage:         {execution['avg_slippage_pts']:.1f} / {execution['max_slippage_pts']:.1f} pts",
        f"  Unparsed Trade Lines:       {execution['unparsed_trade_lines']}",
        "",
        "Realized Performance:",
        f"  Closed Trades:              {perf['total_closed']}",
        f"  Net PnL:                    {perf['net_pnl']:.2f}",
        f"  Winrate:                    {perf['winrate_pct']:.2f}%",
        f"  Profit Factor:              {pf_text}",
        f"  Expectancy:                 {perf['expectancy']:.2f}",
        f"  Average Win / Loss:         {perf['average_win']:.2f} / {perf['average_loss']:.2f}",
        f"  Max Realized Drawdown:      {perf['max_drawdown']:.2f}",
        "",
        "Confidence Outcome:",
        f"  Confidence Matched Trades:  {confidence.get('confidence_matched_trades', 0)}",
        f"  TP / SL Count:              {confidence.get('tp_count', 0)} / {confidence.get('sl_count', 0)}",
        f"  Win / Loss Count:           {confidence.get('win_count', 0)} / {confidence.get('loss_count', 0)}",
        f"  Open/Unknown Count:         {confidence.get('open_unknown_count', 0)}",
        f"  TP Avg Confidence:          {confidence.get('avg_confidence_tp', 0.0):.4f}",
        f"  SL Avg Confidence:          {confidence.get('avg_confidence_sl', 0.0):.4f}",
        f"  Win Avg Confidence:         {confidence.get('avg_confidence_win', 0.0):.4f}",
        f"  Loss Avg Confidence:        {confidence.get('avg_confidence_loss', 0.0):.4f}",
        f"  Open/Unknown Avg Confidence:{confidence.get('avg_confidence_open_unknown', 0.0):>10.4f}",
    ]

    threshold_rows = confidence.get("thresholds") or []
    if threshold_rows:
        lines.extend(["", "Confidence Thresholds:"])
        lines.append("  MinConf  Trades  Closed  Win/Loss  Winrate   NetPnL")
        for row in threshold_rows:
            lines.append(
                f"  >= {row['threshold']:.2f}  "
                f"{row['trades']:>6}  "
                f"{row['closed']:>6}  "
                f"{row['wins']:>3}/{row['losses']:<3}  "
                f"{row['winrate_pct']:>6.2f}%  "
                f"{row['net_pnl']:>8.2f}"
            )

    if execution["failed_reasons"] or execution["skipped_reasons"]:
        lines.extend(["", "Blocked/Failed Reasons:"])
        for label, count in sorted(execution["failed_reasons"].items()):
            lines.append(f"  FAILED  {label:<24} {count}")
        for label, count in sorted(execution["skipped_reasons"].items()):
            lines.append(f"  SKIPPED {label:<24} {count}")

    if backtest:
        lines.extend([
            "",
            "Backtest Baseline:",
            f"  Trades / Winrate:           {backtest.get('total_trades', 0):.0f} / {backtest.get('winrate_pct', 0.0):.2f}%",
            f"  Net Profit:                 {backtest.get('net_profit', 0.0):.2f}",
            f"  Profit Factor:              {backtest.get('profit_factor', 0.0):.2f}",
            f"  Max Drawdown:               {backtest.get('max_drawdown_pct', 0.0):.2f}%",
        ])

    if paper_state:
        lines.extend([
            "",
            "Paper State:",
            f"  Balance / Equity:           {paper_state.get('balance', 0.0):.2f} / {paper_state.get('equity', 0.0):.2f}",
            f"  Open Positions:             {paper_state.get('open_positions', 0)}",
            f"  Last Updated:               {paper_state.get('last_updated') or 'N/A'}",
        ])

    lines.extend(["", "Operational Decision:", f"  Status:                     {decision['status']}"])
    lines.extend(f"  - {reason}" for reason in decision["reasons"])
    return "\n".join(lines)


def analysis_to_json(analysis: dict[str, Any]) -> str:
    def default(value: Any):
        if isinstance(value, datetime):
            return value.isoformat(sep=" ")
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    return json.dumps(analysis, indent=2, default=default)
