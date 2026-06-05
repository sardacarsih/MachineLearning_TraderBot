"""
Rich terminal dashboard for the live trading loop.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable
try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python < 3.9 fallback
    ZoneInfo = None

from loguru import logger
from rich import box
from rich.align import Align
from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from utils.banner import render_banner


HEADER_HEIGHT = 12
if ZoneInfo is not None:
    DISPLAY_TIMEZONE = ZoneInfo("Asia/Jakarta")
else:
    DISPLAY_TIMEZONE = timezone(timedelta(hours=7), name="WIB")


def format_dashboard_time(value: Any) -> str:
    """Display MT5 UTC candle timestamps in local WIB time."""
    if value is None:
        return "N/A"
    text = str(value)
    if text.upper() in {"", "N/A", "NONE"}:
        return "N/A"

    try:
        if hasattr(value, "to_pydatetime"):
            dt = value.to_pydatetime()
        elif isinstance(value, datetime):
            dt = value
        else:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return text

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local_dt = dt.astimezone(DISPLAY_TIMEZONE)
    return local_dt.strftime("%Y-%m-%d %H:%M:%S WIB")


def status_style(status: str) -> str:
    normalized = str(status or "").strip().upper()
    if normalized in {"CONNECTED", "ACTIVE", "LIVE", "BUY", "PROFIT", "OK"}:
        return "bold green"
    if normalized in {"DISCONNECTED", "ERROR", "FAILED", "SELL", "LOSS", "HALTED"}:
        return "bold red"
    if normalized in {"PAPER", "WARNING", "WARN", "NO_TRADE", "BLOCKED", "SKIPPED"}:
        return "bold yellow"
    return "bold cyan"


def pnl_style(value: float) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "white"
    if numeric > 0:
        return "green"
    if numeric < 0:
        return "red"
    return "white"


def format_money(value: Any) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "N/A"


def format_number(value: Any, decimals: int = 2) -> str:
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def _read_position_field(position: Any, key: str, default: Any = None) -> Any:
    if isinstance(position, dict):
        return position.get(key, default)
    return getattr(position, key, default)


def normalize_position(position: Any) -> dict[str, Any]:
    pos_type = _read_position_field(position, "type", "")
    if not isinstance(pos_type, str):
        pos_type = "BUY" if pos_type == 0 else "SELL"

    return {
        "ticket": _read_position_field(position, "ticket", ""),
        "type": str(pos_type).upper(),
        "volume": _read_position_field(position, "volume", 0.0),
        "open_price": _read_position_field(position, "open_price", _read_position_field(position, "price_open", 0.0)),
        "sl": _read_position_field(position, "sl", 0.0),
        "tp": _read_position_field(position, "tp", 0.0),
        "profit": _read_position_field(position, "profit", 0.0),
        "confidence": _read_position_field(position, "confidence", None),
    }


def _flag(features: dict[str, Any], name: str) -> bool:
    try:
        return int(float(features.get(name, 0) or 0)) == 1
    except (TypeError, ValueError):
        return False


def _numeric(features: dict[str, Any], name: str) -> float | None:
    try:
        return float(features.get(name))
    except (TypeError, ValueError):
        return None


def derive_timeframe_trends(features: dict[str, Any], base_timeframe: str = "M5") -> dict[str, dict[str, Any]]:
    """Derive reader-friendly timeframe trends from the latest feature row."""
    if not features:
        return {}

    trends: dict[str, dict[str, Any]] = {}
    base_tf = str(base_timeframe or "M5").strip().upper()

    if _flag(features, "ema_alignment_bullish"):
        base_label, base_style = "BULL", "bold green"
    elif _flag(features, "ema_alignment_bearish"):
        base_label, base_style = "BEAR", "bold red"
    else:
        plus_di = _numeric(features, "plus_di")
        minus_di = _numeric(features, "minus_di")
        if plus_di is not None and minus_di is not None and plus_di > minus_di:
            base_label, base_style = "UP", "green"
        elif plus_di is not None and minus_di is not None and minus_di > plus_di:
            base_label, base_style = "DOWN", "red"
        else:
            base_label, base_style = "RANGE", "yellow"
    trends[base_tf] = {
        "label": base_label,
        "detail": f"ADX {format_number(features.get('trend_strength'), 1)}",
        "style": base_style,
    }

    if base_tf != "M15":
        m15_pos = _numeric(features, "htf_m15_breakout_position")
        if _flag(features, "htf_m15_breakout_high"):
            m15_label, m15_style = "BULL BRK", "bold green"
        elif _flag(features, "htf_m15_breakout_low"):
            m15_label, m15_style = "BEAR BRK", "bold red"
        elif m15_pos is not None and m15_pos >= 0.65:
            m15_label, m15_style = "BULL", "green"
        elif m15_pos is not None and m15_pos <= 0.35:
            m15_label, m15_style = "BEAR", "red"
        else:
            m15_label, m15_style = "RANGE", "yellow"
        trends["M15"] = {
            "label": m15_label,
            "detail": f"Pos {format_number(m15_pos, 2)}",
            "style": m15_style,
        }

    if base_tf != "H1":
        if _flag(features, "htf_h1_ema_alignment_bullish"):
            h1_label, h1_style = "BULL", "bold green"
        elif _flag(features, "htf_h1_ema_alignment_bearish"):
            h1_label, h1_style = "BEAR", "bold red"
        else:
            h1_dist = _numeric(features, "htf_h1_dist_ema_200")
            if h1_dist is not None and h1_dist > 0:
                h1_label, h1_style = "ABOVE 200", "green"
            elif h1_dist is not None and h1_dist < 0:
                h1_label, h1_style = "BELOW 200", "red"
            else:
                h1_label, h1_style = "MIXED", "yellow"
        trends["H1"] = {
            "label": h1_label,
            "detail": f"D200 {format_number(features.get('htf_h1_dist_ema_200'), 2)}%",
            "style": h1_style,
        }

    if base_tf != "H4":
        h4_ratio = _numeric(features, "htf_h4_atr_regime_ratio")
        if _flag(features, "htf_h4_atr_regime_high"):
            h4_label, h4_style = "VOL HIGH", "bold yellow"
        elif _flag(features, "htf_h4_atr_regime_low"):
            h4_label, h4_style = "VOL LOW", "cyan"
        else:
            h4_label, h4_style = "VOL NORMAL", "white"
        trends["H4"] = {
            "label": h4_label,
            "detail": f"ATRx {format_number(h4_ratio, 2)}",
            "style": h4_style,
        }

    return trends


@dataclass
class DashboardState:
    symbol: str
    timeframe: str
    mode: str = "PAPER"
    strategy_mode: str = "HYBRID"
    htf_enabled: bool = False
    connection: str = "DISCONNECTED"
    heartbeat: str = "UNKNOWN"
    trade_allowed: bool = False
    balance: float = 0.0
    equity: float = 0.0
    free_margin: float = 0.0
    margin_level: float = 0.0
    daily_pnl: float = 0.0
    bid: float | None = None
    ask: float | None = None
    spread_points: float | None = None
    latest_candle_time: str = "N/A"
    cached_atr: float | None = None
    last_signal: dict[str, Any] = field(default_factory=dict)
    timeframe_trends: dict[str, dict[str, Any]] = field(default_factory=dict)
    positions: list[dict[str, Any]] = field(default_factory=list)
    last_update: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


class RichLiveDashboard:
    """Owns the terminal screen while the live trading loop runs."""

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        mode: str,
        strategy_mode: str,
        htf_enabled: bool = False,
        refresh_per_second: int = 4,
        max_log_lines: int = 8,
    ):
        self.state = DashboardState(
            symbol=symbol,
            timeframe=str(timeframe).upper(),
            mode=str(mode).upper(),
            strategy_mode=str(strategy_mode).upper(),
            htf_enabled=bool(htf_enabled),
        )
        self.max_log_lines = max_log_lines
        self.main_logs = deque(maxlen=max_log_lines)
        self.trade_logs = deque(maxlen=max_log_lines)
        self.signal_logs = deque(maxlen=max_log_lines)
        self.error_logs = deque(maxlen=max_log_lines)
        self._live = Live(self.render(), refresh_per_second=refresh_per_second, screen=True)
        self._sink_id = None

    def __enter__(self):
        self._live.__enter__()
        self._sink_id = logger.add(self._log_sink, level="DEBUG", enqueue=False)
        self.refresh()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._sink_id is not None:
            logger.remove(self._sink_id)
            self._sink_id = None
        self._live.__exit__(exc_type, exc, tb)

    def _log_sink(self, message):
        record = message.record
        category = record["extra"].get("category", "main")
        level = record["level"].name
        text = f"{record['time'].strftime('%H:%M:%S')} {level:<7} {record['message']}"
        self.add_log(category=category, level=level, message=text)

    def add_log(self, category: str, level: str, message: str) -> None:
        target = self.main_logs
        if category == "trade":
            target = self.trade_logs
        elif category == "signal":
            target = self.signal_logs
        elif str(level).upper() in {"ERROR", "CRITICAL"}:
            target = self.error_logs
        target.append(str(message))

    def update_connection(self, connected: bool, heartbeat_ok: bool | None = None) -> None:
        self.state.connection = "CONNECTED" if connected else "DISCONNECTED"
        if heartbeat_ok is not None:
            self.state.heartbeat = "OK" if heartbeat_ok else "FAILED"
        self._touch()

    def update_account(self, summary: dict[str, Any]) -> None:
        self.state.balance = float(summary.get("balance", 0.0) or 0.0)
        self.state.equity = float(summary.get("equity", 0.0) or 0.0)
        self.state.free_margin = float(summary.get("free_margin", 0.0) or 0.0)
        self.state.margin_level = float(summary.get("margin_level", 0.0) or 0.0)
        self.state.daily_pnl = float(summary.get("daily_pnl", 0.0) or 0.0)
        self.state.trade_allowed = bool(summary.get("trade_allowed", False))
        self._touch()

    def update_market(self, tick: Any = None, spread_points: float | None = None, cached_atr: float | None = None) -> None:
        if tick is not None:
            self.state.bid = getattr(tick, "bid", None)
            self.state.ask = getattr(tick, "ask", None)
        if spread_points is not None:
            self.state.spread_points = spread_points
        if cached_atr is not None:
            self.state.cached_atr = cached_atr
        self._touch()

    def update_candle(self, candle_time: Any) -> None:
        self.state.latest_candle_time = format_dashboard_time(candle_time)
        self._touch()

    def update_signal(self, signal: dict[str, Any]) -> None:
        payload = dict(signal or {})
        if payload.get("signal_time") is not None:
            payload["signal_time"] = format_dashboard_time(payload["signal_time"])
        self.state.last_signal = payload
        self._touch()

    def update_trends(self, features: dict[str, Any]) -> None:
        self.state.timeframe_trends = derive_timeframe_trends(features or {}, self.state.timeframe)
        self._touch()

    def update_positions(self, positions: Iterable[Any]) -> None:
        self.state.positions = [normalize_position(position) for position in (positions or [])]
        self._touch()

    def refresh(self) -> None:
        self._live.update(self.render())

    def _touch(self) -> None:
        self.state.last_update = datetime.now().strftime("%H:%M:%S")
        self.refresh()

    def render(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(self._header(), size=HEADER_HEIGHT),
            Layout(name="body", ratio=1),
            Layout(name="logs", size=12),
        )
        layout["body"].split_row(
            Layout(self._left_column(), ratio=1),
            Layout(self._positions_panel(), ratio=2),
            Layout(self._right_column(), ratio=1),
        )
        layout["logs"].split_row(
            Layout(self._log_panel("Main", self.main_logs, "cyan"), ratio=1),
            Layout(self._log_panel("Signals", self.signal_logs, "yellow"), ratio=1),
            Layout(self._log_panel("Trades", self.trade_logs, "green"), ratio=1),
            Layout(self._log_panel("Errors", self.error_logs, "red"), ratio=1),
        )
        return layout

    def _header(self) -> Panel:
        banner = Text(render_banner(self.state.symbol, self.state.timeframe).strip(), style="bold cyan", no_wrap=True)
        return Panel(Align.center(banner), border_style="cyan", box=box.SQUARE)

    def _left_column(self) -> Group:
        return Group(
            self._status_panel(),
            self._account_panel(),
        )

    def _right_column(self) -> Group:
        return Group(
            self._market_panel(),
            self._trend_panel(),
            self._signal_panel(),
        )

    def _status_panel(self) -> Panel:
        table = Table.grid(expand=True)
        table.add_column(style="dim")
        table.add_column(justify="right")
        table.add_row("Symbol", f"{self.state.symbol} {self.state.timeframe}")
        table.add_row("Mode", Text(self.state.mode, style=status_style(self.state.mode)))
        table.add_row("Strategy", self.state.strategy_mode)
        table.add_row("HTF", Text("ON" if self.state.htf_enabled else "OFF", style="bold green" if self.state.htf_enabled else "bold yellow"))
        table.add_row("Connection", Text(self.state.connection, style=status_style(self.state.connection)))
        table.add_row("Heartbeat", Text(self.state.heartbeat, style=status_style(self.state.heartbeat)))
        table.add_row("Trade Allowed", Text("YES" if self.state.trade_allowed else "NO", style="green" if self.state.trade_allowed else "red"))
        table.add_row("Updated", self.state.last_update)
        return Panel(table, title="Status", border_style="cyan", box=box.SQUARE)

    def _account_panel(self) -> Panel:
        table = Table.grid(expand=True)
        table.add_column(style="dim")
        table.add_column(justify="right")
        table.add_row("Balance", format_money(self.state.balance))
        table.add_row("Equity", format_money(self.state.equity))
        table.add_row("Free Margin", format_money(self.state.free_margin))
        table.add_row("Margin Level", f"{format_number(self.state.margin_level, 1)}%")
        table.add_row("Daily PnL", Text(format_money(self.state.daily_pnl), style=pnl_style(self.state.daily_pnl)))
        return Panel(table, title="Account", border_style="blue", box=box.SQUARE)

    def _market_panel(self) -> Panel:
        table = Table.grid(expand=True)
        table.add_column(style="dim")
        table.add_column(justify="right")
        table.add_row("Bid", format_number(self.state.bid, 2))
        table.add_row("Ask", format_number(self.state.ask, 2))
        table.add_row("Spread", f"{format_number(self.state.spread_points, 1)} pts")
        table.add_row("Last Candle", self.state.latest_candle_time)
        table.add_row("ATR", format_number(self.state.cached_atr, 4))
        return Panel(table, title="Market", border_style="cyan", box=box.SQUARE)

    def _trend_panel(self) -> Panel:
        table = Table.grid(expand=True)
        table.add_column(style="dim")
        table.add_column(justify="right")
        table.add_column(justify="right", style="dim")
        trends = self.state.timeframe_trends or {}
        display_timeframes = []
        for timeframe in (self.state.timeframe, "M15", "H1", "H4"):
            timeframe = str(timeframe).upper()
            if timeframe not in display_timeframes:
                display_timeframes.append(timeframe)
        for timeframe in display_timeframes:
            trend = trends.get(timeframe, {})
            label = trend.get("label", "WAIT")
            detail = trend.get("detail", "")
            style = trend.get("style", "bold cyan")
            table.add_row(timeframe, Text(str(label), style=style), str(detail))
        return Panel(table, title="Trend", border_style="magenta", box=box.SQUARE)

    def _signal_panel(self) -> Panel:
        signal = self.state.last_signal or {}
        action = str(signal.get("action", "WAITING")).upper()
        confidence = signal.get("confidence")
        probabilities = signal.get("probabilities") or []
        table = Table.grid(expand=True)
        table.add_column(style="dim")
        table.add_column(justify="right")
        table.add_row("Action", Text(action, style=status_style(action)))
        table.add_row("Time", str(signal.get("signal_time") or self.state.latest_candle_time))
        if signal.get("timeframe"):
            table.add_row("TF", str(signal.get("timeframe")).upper())
        table.add_row("Confidence", format_number(confidence, 4))
        if len(probabilities) >= 3:
            table.add_row("NO_TRADE", format_number(probabilities[0], 4))
            table.add_row("BUY", Text(format_number(probabilities[1], 4), style="green"))
            table.add_row("SELL", Text(format_number(probabilities[2], 4), style="red"))
        reason = signal.get("reason")
        if reason:
            table.add_row("Reason", str(reason))
        return Panel(table, title="Signal", border_style=status_style(action), box=box.SQUARE)

    def _positions_panel(self) -> Panel:
        table = Table(expand=True, box=box.SIMPLE_HEAVY)
        table.add_column("Ticket", style="dim")
        table.add_column("Side")
        table.add_column("Confidence", justify="right")
        table.add_column("Lots", justify="right")
        table.add_column("Open", justify="right")
        table.add_column("SL", justify="right")
        table.add_column("TP", justify="right")
        table.add_column("PnL", justify="right")

        if not self.state.positions:
            table.add_row("-", Text("FLAT", style="bold cyan"), "-", "-", "-", "-", "-", "-")
        else:
            for position in self.state.positions:
                pos_type = str(position["type"]).upper()
                profit = position.get("profit", 0.0)
                table.add_row(
                    str(position["ticket"]),
                    Text(pos_type, style=status_style(pos_type)),
                    format_number(position.get("confidence"), 4),
                    format_number(position["volume"], 2),
                    format_number(position["open_price"], 2),
                    format_number(position["sl"], 2),
                    format_number(position["tp"], 2),
                    Text(format_money(profit), style=pnl_style(profit)),
                )
        return Panel(table, title="Open Positions", border_style="green", box=box.SQUARE)

    def _log_panel(self, title: str, lines: deque[str], color: str) -> Panel:
        if lines:
            text = Text("\n".join(lines))
        else:
            text = Text("No events yet", style="dim")
        return Panel(text, title=title, border_style=color, box=box.SQUARE)
