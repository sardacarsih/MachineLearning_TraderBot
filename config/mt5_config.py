"""
============================================
MetaTrader 5 Connection Configuration
============================================
MT5 login credentials, server settings, and connection parameters.
"""

from dataclasses import dataclass, fields, field
from typing import Optional


@dataclass
class MT5Config:
    """MetaTrader 5 connection and execution settings."""

    # ---- Connection ----
    # MT5 terminal path (auto-detect if None)
    terminal_path: Optional[str] = None
    # Login credentials
    login: int = 0            # Your MT5 account number
    password: str = ""        # Your MT5 password
    server: str = ""          # Your broker server name
    # Connection timeout (ms)
    timeout: int = 10000

    # ---- Reconnection ----
    # Auto reconnect on disconnect
    auto_reconnect: bool = True
    # Maximum reconnection attempts
    max_reconnect_attempts: int = 5
    # Delay between reconnection attempts (seconds)
    reconnect_delay: int = 5
    # Heartbeat interval to check connection (seconds)
    heartbeat_interval: int = 30

    # ---- Execution ----
    # Order execution type: FILLING_FOK, FILLING_IOC, FILLING_RETURN
    filling_type: str = "FILLING_IOC"
    # Maximum deviation/slippage (points)
    max_deviation: int = 10
    # Magic number for identifying bot's orders
    magic_number: int = 202401
    # Order comment prefix
    order_comment: str = "ML_BOT"

    # ---- Symbol Mapping ----
    # Some brokers use different symbol names
    symbol_name: str = "XAUUSD"
    # Alternative symbol names to try
    symbol_alternatives: list = field(
        default_factory=lambda: ["XAUUSD", "GOLD", "XAUUSDm", "XAUUSD."]
    )

    # ---- Safety ----
    # Enable trading (set False for read-only mode)
    trading_enabled: bool = False  # Must explicitly enable
    # Paper trading mode (no real orders)
    paper_trading: bool = True
    # Maximum allowed lot size (safety cap)
    safety_max_lot: float = 1.0
    # Broker account currency. Use AUTO to read account_info().currency.
    account_currency: str = "AUTO"
    # Currency used by symbol risk math, e.g. XAUUSD PnL/risk is USD.
    risk_quote_currency: str = "USD"
    # Broker symbols to try for USD/IDR conversion.
    usd_idr_symbol_candidates: list = field(
        default_factory=lambda: ["USDIDR", "USDIDRm", "USDIDR."]
    )
    # USD/IDR rate source: "broker" uses MT5 quotes, "manual" uses usd_idr_manual_rate.
    usd_idr_rate_mode: str = "broker"
    # Manual USD/IDR rate used when usd_idr_rate_mode is "manual".
    usd_idr_manual_rate: float = 15000.0
    # Fallback USD/IDR rate for paper/offline mode or missing broker quote.
    usd_idr_fallback_rate: float = 16400.0
    # Fraction of free margin to reserve when pre-checking orders.
    margin_reserve_pct: float = 0.10


# Default MT5 config instance
mt5_config = MT5Config()


def _parse_bool(value) -> bool:
    """Parse YAML/env-style boolean values without treating "false" as truthy."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def load_mt5_config_from_env() -> MT5Config:
    """
    Load MT5 configuration from environment variables.
    Environment variables:
        MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH
    """
    import os
    cfg = MT5Config()
    cfg.login = int(os.getenv("MT5_LOGIN", "0"))
    cfg.password = os.getenv("MT5_PASSWORD", "")
    cfg.server = os.getenv("MT5_SERVER", "")
    cfg.terminal_path = os.getenv("MT5_PATH", None)
    return cfg


def load_mt5_config_from_yaml(filepath: str) -> MT5Config:
    """
    Load MT5 configuration from a YAML file.

    Expected YAML structure:
        mt5:
          login: 12345678
          password: "your_password"
          server: "YourBroker-Server"
          terminal_path: "C:/Program Files/MetaTrader 5/terminal64.exe"
          magic_number: 202401
          paper_trading: true
    """
    import yaml
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        data = yaml.safe_load(f)

    mt5_data = data.get("mt5", {})
    cfg = MT5Config()

    if "login" in mt5_data:
        cfg.login = int(mt5_data["login"])
    if "password" in mt5_data:
        cfg.password = str(mt5_data["password"])
    if "server" in mt5_data:
        cfg.server = str(mt5_data["server"])
    if "terminal_path" in mt5_data:
        cfg.terminal_path = str(mt5_data["terminal_path"])
    if "magic_number" in mt5_data:
        cfg.magic_number = int(mt5_data["magic_number"])
    if "paper_trading" in mt5_data:
        cfg.paper_trading = bool(mt5_data["paper_trading"])
    if "trading_enabled" in mt5_data:
        cfg.trading_enabled = bool(mt5_data["trading_enabled"])
    if "safety_max_lot" in mt5_data:
        cfg.safety_max_lot = float(mt5_data["safety_max_lot"])
    if "account_currency" in mt5_data:
        cfg.account_currency = str(mt5_data["account_currency"]).upper()
    if "risk_quote_currency" in mt5_data:
        cfg.risk_quote_currency = str(mt5_data["risk_quote_currency"]).upper()
    if "usd_idr_symbol_candidates" in mt5_data:
        cfg.usd_idr_symbol_candidates = [str(symbol) for symbol in mt5_data["usd_idr_symbol_candidates"]]
    if "usd_idr_rate_mode" in mt5_data:
        cfg.usd_idr_rate_mode = str(mt5_data["usd_idr_rate_mode"]).lower()
    if "usd_idr_manual_rate" in mt5_data:
        cfg.usd_idr_manual_rate = float(mt5_data["usd_idr_manual_rate"])
    if "usd_idr_fallback_rate" in mt5_data:
        cfg.usd_idr_fallback_rate = float(mt5_data["usd_idr_fallback_rate"])
    if "margin_reserve_pct" in mt5_data:
        cfg.margin_reserve_pct = float(mt5_data["margin_reserve_pct"])

    return cfg


def apply_mt5_config_from_yaml(filepath: str) -> MT5Config:
    """
    Load MT5 configuration from YAML and copy it into the shared singleton.

    Modules import ``mt5_config`` directly, so replacing the object would leave
    older imports pointing at stale defaults. Mutating the existing instance
    keeps all modules in sync.
    """
    loaded = load_mt5_config_from_yaml(filepath)
    for cfg_field in fields(MT5Config):
        setattr(mt5_config, cfg_field.name, getattr(loaded, cfg_field.name))
    apply_runtime_config_from_yaml(filepath)
    return mt5_config


def apply_runtime_config_from_yaml(filepath: str) -> None:
    """Apply non-MT5 runtime overrides supported by the credentials YAML."""
    import yaml
    from config.settings import config

    with open(filepath, "r", encoding='utf-8-sig') as f:
        data = yaml.safe_load(f) or {}

    risk_data = data.get("risk", {}) or {}
    if "max_risk_per_trade" in risk_data:
        config.risk.max_risk_per_trade = float(risk_data["max_risk_per_trade"])
    if "high_confidence_threshold" in risk_data:
        config.risk.high_confidence_threshold = float(risk_data["high_confidence_threshold"])
    if "high_confidence_lot_multiplier" in risk_data:
        config.risk.high_confidence_lot_multiplier = float(risk_data["high_confidence_lot_multiplier"])
    if "consecutive_loss_cooldown_enabled" in risk_data:
        config.risk.consecutive_loss_cooldown_enabled = _parse_bool(
            risk_data["consecutive_loss_cooldown_enabled"]
        )
    if "consecutive_loss_cooldown_count" in risk_data:
        config.risk.consecutive_loss_cooldown_count = int(risk_data["consecutive_loss_cooldown_count"])
    if "consecutive_loss_cooldown_hours" in risk_data:
        config.risk.consecutive_loss_cooldown_hours = float(risk_data["consecutive_loss_cooldown_hours"])

    confidence_data = data.get("confidence")
    if confidence_data is not None:
        config.confidence.load_from_mapping(confidence_data)
