"""
Currency conversion helpers for MT5 account risk sizing.
"""

from typing import Optional, Tuple

from config.mt5_config import mt5_config
from utils.logger import get_logger

logger = get_logger()


class CurrencyConverter:
    """Converts account-currency risk budgets into symbol quote currency."""

    def __init__(self, connector):
        self.connector = connector

    def resolve_account_currency(self) -> str:
        configured = (mt5_config.account_currency or "AUTO").upper()
        if configured != "AUTO":
            return configured

        if self.connector.is_connected():
            info = self.connector.get_account_info()
            currency = getattr(info, "currency", None) if info is not None else None
            if currency:
                return str(currency).upper()

        logger.warning("Could not auto-detect account currency. Falling back to USD.")
        return "USD"

    def get_usd_idr_rate(self) -> Tuple[float, str]:
        if (mt5_config.usd_idr_rate_mode or "broker").lower() == "manual":
            manual_rate = float(mt5_config.usd_idr_manual_rate)
            if manual_rate <= 0:
                raise ValueError("Manual USDIDR rate must be greater than zero")
            logger.info(f"Using manual USDIDR rate: {manual_rate:.2f}")
            return manual_rate, "manual"

        if self.connector.is_connected():
            mt5 = getattr(self.connector, "_mt5", None)
            for symbol in mt5_config.usd_idr_symbol_candidates:
                tick = mt5.symbol_info_tick(symbol) if mt5 is not None else None
                bid = float(getattr(tick, "bid", 0.0) or 0.0) if tick is not None else 0.0
                ask = float(getattr(tick, "ask", 0.0) or 0.0) if tick is not None else 0.0
                if bid > 0 and ask > 0:
                    return (bid + ask) / 2.0, symbol

        logger.warning(
            f"USDIDR broker quote unavailable. Using fallback rate: {mt5_config.usd_idr_fallback_rate:.2f}"
        )
        return float(mt5_config.usd_idr_fallback_rate), "fallback"

    def convert_risk_amount(
        self,
        risk_amount: float,
        account_currency: Optional[str] = None,
        risk_quote_currency: Optional[str] = None,
    ) -> Tuple[float, str, str, Optional[float]]:
        account_ccy = (account_currency or self.resolve_account_currency()).upper()
        quote_ccy = (risk_quote_currency or mt5_config.risk_quote_currency or "USD").upper()

        if account_ccy == quote_ccy:
            return float(risk_amount), account_ccy, quote_ccy, None

        if account_ccy == "IDR" and quote_ccy == "USD":
            usd_idr, source = self.get_usd_idr_rate()
            if usd_idr <= 0:
                raise ValueError("USDIDR conversion rate must be greater than zero")
            converted = float(risk_amount) / usd_idr
            logger.info(
                f"Risk conversion: {risk_amount:.2f} IDR ~= {converted:.2f} USD "
                f"(USDIDR={usd_idr:.2f}, source={source})"
            )
            return converted, account_ccy, quote_ccy, usd_idr

        raise ValueError(f"Unsupported risk currency conversion: {account_ccy} -> {quote_ccy}")
