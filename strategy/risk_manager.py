"""
============================================
Risk Manager
============================================
Controls position sizing, daily drawdown, consecutive losses, and
stop-loss / take-profit calculations based on ATR and config.
"""

from typing import Dict, Any, List, Optional, Union

from config.settings import config
from utils.logger import get_logger, TradingLogger

logger = get_logger()


class RiskManager:
    """
    Implements all trading risk rules to protect capital.
    Includes dynamic lot sizing, daily drawdown checks, consecutive loss halts,
    and ATR-based SL/TP/Trailing Stop calculations.
    """

    def __init__(self):
        self.daily_pnl: float = 0.0
        self.daily_trade_count: int = 0
        self.consecutive_losses: int = 0
        self.start_of_day_balance: Optional[float] = None
        self.last_block_reason: str = ""
        logger.info("RiskManager initialized")

    @staticmethod
    def _lot_step_decimals(step: float) -> int:
        """Return decimal places needed to represent a broker lot step."""
        step_text = f"{step:.10f}".rstrip("0").rstrip(".")
        if "." not in step_text:
            return 0
        return len(step_text.split(".", 1)[1])

    def reset_daily(self, current_balance: float):
        """Reset daily counters at the start of the trading day."""
        self.daily_pnl = 0.0
        self.daily_trade_count = 0
        self.start_of_day_balance = current_balance
        self.last_block_reason = ""
        logger.info(f"Daily risk metrics reset. Starting balance: {current_balance:.2f}")

    def update_trade_result(self, pnl: float):
        """
        Update risk counters after a trade closes.

        Args:
            pnl: Realized profit or loss in quote currency.
        """
        self.daily_pnl += pnl
        self.daily_trade_count += 1

        if pnl < 0:
            self.consecutive_losses += 1
            logger.info(f"Trade lost: {pnl:.2f}. Consecutive losses: {self.consecutive_losses}")
        else:
            self.consecutive_losses = 0
            logger.info(f"Trade won: {pnl:.2f}. Consecutive losses reset to 0")

    def calculate_position_size(
        self,
        balance: float,
        sl_distance: float,
        risk_amount_quote: Optional[float] = None,
        account_currency: str = "USD",
        risk_quote_currency: str = "USD",
        conversion_rate: Optional[float] = None,
    ) -> float:
        """
        Calculates dynamic lot size risking a fixed percentage of balance.

        Formula:
            Lot Size = (Balance * Risk%) / (SL Distance * Contract Size)

        Args:
            balance: Current account balance.
            sl_distance: Distance to stop loss in price (e.g. 5.50 for Gold).
            risk_amount_quote: Optional risk budget already converted to the
                symbol risk currency. When omitted, balance is treated as being
                denominated in the symbol risk currency for backward-compatible
                backtests.

        Returns:
            Lot size rounded to the symbol's step and clipped to limits.
        """
        if sl_distance <= 0:
            logger.warning(f"Invalid stop loss distance: {sl_distance}. Using min lot.")
            return config.symbol.min_lot

        risk_pct = config.risk.max_risk_per_trade
        account_risk_amount = balance * risk_pct
        risk_amount = float(risk_amount_quote) if risk_amount_quote is not None else account_risk_amount
        contract_size = config.symbol.contract_size

        # Dynamic lot sizing
        raw_lot = risk_amount / (sl_distance * contract_size)

        # Round to step
        step = config.symbol.lot_step
        lot_size = round(raw_lot / step) * step

        # Clip limits
        lot_size = max(config.symbol.min_lot, min(config.symbol.max_lot, lot_size))

        # Precision formatting
        decimals = self._lot_step_decimals(step)
        lot_size = round(lot_size, decimals)

        if risk_amount_quote is not None and account_currency.upper() != risk_quote_currency.upper():
            rate_msg = f", Rate={conversion_rate:.2f}" if conversion_rate else ""
            logger.info(
                f"Lot Calculation: Bal={balance:.2f} {account_currency.upper()}, "
                f"Risk={account_risk_amount:.2f} {account_currency.upper()} ~= "
                f"{risk_amount:.2f} {risk_quote_currency.upper()}{rate_msg}, "
                f"SL Dist={sl_distance:.4f}, Raw Lot={raw_lot:.4f}, Filled Lot={lot_size}"
            )
        else:
            logger.info(
                f"Lot Calculation: Bal={balance:.2f}, Risk={risk_amount:.2f} {risk_quote_currency.upper()}, "
                f"SL Dist={sl_distance:.4f}, Raw Lot={raw_lot:.4f}, Filled Lot={lot_size}"
            )
        return lot_size

    def apply_confidence_lot_multiplier(self, lot_size: float, confidence: Optional[float]) -> float:
        """
        Increase lot size for high-confidence trade signals.

        The multiplier is applied after base risk sizing, then rounded to the
        configured lot step and clipped to the symbol's lot bounds.
        """
        if confidence is None:
            return lot_size

        threshold = config.risk.high_confidence_threshold
        multiplier = config.risk.high_confidence_lot_multiplier
        if confidence <= threshold or multiplier <= 1.0:
            return lot_size

        boosted_lot = lot_size * multiplier
        step = config.symbol.lot_step
        boosted_lot = round(boosted_lot / step) * step
        boosted_lot = max(config.symbol.min_lot, min(config.symbol.max_lot, boosted_lot))

        decimals = self._lot_step_decimals(step)
        boosted_lot = round(boosted_lot, decimals)

        logger.info(
            f"High-confidence lot multiplier applied: Conf={confidence:.4f} > {threshold:.4f}, "
            f"Multiplier={multiplier:.2f}, Base Lot={lot_size}, Final Lot={boosted_lot}"
        )
        return boosted_lot

    def check_daily_drawdown(self, current_balance: float) -> bool:
        """
        Checks if the daily drawdown exceeds the configured limit.

        Returns:
            True if drawdown is within limits, False if exceeded.
        """
        if self.start_of_day_balance is None:
            self.start_of_day_balance = current_balance
            return True

        if current_balance < self.start_of_day_balance:
            drawdown_pct = (self.start_of_day_balance - current_balance) / self.start_of_day_balance
            if drawdown_pct >= config.risk.max_daily_drawdown:
                self.last_block_reason = (
                    "Daily drawdown limit exceeded: "
                    f"start={self.start_of_day_balance:.2f}, current={current_balance:.2f}, "
                    f"drawdown={drawdown_pct * 100:.2f}%, limit={config.risk.max_daily_drawdown * 100:.2f}%"
                )
                logger.warning(self.last_block_reason)
                return False

        # Also check PnL drawdown from start of day balance
        if self.daily_pnl < 0 and abs(self.daily_pnl) >= (self.start_of_day_balance * config.risk.max_daily_drawdown):
            self.last_block_reason = (
                "Daily PnL drawdown limit exceeded: "
                f"daily_pnl={self.daily_pnl:.2f}, "
                f"limit={self.start_of_day_balance * config.risk.max_daily_drawdown:.2f}"
            )
            logger.warning(self.last_block_reason)
            return False

        return True

    def check_consecutive_losses(self) -> bool:
        """Checks if consecutive losses exceed the limit (default 3)."""
        if self.consecutive_losses >= config.risk.max_consecutive_losses:
            self.last_block_reason = (
                f"Consecutive loss limit exceeded: {self.consecutive_losses} losses "
                f"(Limit: {config.risk.max_consecutive_losses})"
            )
            logger.warning(self.last_block_reason)
            return False
        return True

    def calculate_sl(self, entry_price: float, atr: float, direction: str) -> float:
        """
        Calculate stop loss price based on ATR.

        Args:
            entry_price: Market entry price.
            atr: Current ATR(14) value.
            direction: 'BUY' or 'SELL'.
        """
        sl_multiplier = config.risk.atr_sl_multiplier
        sl_distance = atr * sl_multiplier

        if direction == "BUY":
            sl_price = entry_price - sl_distance
        else:
            sl_price = entry_price + sl_distance

        return round(sl_price, config.symbol.digits)

    def calculate_tp(self, entry_price: Union[float, int], sl_distance: float, direction: str) -> float:
        """
        Calculate take profit price based on stop loss distance (R:R ratio).

        Args:
            entry_price: Entry price.
            sl_distance: Distance of SL from entry.
            direction: 'BUY' or 'SELL'.
        """
        tp_multiplier = config.data.reward_risk_ratio  # 1.5
        tp_distance = sl_distance * tp_multiplier

        if direction == "BUY":
            tp_price = entry_price + tp_distance
        else:
            tp_price = entry_price - tp_distance

        return round(tp_price, config.symbol.digits)

    def calculate_trailing_stop(
        self, entry_price: float, current_price: float, current_sl: float,
        atr: float, direction: str, tp_price: float
    ) -> Optional[float]:
        """
        Calculate trailing stop loss price if active.
        Activates when price has covered 50% of the target profit.

        Args:
            entry_price: Trade open price.
            current_price: Current market price.
            current_sl: Current stop loss price.
            atr: Current ATR(14) value.
            direction: 'BUY' or 'SELL'.
            tp_price: Take profit price.

        Returns:
            New stop loss price if it should be modified, else None.
        """
        tp_distance = abs(tp_price - entry_price)
        activation_price = entry_price + (tp_distance * config.risk.trailing_stop_activation) if direction == "BUY" else entry_price - (tp_distance * config.risk.trailing_stop_activation)

        trail_distance = atr * config.risk.trailing_stop_atr

        if direction == "BUY":
            # Check if price has crossed activation threshold
            if current_price >= activation_price:
                new_sl = current_price - trail_distance
                # Stop loss can only move up
                if new_sl > current_sl and new_sl > entry_price:
                    return round(new_sl, config.symbol.digits)
        else:
            # Check if price has crossed activation threshold
            if current_price <= activation_price:
                new_sl = current_price + trail_distance
                # Stop loss can only move down
                if new_sl < current_sl and new_sl < entry_price:
                    return round(new_sl, config.symbol.digits)

        return None

    def can_trade(self, current_balance: float) -> bool:
        """
        Master check to determine if trading is allowed based on risk metrics.
        """
        self.last_block_reason = ""
        if not self.check_daily_drawdown(current_balance):
            return False

        if not self.check_consecutive_losses():
            return False

        return True
