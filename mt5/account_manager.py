"""
============================================
Account Manager
============================================
Queries account state, balance, equity, free margin, margin level,
and tracks daily P&L and recent trade history from MT5.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from config.mt5_config import mt5_config
from config.settings import config
from utils.logger import get_logger
from mt5.connector import MT5Connector

logger = get_logger()


class AccountManager:
    """
    Manages account state querying (balance, equity, margin) and transaction history.
    """

    def __init__(self, connector: MT5Connector):
        self.connector = connector
        self.paper_mode = mt5_config.paper_trading
        
        # Virtual account trackers for paper trading offline
        if self.paper_mode:
            from mt5.paper_db import PaperDBManager
            self._db = PaperDBManager()
            state = self._db.get_account_state(config.backtest.initial_balance)
            self._virtual_balance = state["balance"]
            self._virtual_equity = state["equity"]
            logger.info(f"Virtual balance loaded from persistent SQLite DB: {self._virtual_balance:.2f}")
        else:
            self._db = None
            self._virtual_balance = config.backtest.initial_balance
            self._virtual_equity = config.backtest.initial_balance
            
        logger.info(f"AccountManager initialized. Paper Trading Mode: {self.paper_mode}")


    def get_balance(self) -> float:
        """Returns the current account balance."""
        if self.connector.is_connected():
            info = self.connector.get_account_info()
            if info:
                return float(info.balance)
        return self._virtual_balance

    def get_equity(self) -> float:
        """Returns the current account equity (balance + open trade profit)."""
        if self.connector.is_connected():
            info = self.connector.get_account_info()
            if info:
                return float(info.equity)
        return self._virtual_equity

    def get_free_margin(self) -> float:
        """Returns the current free margin."""
        if self.connector.is_connected():
            info = self.connector.get_account_info()
            if info:
                return float(info.margin_free)
        return self._virtual_balance  # Default offline value

    def check_margin_level(self, required_margin: float = 0.0) -> bool:
        """
        Verify if account has sufficient margin level for placing trades.
        """
        if not self.connector.is_connected():
            return True  # Skip offline

        info = self.connector.get_account_info()
        if info is None:
            return False

        # Margin level percentage = (equity / margin) * 100
        # Margin call level is usually around 100%, stop out at 50%
        if info.margin > 0:
            margin_level = (info.equity / info.margin) * 100
            if margin_level < 150.0:
                logger.warning(f"Low margin level warning: {margin_level:.2f}% (Limit: 150%)")
                return False

        # Check free margin vs required
        if required_margin > 0 and info.margin_free < required_margin:
            logger.warning(f"Insufficient free margin. Available: {info.margin_free:.2f}, Required: {required_margin:.2f}")
            return False

        return True

    def get_daily_pnl(self) -> float:
        """
        Calculate today's realized profit and loss.
        """
        if not self.connector.is_connected():
            return 0.0

        import MetaTrader5 as mt5

        # Query history from start of today (local or server time)
        now = datetime.now()
        start_of_day = datetime(now.year, now.month, now.day)

        # Get historical deals (trades)
        deals = mt5.history_deals_get(start_of_day, now)
        if deals is None:
            logger.warning(f"Failed to fetch historical deals. Error: {mt5.last_error()}")
            return 0.0

        daily_pnl = 0.0
        for deal in deals:
            # Filter deals by our magic number if specified
            if getattr(deal, "magic", 0) == mt5_config.magic_number or mt5_config.magic_number == 0:
                # Include profit, commission, and swap
                profit = getattr(deal, "profit", 0.0)
                commission = getattr(deal, "commission", 0.0)
                swap = getattr(deal, "swap", 0.0)
                daily_pnl += (profit + commission + swap)

        return daily_pnl

    def get_trade_history(self, days: int = 1) -> List[Dict[str, Any]]:
        """
        Fetch trade history for the past N days.
        """
        if not self.connector.is_connected():
            return []

        import MetaTrader5 as mt5

        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)

        deals = mt5.history_deals_get(from_date, to_date)
        if deals is None or len(deals) == 0:
            return []

        reason_labels = {
            getattr(mt5, "DEAL_REASON_TP", object()): "TP",
            getattr(mt5, "DEAL_REASON_SL", object()): "SL",
            getattr(mt5, "DEAL_REASON_CLIENT", object()): "MANUAL",
            getattr(mt5, "DEAL_REASON_MOBILE", object()): "MANUAL",
            getattr(mt5, "DEAL_REASON_WEB", object()): "MANUAL",
            getattr(mt5, "DEAL_REASON_EXPERT", object()): "EXPERT",
        }

        trades = []
        for deal in deals:
            # Skip entry orders (we only care about deals that close or open positions)
            # entry: 0 = IN (opening), 1 = OUT (closing), 2 = IN/OUT (reverse)
            entry = getattr(deal, "entry", -1)
            reason = getattr(deal, "reason", None)
            
            # Check magic number
            if getattr(deal, "magic", 0) == mt5_config.magic_number or mt5_config.magic_number == 0:
                trades.append({
                    "ticket": int(deal.ticket),
                    "order": int(deal.order),
                    "position_id": int(getattr(deal, "position_id", 0) or 0),
                    "time": datetime.fromtimestamp(deal.time).strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": deal.symbol,
                    "type": "BUY" if deal.type == mt5.DEAL_TYPE_BUY else "SELL",
                    "entry": "IN" if entry == 0 else "OUT" if entry == 1 else "REV",
                    "volume": float(deal.volume),
                    "price": float(deal.price),
                    "commission": float(deal.commission),
                    "swap": float(deal.swap),
                    "profit": float(deal.profit),
                    "magic": int(deal.magic),
                    "reason": int(reason) if reason is not None else None,
                    "reason_label": reason_labels.get(reason, "UNKNOWN"),
                    "comment": getattr(deal, "comment", "")
                })

        return trades

    def get_account_summary(self) -> Dict[str, Any]:
        """
        Generates a summary dictionary of the account's state.
        """
        balance = self.get_balance()
        equity = self.get_equity()
        free_margin = self.get_free_margin()
        daily_pnl = self.get_daily_pnl()

        summary = {
            "balance": balance,
            "equity": equity,
            "free_margin": free_margin,
            "margin_level": 0.0,
            "daily_pnl": daily_pnl,
            "trade_allowed": False,
            "paper_trading": self.paper_mode
        }

        if self.connector.is_connected():
            info = self.connector.get_account_info()
            if info:
                summary["margin_level"] = float(info.margin_level) if info.margin > 0 else 0.0
                summary["trade_allowed"] = bool(info.trade_allowed)

        return summary

    def apply_paper_pnl(self, pnl: float) -> None:
        """
        Update the virtual paper-trading balance and equity after a closed trade.
        No-op when running live against a real broker.

        Args:
            pnl: Realised net profit (already net of commission) in quote currency.
        """
        if not self.paper_mode:
            return
        self._virtual_balance = float(self._virtual_balance) + float(pnl)
        # Equity follows balance until open PnL is reported separately by the
        # order executor. The risk manager's start-of-day check is the
        # authoritative drawdown reference, not this number.
        self._virtual_equity = self._virtual_balance
        logger.info(
            f"[PAPER] Virtual balance updated by {pnl:+.2f} -> {self._virtual_balance:.2f}"
        )
        if self._db:
            self._db.save_account_state(self._virtual_balance, self._virtual_equity)
