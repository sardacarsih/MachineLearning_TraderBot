"""
============================================
MetaTrader 5 Connector
============================================
Manages terminal initialization, login, connection status, auto-reconnection,
and status heartbeat checks. Supports context manager interface.
"""

import time
import os
from typing import Optional, Any

from config.mt5_config import mt5_config
from utils.logger import get_logger
from utils.helpers import retry

logger = get_logger()


class MT5Connector:
    """
    Handles connection lifecycle to MetaTrader 5 terminal and broker account.
    """

    def __init__(self):
        self._connected = False
        self._mt5 = None
        self._symbol = mt5_config.symbol_name

    def connect(self) -> bool:
        """
        Initialize MT5 terminal and login to broker account.

        Returns:
            True if connected and logged in, False otherwise.
        """
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5
        except ImportError:
            logger.error("MetaTrader5 python library not installed. pip install MetaTrader5")
            return False

        logger.info("Connecting to MetaTrader 5...")

        # Initialize connection
        init_kwargs = {}
        if mt5_config.terminal_path:
            init_kwargs["path"] = mt5_config.terminal_path

        if not self._mt5.initialize(**init_kwargs):
            logger.error(f"MT5 initialization failed. Error: {self._mt5.last_error()}")
            return False

        # Login to account if credentials provided
        if mt5_config.login > 0:
            logger.info(f"Logging into account {mt5_config.login} on server '{mt5_config.server}'...")
            login_success = self._mt5.login(
                login=mt5_config.login,
                password=mt5_config.password,
                server=mt5_config.server
            )
            if not login_success:
                logger.error(f"MT5 login failed. Error code: {self._mt5.last_error()}")
                self._mt5.shutdown()
                return False

        # Verify connection and terminal status
        terminal_info = self._mt5.terminal_info()
        if terminal_info is None:
            logger.error("Failed to get terminal info")
            self._mt5.shutdown()
            return False

        # Verify account details
        account_info = self._mt5.account_info()
        if account_info is None:
            logger.error("Failed to get account info")
            self._mt5.shutdown()
            return False

        logger.info(
            f"Successfully connected to MT5! "
            f"Terminal Company: {terminal_info.company}, "
            f"Account: {account_info.login}, "
            f"Balance: {account_info.balance:.2f}"
        )

        # Check if trading is enabled at account level
        if not account_info.trade_allowed:
            logger.warning("Trading is disabled at the broker account level!")
        if not terminal_info.trade_allowed:
            logger.warning("Automated trading is disabled in MetaTrader 5 settings (Allow Algo Trading)!")

        self._connected = True
        return True

    def disconnect(self):
        """Shutdown MT5 terminal connection."""
        if self._connected and self._mt5:
            self._mt5.shutdown()
            self._connected = False
            logger.info("Disconnected from MetaTrader 5 terminal")

    def is_connected(self) -> bool:
        """
        Verify connection status with a lightweight terminal check.
        """
        if not self._connected or not self._mt5:
            return False

        # Check terminal info to verify connection is alive
        info = self._mt5.terminal_info()
        if info is None:
            logger.warning("MT5 connection lost (terminal_info returned None)")
            self._connected = False
            return False

        # Check if connected to broker server
        if not info.connected:
            logger.warning("MT5 terminal running but disconnected from broker server")
            self._connected = False
            return False

        return True

    def reconnect(self) -> bool:
        """
        Attempt to reconnect to MT5 with exponential backoff.
        """
        max_attempts = mt5_config.max_reconnect_attempts
        delay = mt5_config.reconnect_delay

        logger.info(f"Reconnection triggered. Max attempts: {max_attempts}, initial delay: {delay}s")

        for attempt in range(1, max_attempts + 1):
            logger.info(f"Reconnection attempt {attempt}/{max_attempts}...")
            
            # Shutdown and sleep
            self.disconnect()
            time.sleep(delay)

            if self.connect():
                logger.info(f"Reconnected successfully on attempt {attempt}")
                return True

            # Double the delay (exponential backoff)
            delay *= 2
            logger.warning(f"Reconnection attempt {attempt} failed. Retrying in {delay}s...")

        logger.error(f"Reconnection failed after {max_attempts} attempts.")
        return False

    def heartbeat(self) -> bool:
        """
        Perform a periodic connection check and trigger auto-reconnect on failure.
        """
        if not self.is_connected():
            logger.warning("Heartbeat failed: MT5 is disconnected")
            if mt5_config.auto_reconnect:
                return self.reconnect()
            return False
        return True

    def get_symbol_info(self, symbol: str) -> Optional[Any]:
        """Fetch symbol market specifications."""
        if not self.is_connected():
            return None
        info = self._mt5.symbol_info(symbol)
        if info is None:
            logger.error(f"Failed to find symbol info for '{symbol}'. Check spelling/broker suffix.")
        return info

    def get_account_info(self) -> Optional[Any]:
        """Fetch broker account details (balance, equity, margin)."""
        if not self.is_connected():
            return None
        return self._mt5.account_info()

    def check_trading_allowed(self) -> bool:
        """
        Check if trading is permitted in settings and broker account.
        """
        if not self.is_connected():
            return False

        terminal = self._mt5.terminal_info()
        account = self._mt5.account_info()

        if not terminal or not account:
            return False

        # System check
        if not terminal.trade_allowed:
            logger.error("Algo trading is disabled in terminal settings")
            return False

        if not account.trade_allowed:
            logger.error("Broker account does not allow trading")
            return False

        # Master config flag check
        if not mt5_config.trading_enabled and not mt5_config.paper_trading:
            logger.warning("Trading is disabled in mt5_config.trading_enabled")
            return False

        return True

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
