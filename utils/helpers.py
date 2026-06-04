"""
============================================
Utility Helper Functions
============================================
Common utilities used across the trading system.
"""

import os
import time
import functools
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Callable, Any


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0,
          exceptions: tuple = (Exception,)):
    """
    Decorator for retrying a function on failure with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts.
        delay: Initial delay between retries (seconds).
        backoff: Multiplier for delay on each retry.
        exceptions: Tuple of exception types to catch.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            from utils.logger import get_logger
            logger = get_logger()
            current_delay = delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise
                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} "
                        f"failed: {e}. Retrying in {current_delay:.1f}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator


def calculate_lot_size(balance: float, risk_pct: float, sl_points: float,
                       point_value: float, contract_size: float = 100.0,
                       min_lot: float = 0.01, max_lot: float = 100.0,
                       lot_step: float = 0.01) -> float:
    """
    Calculate position size based on risk percentage.

    Args:
        balance: Account balance.
        risk_pct: Risk percentage (e.g., 0.01 for 1%).
        sl_points: Stop loss distance in points.
        point_value: Value per point per lot.
        contract_size: Contract size (100 oz for gold).
        min_lot: Minimum allowed lot size.
        max_lot: Maximum allowed lot size.
        lot_step: Lot size increment.

    Returns:
        Calculated lot size rounded to lot_step.
    """
    if sl_points <= 0:
        return min_lot

    risk_amount = balance * risk_pct
    # For XAUUSD: 1 lot = 100 oz, point = 0.01
    # Value per point per lot = contract_size * point = 100 * 0.01 = 1.0
    lot_size = risk_amount / (sl_points * point_value)

    # Round to lot step
    lot_size = round(lot_size / lot_step) * lot_step
    # Clamp to min/max
    lot_size = max(min_lot, min(max_lot, lot_size))

    return round(lot_size, 2)


def normalize_features(df: pd.DataFrame, columns: list,
                       method: str = "zscore") -> pd.DataFrame:
    """
    Normalize feature columns.

    Args:
        df: DataFrame with features.
        columns: Columns to normalize.
        method: 'zscore' or 'minmax'.

    Returns:
        DataFrame with normalized columns.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        if method == "zscore":
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col] = (df[col] - mean) / std
        elif method == "minmax":
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
    return df


def get_session(hour_utc: int) -> str:
    """
    Determine trading session from UTC hour.

    Args:
        hour_utc: Hour in UTC (0-23).

    Returns:
        Session name: 'asia', 'london', 'newyork', or 'closed'.
    """
    if 0 <= hour_utc < 8:
        return "asia"
    elif 7 <= hour_utc < 12:
        return "london"
    elif 12 <= hour_utc < 16:
        return "london_ny_overlap"
    elif 16 <= hour_utc < 21:
        return "newyork"
    else:
        return "closed"


def pips_to_points(pips: float, point: float = 0.01) -> float:
    """Convert pips to points for XAUUSD."""
    # For XAUUSD, 1 pip = 0.10, 1 point = 0.01
    return pips * 10 * point


def points_to_price(points: float, point: float = 0.01) -> float:
    """Convert points to price difference."""
    return points * point


def format_price(price: float, digits: int = 2) -> str:
    """Format price with correct decimal places."""
    return f"{price:.{digits}f}"


def timestamp_to_datetime(timestamp: int) -> datetime:
    """Convert Unix timestamp to datetime."""
    return datetime.fromtimestamp(timestamp)


def safe_division(numerator: float, denominator: float,
                  default: float = 0.0) -> float:
    """Safe division that returns default on zero denominator."""
    if denominator == 0:
        return default
    return numerator / denominator


def calculate_sharpe_ratio(returns: pd.Series,
                           risk_free_rate: float = 0.0,
                           periods: int = 252 * 24 * 12) -> float:
    """
    Calculate annualized Sharpe ratio for M5 timeframe.

    Args:
        returns: Series of returns.
        risk_free_rate: Annual risk-free rate.
        periods: Number of periods per year (M5 = 252*24*12).

    Returns:
        Annualized Sharpe ratio.
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / periods
    return np.sqrt(periods) * excess_returns.mean() / excess_returns.std()


def calculate_max_drawdown(equity_curve: pd.Series) -> tuple:
    """
    Calculate maximum drawdown from equity curve.

    Returns:
        Tuple of (max_drawdown_pct, max_drawdown_amount, peak_idx, trough_idx).
    """
    if len(equity_curve) == 0:
        return 0.0, 0.0, None, None

    rolling_max = equity_curve.expanding().max()
    drawdown = equity_curve / rolling_max - 1.0

    max_dd_pct = drawdown.min()
    trough_idx = drawdown.idxmin()

    # Find the peak before the trough
    peak_idx = equity_curve[:trough_idx].idxmax()

    max_dd_amount = equity_curve[peak_idx] - equity_curve[trough_idx]

    return abs(max_dd_pct), max_dd_amount, peak_idx, trough_idx


def calculate_profit_factor(gross_profit: float,
                            gross_loss: float) -> float:
    """Calculate profit factor."""
    return safe_division(gross_profit, abs(gross_loss), default=0.0)


def calculate_expectancy(wins: int, losses: int,
                         avg_win: float, avg_loss: float) -> float:
    """
    Calculate trading expectancy.

    Expectancy = (Win% * Avg Win) - (Loss% * Avg Loss)
    """
    total = wins + losses
    if total == 0:
        return 0.0
    win_rate = wins / total
    loss_rate = losses / total
    return (win_rate * avg_win) - (loss_rate * abs(avg_loss))
