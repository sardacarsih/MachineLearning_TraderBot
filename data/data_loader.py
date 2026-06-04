"""
============================================
Historical Data Loader
============================================
Loads OHLCV data from MetaTrader 5 or CSV cache.
Handles data validation, gap detection, and timezone normalization.
"""

import os
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional

from config.settings import config, timeframe_to_minutes
from utils.logger import get_logger
from utils.helpers import retry

logger = get_logger()


class DataLoader:
    """
    Loads and validates XAUUSD M5 historical data from MetaTrader 5
    or cached CSV files.

    Usage:
        loader = DataLoader()
        df = loader.load_from_mt5(months=12)
        # or
        df = loader.load_from_csv("data/xauusd_m5.csv")
    """

    def __init__(self, symbol: str = None, timeframe: int = None):
        """
        Initialize DataLoader.

        Args:
            symbol: Trading symbol (default from config).
            timeframe: MT5 timeframe constant (default from config).
        """
        self.symbol = symbol or config.symbol.symbol
        self.timeframe = timeframe or config.symbol.mt5_timeframe
        self._mt5_initialized = False
        logger.info(f"DataLoader initialized for {self.symbol} {config.symbol.timeframe}")

    def _init_mt5(self):
        """Initialize MetaTrader 5 connection for data loading."""
        if self._mt5_initialized:
            return True
        try:
            import MetaTrader5 as mt5
            if not mt5.initialize():
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            self._mt5_initialized = True
            logger.info("MT5 initialized for data loading")
            return True
        except ImportError:
            logger.error("MetaTrader5 package not installed. Use: pip install MetaTrader5")
            return False
        except Exception as e:
            logger.error(f"MT5 initialization error: {e}")
            return False

    def _shutdown_mt5(self):
        """Shutdown MT5 connection."""
        if self._mt5_initialized:
            try:
                import MetaTrader5 as mt5
                mt5.shutdown()
                self._mt5_initialized = False
            except Exception:
                pass

    @retry(max_attempts=3, delay=2.0)
    def load_from_mt5(self, months: int = None) -> pd.DataFrame:
        """
        Load historical data from MetaTrader 5.

        Args:
            months: Number of months of historical data to load.
                    Defaults to config.data.training_months.

        Returns:
            DataFrame with columns: time, open, high, low, close, volume, spread

        Raises:
            ConnectionError: If MT5 connection fails.
            ValueError: If insufficient data is returned.
        """
        months = months or config.data.training_months

        if not self._init_mt5():
            raise ConnectionError("Cannot connect to MetaTrader 5")

        import MetaTrader5 as mt5

        # Map timeframe string to MT5 constant
        tf_map = {
            1: mt5.TIMEFRAME_M1, 5: mt5.TIMEFRAME_M5,
            15: mt5.TIMEFRAME_M15, 30: mt5.TIMEFRAME_M30,
            60: mt5.TIMEFRAME_H1, 240: mt5.TIMEFRAME_H4,
            1440: mt5.TIMEFRAME_D1,
        }
        mt5_tf = tf_map.get(self.timeframe, mt5.TIMEFRAME_M5)

        # Calculate date range. MT5 expects UTC datetimes for range requests.
        date_to = datetime.now(timezone.utc)
        date_from = date_to - timedelta(days=months * 30)

        logger.info(
            f"Loading {self.symbol} data from {date_from.date()} to "
            f"{date_to.date()} ({months} months)"
        )

        # Select symbol in Market Watch to ensure it's active
        if not mt5.symbol_select(self.symbol, True):
            logger.warning(f"Failed to select symbol {self.symbol} in Market Watch.")

        term_info = mt5.terminal_info()
        terminal_max_bars = 90000
        if term_info is not None and hasattr(term_info, 'maxbars'):
            terminal_max_bars = max(1000, int(term_info.maxbars) - 1000)
        max_bars = min(terminal_max_bars, 95000)

        timeframe_minutes = timeframe_to_minutes(config.symbol.timeframe)
        estimated_calendar_bars = math.ceil((months * 30 * 24 * 60) / timeframe_minutes)

        # MT5 may reject large range requests with "Terminal: Invalid params"
        # when they exceed the terminal chart history limit. Use positional
        # loading directly in that case; it is the same fallback path, but
        # avoids alarming warnings for expected high-volume requests.
        if estimated_calendar_bars > max_bars:
            logger.info(
                f"Requested {estimated_calendar_bars} calendar bars for {self.symbol} "
                f"{config.symbol.timeframe}, above terminal-safe limit {max_bars}. "
                f"Fetching latest bars via copy_rates_from_pos."
            )
            rates = mt5.copy_rates_from_pos(self.symbol, mt5_tf, 0, max_bars)
        else:
            rates = mt5.copy_rates_range(self.symbol, mt5_tf, date_from, date_to)

            if rates is None or len(rates) == 0:
                error = mt5.last_error()
                logger.info(
                    f"copy_rates_range returned no data for {self.symbol} ({error}). "
                    f"Fetching latest bars via copy_rates_from_pos."
                )
                rates = mt5.copy_rates_from_pos(self.symbol, mt5_tf, 0, max_bars)

        if rates is None or len(rates) == 0:
            error = mt5.last_error()
            raise ValueError(
                f"No data received from MT5 for {self.symbol} even after fallback. Error: {error}"
            )

        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)

        # Rename columns to standard format
        df = df.rename(columns={
            'tick_volume': 'volume',
            'real_volume': 'real_volume',
        })

        # Keep essential columns
        essential_cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'spread']
        available_cols = [c for c in essential_cols if c in df.columns]
        df = df[available_cols]

        # Add spread column if not present
        if 'spread' not in df.columns:
            df['spread'] = 0

        logger.info(f"Loaded {len(df)} bars from MT5")

        # Validate data
        df = self._validate_data(df)

        return df

    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load historical data from CSV file.

        Args:
            filepath: Path to CSV file.

        Returns:
            Validated DataFrame.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file format is invalid.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")

        logger.info(f"Loading data from CSV: {filepath}")

        df = pd.read_csv(filepath)

        # Parse time column
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], utc=True)
        elif 'datetime' in df.columns:
            df['time'] = pd.to_datetime(df['datetime'], utc=True)
            df = df.drop(columns=['datetime'])
        elif 'date' in df.columns:
            df['time'] = pd.to_datetime(df['date'], utc=True)
            df = df.drop(columns=['date'])
        else:
            raise ValueError("CSV must have 'time', 'datetime', or 'date' column")

        # Validate required columns
        required = ['open', 'high', 'low', 'close']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Add volume and spread if not present
        if 'volume' not in df.columns:
            df['volume'] = 0
        if 'spread' not in df.columns:
            df['spread'] = 0

        logger.info(f"Loaded {len(df)} bars from CSV")

        df = self._validate_data(df)
        return df

    def save_to_csv(self, df: pd.DataFrame, filepath: str = None):
        """
        Save data to CSV cache.

        Args:
            df: DataFrame to save.
            filepath: Output path (default: data/<symbol>/<timeframe>/<symbol>_<timeframe>_cache.csv).
        """
        if filepath is None:
            os.makedirs(config.paths.data_dir, exist_ok=True)
            tf = config.symbol.timeframe.lower()
            filepath = os.path.join(config.paths.data_dir, f"{self.symbol.lower()}_{tf}_cache.csv")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath} ({len(df)} bars)")

    @retry(max_attempts=3, delay=1.0)
    def get_latest_bars(self, n: int = 100) -> pd.DataFrame:
        """
        Get the latest N bars for live trading predictions.

        Args:
            n: Number of recent bars to fetch.

        Returns:
            DataFrame with latest bars.
        """
        if not self._init_mt5():
            raise ConnectionError("Cannot connect to MetaTrader 5")

        import MetaTrader5 as mt5

        tf_map = {
            1: mt5.TIMEFRAME_M1, 5: mt5.TIMEFRAME_M5,
            15: mt5.TIMEFRAME_M15, 30: mt5.TIMEFRAME_M30,
            60: mt5.TIMEFRAME_H1, 240: mt5.TIMEFRAME_H4,
            1440: mt5.TIMEFRAME_D1,
        }
        mt5_tf = tf_map.get(self.timeframe, mt5.TIMEFRAME_M5)

        rates = mt5.copy_rates_from_pos(self.symbol, mt5_tf, 0, n)

        if rates is None or len(rates) == 0:
            raise ValueError(f"No recent data from MT5: {mt5.last_error()}")

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)

        if 'tick_volume' in df.columns:
            df = df.rename(columns={'tick_volume': 'volume'})

        essential = ['time', 'open', 'high', 'low', 'close', 'volume', 'spread']
        available = [c for c in essential if c in df.columns]
        df = df[available]

        if 'spread' not in df.columns:
            df['spread'] = 0

        return df

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean the data.

        Checks:
            - Remove duplicates
            - Sort by time
            - Handle NaN values
            - Detect and log gaps
            - Validate OHLC relationships

        Args:
            df: Raw DataFrame.

        Returns:
            Cleaned DataFrame.
        """
        original_len = len(df)

        # Remove duplicates
        df = df.drop_duplicates(subset='time', keep='last')
        dupes_removed = original_len - len(df)
        if dupes_removed > 0:
            logger.warning(f"Removed {dupes_removed} duplicate rows")

        # Sort by time
        df = df.sort_values('time').reset_index(drop=True)

        # Handle NaN in OHLCV
        nan_count = df[['open', 'high', 'low', 'close']].isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in OHLCV, forward-filling")
            df[['open', 'high', 'low', 'close']] = (
                df[['open', 'high', 'low', 'close']].ffill()
            )
            df = df.dropna(subset=['open', 'high', 'low', 'close'])

        # Validate OHLC relationships
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        invalid_count = invalid_ohlc.sum()
        if invalid_count > 0:
            logger.warning(
                f"Found {invalid_count} bars with invalid OHLC relationships. "
                f"Correcting..."
            )
            # Fix: recalculate high/low
            df.loc[invalid_ohlc, 'high'] = df.loc[
                invalid_ohlc, ['open', 'high', 'low', 'close']
            ].max(axis=1)
            df.loc[invalid_ohlc, 'low'] = df.loc[
                invalid_ohlc, ['open', 'high', 'low', 'close']
            ].min(axis=1)

        # Detect gaps (more than 2x expected interval)
        if len(df) > 1:
            time_diffs = df['time'].diff().dt.total_seconds()
            expected_interval = timeframe_to_minutes(config.symbol.timeframe) * 60
            gaps = time_diffs[time_diffs > expected_interval * 3]
            if len(gaps) > 0:
                logger.info(
                    f"Detected {len(gaps)} time gaps (weekend/holidays expected)"
                )

        # Ensure volume is non-negative
        if 'volume' in df.columns:
            df['volume'] = df['volume'].clip(lower=0)

        logger.info(
            f"Data validated: {len(df)} bars, "
            f"from {df['time'].iloc[0]} to {df['time'].iloc[-1]}"
        )

        return df

    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Get summary information about the dataset.

        Args:
            df: Data DataFrame.

        Returns:
            Dictionary with data statistics.
        """
        return {
            "symbol": self.symbol,
            "timeframe": f"M{self.timeframe}",
            "total_bars": len(df),
            "date_from": str(df['time'].iloc[0]),
            "date_to": str(df['time'].iloc[-1]),
            "days": (df['time'].iloc[-1] - df['time'].iloc[0]).days,
            "price_range": {
                "min": float(df['low'].min()),
                "max": float(df['high'].max()),
                "mean": float(df['close'].mean()),
            },
            "volume": {
                "mean": float(df['volume'].mean()),
                "max": float(df['volume'].max()),
            },
            "nan_count": int(df.isna().sum().sum()),
        }

    def __del__(self):
        """Cleanup MT5 connection on deletion."""
        self._shutdown_mt5()
