"""
============================================
Feature Engineering Module
============================================
Calculates 20+ technical indicators and derived features for ML model input.
All features are computed from OHLCV data using vectorized pandas/numpy operations.
"""

import numpy as np
import pandas as pd
from typing import List, Optional

from config.settings import config
from utils.logger import get_logger

logger = get_logger()


RAW_COLUMNS = ["time", "open", "high", "low", "close", "volume", "spread"]
OPTIONAL_RAW_COLUMNS = {"spread"}
MODEL_EXCLUDE_COLUMNS = {"time", "label", "label_name", "close"}


class FeatureEngineer:
    """
    Comprehensive feature engineering for XAUUSD M5 trading.

    Adds technical indicators, price action features, session markers,
    support/resistance levels, and trend metrics to OHLCV data.

    Usage:
        fe = FeatureEngineer()
        df = fe.add_all_features(df)
        feature_cols = fe.get_feature_columns()
    """

    def __init__(self, cfg=None, include_higher_timeframe: Optional[bool] = None):
        """
        Initialize with configuration.

        Args:
            cfg: FeatureConfig instance. Uses global config if None.
            include_higher_timeframe: Whether to add resampled HTF context features.
        """
        self.cfg = cfg or config.features
        self.include_higher_timeframe = (
            self.cfg.include_higher_timeframe
            if include_higher_timeframe is None
            else include_higher_timeframe
        )
        self._feature_columns: List[str] = []

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all engineered features to the DataFrame.

        Args:
            df: DataFrame with columns: time, open, high, low, close, volume

        Returns:
            DataFrame with all features added. NaN rows at start are dropped.
        """
        logger.info("Starting feature engineering...")
        self._feature_columns = []
        self._validate_raw_input(df)
        df = df.copy()

        # --- Trend Indicators ---
        df = self._add_ema(df)
        df = self._add_ema_signals(df)
        df = self._add_macd(df)
        df = self._add_adx(df)

        # --- Momentum Indicators ---
        df = self._add_rsi(df)
        df = self._add_stochastic_rsi(df)
        df = self._add_roc(df)

        # --- Volatility Indicators ---
        df = self._add_atr(df)
        df = self._add_bollinger_bands(df)
        df = self._add_volatility(df)

        # --- Price Action Features ---
        df = self._add_candle_ratios(df)
        df = self._add_volume_features(df)

        # --- Structural Features ---
        df = self._add_support_resistance(df)
        df = self._add_breakout_range(df)
        df = self._add_ema_distance(df)

        # --- Session Features ---
        df = self._add_session(df)

        # --- Higher Timeframe Context ---
        if self.include_higher_timeframe:
            df = self._add_higher_timeframe_features(df)

        # Drop NaN rows created by lookback indicators
        initial_len = len(df)
        df = df.dropna().reset_index(drop=True)
        self._validate_output(df)
        dropped = initial_len - len(df)
        model_input_count = len([c for c in df.columns if c not in MODEL_EXCLUDE_COLUMNS])
        logger.info(
            f"Feature engineering complete: {len(self._feature_columns)} engineered features, "
            f"{model_input_count} model input columns, "
            f"{dropped} warmup rows dropped, {len(df)} bars remaining"
        )

        return df

    def get_feature_columns(self) -> List[str]:
        """Return list of all feature column names."""
        return self._feature_columns.copy()

    def get_model_input_columns(self, df: pd.DataFrame) -> List[str]:
        """Return model input columns using the training exclusion policy."""
        return [c for c in df.columns if c not in MODEL_EXCLUDE_COLUMNS]

    def _validate_raw_input(self, df: pd.DataFrame):
        """Validate that feature engineering starts from raw OHLCV data."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("FeatureEngineer.add_all_features expects a pandas DataFrame")

        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            raise ValueError(f"Input dataframe has duplicate columns: {duplicate_cols}")

        missing = [c for c in RAW_COLUMNS if c not in OPTIONAL_RAW_COLUMNS and c not in df.columns]
        if missing:
            raise ValueError(f"Input dataframe is missing required raw columns: {missing}")

        allowed = set(RAW_COLUMNS)
        extras = [c for c in df.columns if c not in allowed]
        if extras:
            raise ValueError(
                "FeatureEngineer.add_all_features must receive raw OHLCV data only. "
                f"Unexpected columns: {extras}"
            )

        if "time" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["time"]):
            raise TypeError("Input column 'time' must be datetime64 dtype")

    def _validate_output(self, df: pd.DataFrame):
        """Validate feature output integrity before returning it."""
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            raise ValueError(f"Feature dataframe has duplicate columns: {duplicate_cols}")

        duplicate_features = sorted({c for c in self._feature_columns if self._feature_columns.count(c) > 1})
        if duplicate_features:
            raise ValueError(f"Feature registry contains duplicate feature names: {duplicate_features}")

    def _register(self, cols):
        """Register feature column names."""
        if isinstance(cols, str):
            cols = [cols]
        duplicates = [c for c in cols if c in self._feature_columns]
        if duplicates:
            raise ValueError(f"Attempted to register duplicate feature names: {duplicates}")
        self._feature_columns.extend(cols)

    # ================================================================
    # TREND INDICATORS
    # ================================================================

    def _add_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Exponential Moving Averages (20, 50, 200)."""
        for period in [self.cfg.ema_fast, self.cfg.ema_medium, self.cfg.ema_slow]:
            col = f"ema_{period}"
            df[col] = df['close'].ewm(span=period, adjust=False).mean()
            self._register(col)
        return df

    def _add_ema_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add EMA cross and alignment signals."""
        fast, med, slow = self.cfg.ema_fast, self.cfg.ema_medium, self.cfg.ema_slow

        df['ema_fast_above_med'] = (df[f'ema_{fast}'] > df[f'ema_{med}']).astype(int)
        df['ema_med_above_slow'] = (df[f'ema_{med}'] > df[f'ema_{slow}']).astype(int)
        df['ema_alignment_bullish'] = (
            (df[f'ema_{fast}'] > df[f'ema_{med}']) &
            (df[f'ema_{med}'] > df[f'ema_{slow}'])
        ).astype(int)
        df['ema_alignment_bearish'] = (
            (df[f'ema_{fast}'] < df[f'ema_{med}']) &
            (df[f'ema_{med}'] < df[f'ema_{slow}'])
        ).astype(int)

        self._register([
            'ema_fast_above_med', 'ema_med_above_slow',
            'ema_alignment_bullish', 'ema_alignment_bearish'
        ])
        return df

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD (line, signal, histogram)."""
        fast_ema = df['close'].ewm(span=self.cfg.macd_fast, adjust=False).mean()
        slow_ema = df['close'].ewm(span=self.cfg.macd_slow, adjust=False).mean()

        df['macd_line'] = fast_ema - slow_ema
        df['macd_signal'] = df['macd_line'].ewm(
            span=self.cfg.macd_signal, adjust=False
        ).mean()
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']
        df['macd_cross_bullish'] = (
            (df['macd_line'] > df['macd_signal']) &
            (df['macd_line'].shift(1) <= df['macd_signal'].shift(1))
        ).astype(int)
        df['macd_cross_bearish'] = (
            (df['macd_line'] < df['macd_signal']) &
            (df['macd_line'].shift(1) >= df['macd_signal'].shift(1))
        ).astype(int)

        self._register([
            'macd_line', 'macd_signal', 'macd_histogram',
            'macd_cross_bullish', 'macd_cross_bearish'
        ])
        return df

    def _add_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Average Directional Index (trend strength)."""
        period = self.cfg.adx_period
        high, low, close = df['high'], df['low'], df['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smoothed averages
        atr_smooth = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
            alpha=1/period, adjust=False
        ).mean() / atr_smooth
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
            alpha=1/period, adjust=False
        ).mean() / atr_smooth

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.ewm(alpha=1/period, adjust=False).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        df['trend_strength'] = df['adx']

        self._register(['adx', 'plus_di', 'minus_di', 'trend_strength'])
        return df

    # ================================================================
    # MOMENTUM INDICATORS
    # ================================================================

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI with zone classification."""
        period = self.cfg.rsi_period
        delta = df['close'].diff()

        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))

        # RSI zones
        df['rsi_overbought'] = (df['rsi'] >= self.cfg.rsi_overbought).astype(int)
        df['rsi_oversold'] = (df['rsi'] <= self.cfg.rsi_oversold).astype(int)
        df['rsi_neutral'] = (
            ~df['rsi_overbought'].astype(bool) &
            ~df['rsi_oversold'].astype(bool)
        ).astype(int)
        # RSI momentum direction
        df['rsi_momentum'] = df['rsi'].diff(3)

        self._register([
            'rsi', 'rsi_overbought', 'rsi_oversold', 'rsi_neutral', 'rsi_momentum'
        ])
        return df

    def _add_stochastic_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Stochastic RSI."""
        period = 14
        rsi = df['rsi'] if 'rsi' in df.columns else self._calc_rsi(df, period)

        rsi_min = rsi.rolling(window=period).min()
        rsi_max = rsi.rolling(window=period).max()
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)

        df['stoch_rsi'] = stoch_rsi
        df['stoch_rsi_k'] = stoch_rsi.rolling(window=3).mean()
        df['stoch_rsi_d'] = df['stoch_rsi_k'].rolling(window=3).mean()

        self._register(['stoch_rsi', 'stoch_rsi_k', 'stoch_rsi_d'])
        return df

    def _add_roc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Rate of Change."""
        for period in [5, 10, 20]:
            col = f'roc_{period}'
            df[col] = df['close'].pct_change(periods=period) * 100
            self._register(col)
        return df

    # ================================================================
    # VOLATILITY INDICATORS
    # ================================================================

    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Average True Range."""
        period = self.cfg.atr_period
        high, low, close = df['high'], df['low'], df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        df['atr'] = tr.ewm(alpha=1/period, adjust=False).mean()
        # Normalized ATR (% of price)
        df['atr_pct'] = df['atr'] / df['close'] * 100

        self._register(['atr', 'atr_pct'])
        return df

    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands with bandwidth and %B."""
        period = self.cfg.bb_period
        std_dev = self.cfg.bb_std

        df['bb_middle'] = df['close'].rolling(window=period).mean()
        rolling_std = df['close'].rolling(window=period).std()
        df['bb_upper'] = df['bb_middle'] + (std_dev * rolling_std)
        df['bb_lower'] = df['bb_middle'] - (std_dev * rolling_std)

        # Bandwidth = (upper - lower) / middle
        df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / (
            df['bb_middle'] + 1e-10
        )
        # %B = (price - lower) / (upper - lower)
        df['bb_pct_b'] = (df['close'] - df['bb_lower']) / (
            df['bb_upper'] - df['bb_lower'] + 1e-10
        )

        self._register([
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_bandwidth', 'bb_pct_b'
        ])
        return df

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling volatility (standard deviation of returns)."""
        returns = df['close'].pct_change()
        df['volatility_5'] = returns.rolling(window=5).std()
        df['volatility_20'] = returns.rolling(window=20).std()
        df['volatility_ratio'] = df['volatility_5'] / (
            df['volatility_20'] + 1e-10
        )

        self._register(['volatility_5', 'volatility_20', 'volatility_ratio'])
        return df

    # ================================================================
    # PRICE ACTION FEATURES
    # ================================================================

    def _add_candle_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candle body ratio and wick ratios."""
        body = abs(df['close'] - df['open'])
        full_range = df['high'] - df['low']

        # Body ratio = body / full range
        df['candle_body_ratio'] = body / (full_range + 1e-10)

        # Upper wick ratio
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        df['upper_wick_ratio'] = upper_wick / (full_range + 1e-10)

        # Lower wick ratio
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        df['lower_wick_ratio'] = lower_wick / (full_range + 1e-10)

        # Candle direction (1=bullish, 0=bearish)
        df['candle_bullish'] = (df['close'] > df['open']).astype(int)

        # Candle body size relative to ATR
        if 'atr' in df.columns:
            df['body_atr_ratio'] = body / (df['atr'] + 1e-10)
            self._register('body_atr_ratio')

        self._register([
            'candle_body_ratio', 'upper_wick_ratio',
            'lower_wick_ratio', 'candle_bullish'
        ])
        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume spike detection and relative volume."""
        # Rolling average volume
        vol_ma = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / (vol_ma + 1e-10)
        df['volume_spike'] = (
            df['volume'] > self.cfg.volume_spike_threshold * vol_ma
        ).astype(int)

        # Volume trend
        df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_increasing'] = (
            df['volume_sma_5'] > df['volume_sma_5'].shift(1)
        ).astype(int)

        self._register([
            'volume_ratio', 'volume_spike',
            'volume_sma_5', 'volume_increasing'
        ])
        return df

    # ================================================================
    # STRUCTURAL FEATURES
    # ================================================================

    def _add_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add support and resistance levels using pivot points
        and swing highs/lows.
        """
        lookback = self.cfg.sr_lookback

        # Rolling support (lowest low) and resistance (highest high)
        df['resistance'] = df['high'].rolling(window=lookback).max()
        df['support'] = df['low'].rolling(window=lookback).min()

        # Distance from S/R as fraction of range
        sr_range = df['resistance'] - df['support']
        df['price_to_resistance'] = (df['resistance'] - df['close']) / (
            sr_range + 1e-10
        )
        df['price_to_support'] = (df['close'] - df['support']) / (
            sr_range + 1e-10
        )

        # Proximity flags (within 0.1% of S/R)
        threshold = self.cfg.sr_threshold
        df['near_resistance'] = (
            df['price_to_resistance'] < threshold
        ).astype(int)
        df['near_support'] = (
            df['price_to_support'] < threshold
        ).astype(int)

        # Pivot Points (classic)
        df['pivot'] = (df['high'].shift(1) + df['low'].shift(1) +
                       df['close'].shift(1)) / 3
        df['pivot_r1'] = 2 * df['pivot'] - df['low'].shift(1)
        df['pivot_s1'] = 2 * df['pivot'] - df['high'].shift(1)

        self._register([
            'resistance', 'support', 'price_to_resistance', 'price_to_support',
            'near_resistance', 'near_support', 'pivot', 'pivot_r1', 'pivot_s1'
        ])
        return df

    def _add_breakout_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add breakout range detection (price vs N-period high/low)."""
        lookback = self.cfg.breakout_lookback

        period_high = df['high'].rolling(window=lookback).max()
        period_low = df['low'].rolling(window=lookback).min()
        period_range = period_high - period_low

        # Price position within range (0=at low, 1=at high)
        df['breakout_position'] = (df['close'] - period_low) / (
            period_range + 1e-10
        )

        # Breakout flags
        df['breakout_high'] = (df['close'] > period_high.shift(1)).astype(int)
        df['breakout_low'] = (df['close'] < period_low.shift(1)).astype(int)

        # Range squeeze (narrowing range = potential breakout)
        df['range_width'] = period_range
        df['range_squeeze'] = (
            period_range < period_range.rolling(window=lookback).mean() * 0.5
        ).astype(int)

        self._register([
            'breakout_position', 'breakout_high', 'breakout_low',
            'range_width', 'range_squeeze'
        ])
        return df

    def _add_ema_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add normalized price distance from each EMA."""
        for period in [self.cfg.ema_fast, self.cfg.ema_medium, self.cfg.ema_slow]:
            ema_col = f'ema_{period}'
            if ema_col in df.columns:
                col = f'dist_ema_{period}'
                df[col] = (df['close'] - df[ema_col]) / (df[ema_col] + 1e-10) * 100
                self._register(col)
        return df

    # ================================================================
    # SESSION FEATURES
    # ================================================================

    def _add_session(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trading session one-hot encoding (UTC-based)."""
        if 'time' not in df.columns:
            logger.warning("No 'time' column for session features")
            return df

        hours = df['time'].dt.hour

        # Session classification
        df['session_asia'] = ((hours >= 0) & (hours < 8)).astype(int)
        df['session_london'] = ((hours >= 7) & (hours < 16)).astype(int)
        df['session_newyork'] = ((hours >= 12) & (hours < 21)).astype(int)
        df['session_overlap'] = ((hours >= 12) & (hours < 16)).astype(int)

        # Hour of day (cyclical encoding)
        df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hours / 24)

        # Day of week (cyclical encoding)
        dow = df['time'].dt.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * dow / 5)
        df['dow_cos'] = np.cos(2 * np.pi * dow / 5)

        self._register([
            'session_asia', 'session_london', 'session_newyork',
            'session_overlap', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
        ])
        return df

    # ================================================================
    # HIGHER TIMEFRAME FEATURES
    # ================================================================

    def _add_higher_timeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add closed-candle higher timeframe context aligned to base bars."""
        if 'time' not in df.columns:
            logger.warning("No 'time' column for higher timeframe features")
            return df

        base = df.sort_values('time').reset_index(drop=True)
        htf_frames = [
            self._build_h1_ema_trend(base),
            self._build_m15_structure(base),
            self._build_h4_atr_regime(base),
        ]

        for htf in htf_frames:
            base = pd.merge_asof(
                base.sort_values('time'),
                htf.sort_values('time'),
                on='time',
                direction='backward',
                allow_exact_matches=True,
            )

        htf_cols = [c for c in base.columns if c.startswith('htf_')]
        self._register(htf_cols)
        return base

    def _resample_ohlcv(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """
        Resample raw bars into closed HTF candles.

        A timestamp labels the candle close, so a 01:00 H1 row contains
        data from [00:00, 01:00) and is safe for a base bar at 01:00 or later.
        """
        raw_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        raw = df[raw_cols + (['spread'] if 'spread' in df.columns else [])].copy()
        if 'spread' not in raw.columns:
            raw['spread'] = 0
        raw = raw.sort_values('time').set_index('time')
        htf = raw.resample(rule, label='right', closed='left').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'spread': 'last',
        })
        return htf.dropna(subset=['open', 'high', 'low', 'close']).reset_index()

    def _build_h1_ema_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build H1 EMA trend features."""
        h1 = self._resample_ohlcv(df, '1h')
        fast, med, slow = self.cfg.ema_fast, self.cfg.ema_medium, self.cfg.ema_slow

        for period in [fast, med, slow]:
            h1[f'htf_h1_ema_{period}'] = h1['close'].ewm(span=period, adjust=False).mean()

        h1['htf_h1_ema_alignment_bullish'] = (
            (h1[f'htf_h1_ema_{fast}'] > h1[f'htf_h1_ema_{med}']) &
            (h1[f'htf_h1_ema_{med}'] > h1[f'htf_h1_ema_{slow}'])
        ).astype(int)
        h1['htf_h1_ema_alignment_bearish'] = (
            (h1[f'htf_h1_ema_{fast}'] < h1[f'htf_h1_ema_{med}']) &
            (h1[f'htf_h1_ema_{med}'] < h1[f'htf_h1_ema_{slow}'])
        ).astype(int)
        h1['htf_h1_dist_ema_200'] = (
            (h1['close'] - h1[f'htf_h1_ema_{slow}']) /
            (h1[f'htf_h1_ema_{slow}'] + 1e-10) * 100
        )

        return h1[[
            'time',
            'htf_h1_ema_20',
            'htf_h1_ema_50',
            'htf_h1_ema_200',
            'htf_h1_ema_alignment_bullish',
            'htf_h1_ema_alignment_bearish',
            'htf_h1_dist_ema_200',
        ]]

    def _build_m15_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build M15 support/resistance and breakout structure features."""
        m15 = self._resample_ohlcv(df, '15min')
        sr_lookback = self.cfg.sr_lookback
        breakout_lookback = self.cfg.breakout_lookback

        m15['htf_m15_resistance'] = m15['high'].rolling(window=sr_lookback).max()
        m15['htf_m15_support'] = m15['low'].rolling(window=sr_lookback).min()
        sr_range = m15['htf_m15_resistance'] - m15['htf_m15_support']
        m15['htf_m15_price_to_support'] = (
            (m15['close'] - m15['htf_m15_support']) / (sr_range + 1e-10)
        )
        m15['htf_m15_price_to_resistance'] = (
            (m15['htf_m15_resistance'] - m15['close']) / (sr_range + 1e-10)
        )

        period_high = m15['high'].rolling(window=breakout_lookback).max()
        period_low = m15['low'].rolling(window=breakout_lookback).min()
        period_range = period_high - period_low
        m15['htf_m15_breakout_position'] = (
            (m15['close'] - period_low) / (period_range + 1e-10)
        )
        m15['htf_m15_breakout_high'] = (m15['close'] > period_high.shift(1)).astype(int)
        m15['htf_m15_breakout_low'] = (m15['close'] < period_low.shift(1)).astype(int)
        m15['htf_m15_range_width'] = period_range
        m15['htf_m15_range_squeeze'] = (
            period_range < period_range.rolling(window=breakout_lookback).mean() * 0.5
        ).astype(int)

        return m15[[
            'time',
            'htf_m15_support',
            'htf_m15_resistance',
            'htf_m15_price_to_support',
            'htf_m15_price_to_resistance',
            'htf_m15_breakout_position',
            'htf_m15_breakout_high',
            'htf_m15_breakout_low',
            'htf_m15_range_width',
            'htf_m15_range_squeeze',
        ]]

    def _build_h4_atr_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build H4 ATR volatility regime features."""
        h4 = self._resample_ohlcv(df, '4h')
        high, low, close = h4['high'], h4['low'], h4['close']
        period = self.cfg.atr_period

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        h4['htf_h4_atr'] = tr.ewm(alpha=1/period, adjust=False).mean()
        h4['htf_h4_atr_pct'] = h4['htf_h4_atr'] / h4['close'] * 100
        atr_baseline = h4['htf_h4_atr_pct'].rolling(window=20).median()
        h4['htf_h4_atr_regime_ratio'] = h4['htf_h4_atr_pct'] / (atr_baseline + 1e-10)
        h4['htf_h4_atr_regime_high'] = (h4['htf_h4_atr_regime_ratio'] >= 1.25).astype(int)
        h4['htf_h4_atr_regime_low'] = (h4['htf_h4_atr_regime_ratio'] <= 0.75).astype(int)

        return h4[[
            'time',
            'htf_h4_atr',
            'htf_h4_atr_pct',
            'htf_h4_atr_regime_ratio',
            'htf_h4_atr_regime_high',
            'htf_h4_atr_regime_low',
        ]]

    # ================================================================
    # HELPER
    # ================================================================

    def _calc_rsi(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate RSI (helper for stochastic RSI)."""
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))
