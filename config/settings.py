"""
============================================
XAUUSD M5 ML Trading Bot - Central Configuration
============================================
All trading parameters, ML hyperparameters, risk limits, and session settings.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import time

TIMEFRAME_MINUTES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}


def normalize_timeframe(timeframe: str) -> str:
    """Normalize a user-provided timeframe into the internal label."""
    tf = str(timeframe).strip().upper()
    if tf.isdigit():
        tf = f"M{tf}"
    if tf not in TIMEFRAME_MINUTES:
        supported = ", ".join(TIMEFRAME_MINUTES)
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Supported: {supported}")
    return tf


def timeframe_to_minutes(timeframe: str) -> int:
    """Return the number of minutes represented by a timeframe label."""
    return TIMEFRAME_MINUTES[normalize_timeframe(timeframe)]


# ============================================
# SYMBOL & TIMEFRAME
# ============================================
@dataclass
class SymbolConfig:
    """Symbol and timeframe configuration."""
    symbol: str = "XAUUSD"
    timeframe: str = "M5"
    # MT5 timeframe constant (TIMEFRAME_M5 = 5)
    mt5_timeframe: int = 5
    point: float = 0.01          # XAUUSD point size
    digits: int = 2              # Price decimal places
    contract_size: float = 100.0 # 1 lot = 100 oz
    min_lot: float = 0.01
    max_lot: float = 100.0
    lot_step: float = 0.01


# ============================================
# DATA CONFIGURATION
# ============================================
@dataclass
class DataConfig:
    """Data loading and processing settings."""
    # Training data range (months)
    training_months: int = 12
    # Minimum bars required
    min_bars: int = 50000
    # Train/Validation/Test split ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    # Forward-looking window for labeling (candles)
    label_lookahead_min: int = 5
    label_lookahead_max: int = 10
    # Risk-reward ratio for label generation
    reward_risk_ratio: float = 1.5
    # ATR period for SL calculation in labeling
    atr_period_label: int = 14
    # ATR multiplier for stop loss in labeling
    atr_sl_multiplier: float = 1.5


# ============================================
# FEATURE ENGINEERING
# ============================================
@dataclass
class FeatureConfig:
    """Technical indicator parameters."""
    # Higher timeframe feature context
    include_higher_timeframe: bool = True
    # EMA periods
    ema_fast: int = 20
    ema_medium: int = 50
    ema_slow: int = 200
    # RSI
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    # ATR
    atr_period: int = 14
    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    # Volume spike threshold (multiplier of average)
    volume_spike_threshold: float = 1.5
    # Support/Resistance lookback
    sr_lookback: int = 50
    sr_threshold: float = 0.001  # Price proximity threshold
    # Breakout range lookback
    breakout_lookback: int = 20
    # Trend strength (ADX period)
    adx_period: int = 14


# ============================================
# TRADING SESSIONS (UTC)
# ============================================
@dataclass
class SessionConfig:
    """Trading session times in UTC."""
    # Asian Session
    asia_start: time = field(default_factory=lambda: time(0, 0))
    asia_end: time = field(default_factory=lambda: time(8, 0))
    # London Session
    london_start: time = field(default_factory=lambda: time(7, 0))
    london_end: time = field(default_factory=lambda: time(16, 0))
    # New York Session
    newyork_start: time = field(default_factory=lambda: time(12, 0))
    newyork_end: time = field(default_factory=lambda: time(21, 0))
    # Preferred trading sessions (London + NY overlap is best)
    preferred_sessions: List[str] = field(
        default_factory=lambda: ["london", "newyork", "london_ny_overlap"]
    )


# ============================================
# ML MODEL CONFIGURATION
# ============================================
@dataclass
class ModelConfig:
    """Machine learning model parameters."""
    # Confidence threshold for trade signals.
    confidence_threshold: float = 0.50
    # Models to train
    models_to_train: List[str] = field(
        default_factory=lambda: [
            "xgboost", "lightgbm", "catboost"
        ]
    )
    # XGBoost defaults
    xgb_params: Dict = field(default_factory=lambda: {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "early_stopping_rounds": 50,
        "random_state": 42,
    })
    # LightGBM defaults
    lgbm_params: Dict = field(default_factory=lambda: {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "verbose": -1,
        "random_state": 42,
    })
    # Random Forest defaults
    rf_params: Dict = field(default_factory=lambda: {
        "n_estimators": 500,
        "max_depth": 10,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    })
    # CatBoost defaults
    catboost_params: Dict = field(default_factory=lambda: {
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3,
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "random_seed": 42,
        "verbose": 0,
        "early_stopping_rounds": 50,
    })

    # Hyperparameter tuning
    optuna_trials: int = 100
    optuna_timeout: int = 600  # Max seconds per model tuning (safety net)
    # Walk-forward analysis
    walk_forward_splits: int = 5
    walk_forward_train_ratio: float = 0.8


# ============================================
# RISK MANAGEMENT
# ============================================
@dataclass
class RiskConfig:
    """Risk management parameters."""
    # Maximum risk per trade (% of balance)
    max_risk_per_trade: float = 0.01  # 1%
    # Maximum daily drawdown (% of balance)
    max_daily_drawdown: float = 0.05  # 5%
    # Consecutive losses before stopping
    max_consecutive_losses: int = 3
    # ATR multiplier for stop loss
    atr_sl_multiplier: float = 1.5
    # ATR multiplier for take profit (1.5x of SL = RR 1:1.5)
    atr_tp_multiplier: float = 2.25  # 1.5 * 1.5
    # Trailing stop activation (% of TP reached)
    trailing_stop_activation: float = 0.5  # 50% of TP
    # Trailing stop distance (ATR multiplier)
    trailing_stop_atr: float = 1.0
    # Maximum spread allowed (in points)
    max_spread: float = 220.0  # Broker points for XAUUSD
    # Maximum slippage (in points)
    max_slippage: int = 10
    # Maximum open positions
    max_open_positions: int = 3
    # Minimum time between trades (minutes)
    min_trade_interval: int = 5
    # Confidence above this threshold gets a larger position size.
    high_confidence_threshold: float = 0.70
    # Lot multiplier applied after risk sizing for high-confidence signals.
    high_confidence_lot_multiplier: float = 2.0


# ============================================
# FILTER CONFIGURATION
# ============================================
@dataclass
class FilterConfig:
    """Trading filters to avoid bad setups."""
    # Spread filter
    spread_filter_enabled: bool = True
    max_spread_points: float = 220.0
    # Volatility filter
    volatility_filter_enabled: bool = True
    min_atr_threshold: float = 0.5   # Min ATR for sufficient volatility
    max_atr_threshold: float = 12000.0   # Max ATR to avoid extreme volatility
    # News filter
    news_filter_enabled: bool = True
    news_avoid_minutes_before: int = 30
    news_avoid_minutes_after: int = 15
    # Ranging/sideways filter
    ranging_filter_enabled: bool = True
    adx_ranging_threshold: float = 20.0  # ADX below = ranging
    # Session filter
    session_filter_enabled: bool = False
    allowed_sessions: List[str] = field(
        default_factory=lambda: ["london", "newyork", "asia"]
    )


# ============================================
# BACKTEST CONFIGURATION
# ============================================
@dataclass
class BacktestConfig:
    """Backtesting parameters."""
    initial_balance: float = 10000.0
    commission_per_lot: float = 7.0   # USD per lot round trip
    slippage_points: float = 2.0
    # Performance thresholds
    min_winrate: float = 0.55
    min_profit_factor: float = 1.5
    max_drawdown_pct: float = 0.20    # 20% max
    min_sharpe_ratio: float = 1.0
    min_trades: int = 100             # Minimum trades for valid backtest


# ============================================
# PATHS
# ============================================
@dataclass
class PathConfig:
    """File and directory paths."""
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir: str = ""
    models_dir: str = ""
    backtest_dir: str = ""
    logs_dir: str = ""
    saved_models_dir: str = ""
    _symbol_key: str = ""

    def __post_init__(self):
        self.data_dir = os.path.join(self.base_dir, "data")
        self.models_dir = os.path.join(self.base_dir, "models")
        self.backtest_dir = os.path.join(self.base_dir, "backtest")
        self.logs_dir = os.path.join(self.base_dir, "logs")
        self.saved_models_dir = os.path.join(self.base_dir, "saved_models")
        # Create directories
        for d in [self.data_dir, self.models_dir, self.backtest_dir,
                  self.logs_dir, self.saved_models_dir]:
            os.makedirs(d, exist_ok=True)

    def set_symbol(self, symbol: str, timeframe: str = None):
        """Set per-symbol/timeframe subdirectories for data isolation.
        
        Each symbol/timeframe pair gets its own subdirectory under data/,
        saved_models/, backtest/, and logs/ to prevent runs from overwriting
        each other's files.
        """
        timeframe = normalize_timeframe(timeframe or "M5")
        symbol_dir = symbol.upper()
        key = f"{symbol_dir}/{timeframe}"
        if self._symbol_key == key:
            return
        self.data_dir = os.path.join(self.base_dir, "data", symbol_dir, timeframe)
        self.backtest_dir = os.path.join(self.base_dir, "backtest", symbol_dir, timeframe)
        self.logs_dir = os.path.join(self.base_dir, "logs", symbol_dir, timeframe)
        self.saved_models_dir = os.path.join(self.base_dir, "saved_models", symbol_dir, timeframe)
        # Create symbol-specific directories
        for d in [self.data_dir, self.backtest_dir,
                  self.logs_dir, self.saved_models_dir]:
            os.makedirs(d, exist_ok=True)
        self._symbol_key = key


# ============================================
# MASTER CONFIGURATION
# ============================================
@dataclass
class TradingConfig:
    """Master configuration combining all sub-configs."""
    symbol: SymbolConfig = field(default_factory=SymbolConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    sessions: SessionConfig = field(default_factory=SessionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    # Trading mode
    mode: str = "backtest"  # "backtest", "paper", "live"
    strategy_mode: str = "hybrid"  # "ml" or "hybrid"
    # Debug mode
    debug: bool = False

    def set_timeframe(self, timeframe: str):
        """Apply a timeframe label to the active symbol configuration."""
        tf = normalize_timeframe(timeframe)
        self.symbol.timeframe = tf
        self.symbol.mt5_timeframe = timeframe_to_minutes(tf)

    def set_symbol(self, symbol: str):
        """Set symbol, isolate paths, and adjust volatility/filter parameters."""
        raw_symbol = str(symbol).strip()
        sym = raw_symbol.upper()
        self.symbol.symbol = raw_symbol
        self.paths.set_symbol(sym, self.symbol.timeframe)
        
        # Adjust filters based on symbol
        if sym == "XAGUSD":
            self.filters.min_atr_threshold = 0.02
            self.filters.max_atr_threshold = 10.0
            self.risk.max_spread = 220.0
            self.filters.max_spread_points = 220.0
        elif sym == "XAUUSD":
            self.filters.min_atr_threshold = 0.5
            self.filters.max_atr_threshold = 12000.0
            self.risk.max_spread = 220.0
            self.filters.max_spread_points = 220.0
        elif "USTEC" in sym:
            self.filters.min_atr_threshold = 1.0
            self.filters.max_atr_threshold = 50000.0
            self.risk.max_spread = 1500.0
            self.filters.max_spread_points = 1500.0
        elif sym == "GBPJPY":
            self.filters.min_atr_threshold = 0.05
            self.filters.max_atr_threshold = 5.0
            self.risk.max_spread = 50.0
            self.filters.max_spread_points = 50.0
        elif sym == "EURUSD":
            self.filters.min_atr_threshold = 0.00005
            self.filters.max_atr_threshold = 0.01
            self.risk.max_spread = 20.0
            self.filters.max_spread_points = 20.0


# Global config instance
config = TradingConfig()
