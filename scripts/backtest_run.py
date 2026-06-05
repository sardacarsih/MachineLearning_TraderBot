"""
============================================
Backtest Execution Script
============================================
CLI tool to load a saved ML model, generate trade signals, and run backtesting.
Generates comprehensive performance reports and saves all visual charts to disk.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Adjust path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import config, normalize_timeframe
from utils.logger import setup_logging, get_logger
from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from data.feature_schema import FeatureSchema, FeatureSchemaError, FeatureValidator
from models.base_model import BaseModel
from backtest.backtester import Backtester
from backtest.performance import PerformanceAnalyzer

logger = get_logger()


def find_cached_csv(symbol: str, timeframe: str, months: int) -> str | None:
    """Find the best available cached CSV for a symbol/timeframe pair."""
    symbol_upper = symbol.upper()
    symbol_lower = symbol.lower()
    tf_lower = timeframe.lower()
    base = Path(config.paths.base_dir)
    candidates = [
        Path(config.paths.data_dir) / f"{symbol_lower}_{tf_lower}_{months}m.csv",
    ]
    if timeframe == "M5":
        candidates.extend([
            base / "data" / symbol_upper / f"{symbol_lower}_m5_{months}m.csv",
            base / "data" / f"{symbol_lower}_m5_{months}m.csv",
        ])

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    search_dirs = [
        Path(config.paths.data_dir),
        base / "data" / symbol_upper / timeframe,
        base / "data" / symbol_upper,
    ]
    matches = []
    for directory in search_dirs:
        if directory.exists():
            matches.extend(directory.glob(f"{symbol_lower}_{tf_lower}_*m.csv"))
    if not matches:
        return None
    matches = sorted(set(matches), key=lambda path: path.stat().st_mtime, reverse=True)
    return str(matches[0])


def load_saved_model(path: str) -> BaseModel:
    """Helper to load correct model instance based on filepath name."""
    path_lower = os.path.basename(path).lower()
    
    if "xgboost" in path_lower or "xgb" in path_lower:
        from models.tree_models import create_model
        model = create_model("xgboost")
    elif "lightgbm" in path_lower or "lgb" in path_lower:
        from models.tree_models import create_model
        model = create_model("lightgbm")
    elif "random" in path_lower or "rf" in path_lower:
        from models.tree_models import create_model
        model = create_model("random_forest")
    elif "cat" in path_lower:
        from models.tree_models import create_model
        model = create_model("catboost")
    elif "lstm" in path_lower:
        from models.lstm_model import LSTMModel
        model = LSTMModel()
    else:
        # Check files in folder if it's a directory
        if "lstm" in path_lower or os.path.exists(path + "_keras"):
            from models.lstm_model import LSTMModel
            model = LSTMModel()
        else:
            # Default fallback to LightGBM
            logger.warning(f"Unable to auto-detect model type from path. Defaulting to LightGBM.")
            from models.tree_models import create_model
            model = create_model("lightgbm")

    model.load(path)
    return model


def main():
    parser = argparse.ArgumentParser(description="Run historical backtesting for XAUUSD Trading Bot")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model file/directory")
    parser.add_argument("--symbol", type=str, default=None, help="Trading symbol (e.g. XAUUSD, USTEC_x100)")
    parser.add_argument("--timeframe", type=str, default=config.symbol.timeframe,
                        help="Trading timeframe (e.g. M5, M15)")
    parser.add_argument("--months", type=int, default=config.data.training_months,
                        help="Months of historical data to backtest. Defaults to training config.")
    parser.add_argument("--csv", type=str, default=None, help="Path to historical CSV data")
    parser.add_argument("--balance", type=float, default=config.backtest.initial_balance, help="Initial account balance")
    parser.add_argument("--max-spread", type=float, default=None,
                        help="Maximum allowed spread in broker points for backtest entries")
    parser.add_argument("--strategy-mode", choices=["ml", "hybrid"], default="hybrid",
                        help="Entry validation mode. 'ml' uses model signals after safety filters; "
                             "'hybrid' also requires technical rule confirmation.")
    
    args = parser.parse_args()

    setup_logging(debug=config.debug)
    logger.info("======================================================================")
    logger.info(f"Running Historical Backtest with Model: {args.model}")
    logger.info(f"Strategy Mode: {args.strategy_mode.upper()}")
    logger.info("======================================================================")

    # 1. Update config parameters
    config.backtest.initial_balance = args.balance
    config.strategy_mode = args.strategy_mode
    if args.max_spread is not None:
        config.risk.max_spread = args.max_spread
        config.filters.max_spread_points = args.max_spread
    config.set_timeframe(normalize_timeframe(args.timeframe))
    config.set_symbol(args.symbol or config.symbol.symbol)
    months = args.months if args.months > 0 else config.data.training_months
    
    if args.symbol:
        # Try to connect and fetch specifications from MT5 dynamically
        try:
            import MetaTrader5 as mt5
            if mt5.initialize():
                mt5.symbol_select(args.symbol, True)
                info = mt5.symbol_info(args.symbol)
                if info is not None:
                    config.symbol.point = info.point
                    config.symbol.digits = info.digits
                    config.symbol.contract_size = info.trade_contract_size
                    logger.info(f"Dynamically configured backtest specifications for {args.symbol}: "
                                f"Point={info.point}, Digits={info.digits}, ContractSize={info.trade_contract_size}")
                else:
                    logger.error(f"Symbol {args.symbol} not found in Market Watch. Make sure it matches broker spelling.")
                mt5.shutdown()
        except Exception as e:
            logger.warning(f"Could not dynamically query symbol properties from MT5: {e}")

    # Set per-symbol data directories (already set paths in set_symbol, but we ensure output dir matches mode)
    config.paths.backtest_dir = os.path.join(config.paths.backtest_dir, args.strategy_mode)
    os.makedirs(config.paths.backtest_dir, exist_ok=True)

    # 2. Load Saved Model
    try:
        model = load_saved_model(args.model)
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        sys.exit(1)

    # 3. Load Data
    loader = DataLoader()
    if args.csv:
        try:
            df = loader.load_from_csv(args.csv)
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            sys.exit(1)
    else:
        try:
            df = loader.load_from_mt5(months=months)
        except Exception as e:
            logger.error(f"Failed to load data from MT5: {e}")
            logger.info("Attempting to fall back to cached CSV data...")
            cache_path = find_cached_csv(config.symbol.symbol, config.symbol.timeframe, months)
            if cache_path:
                logger.info(f"Using cached CSV fallback: {cache_path}")
                df = loader.load_from_csv(cache_path)
            else:
                logger.error("No cached CSV data found. Cannot proceed.")
                sys.exit(1)

    model_features = getattr(model, '_feature_names', None)
    if not model_features:
        logger.error("Loaded model does not contain feature_names. Refusing schema-unsafe backtest.")
        sys.exit(1)

    schema = getattr(model, '_feature_schema', None) or FeatureSchema.from_feature_names(model_features)
    expects_htf = any(str(name).startswith("htf_") for name in schema.feature_names)

    # 4. Feature Engineering
    fe = FeatureEngineer(include_higher_timeframe=expects_htf)
    df_feats = fe.add_all_features(df)

    try:
        FeatureValidator.validate_dataframe(df_feats, schema)
    except FeatureSchemaError as e:
        logger.error(f"Backtesting feature schema validation failed: {e}")
        sys.exit(1)

    # 5. Generate Signals using Model
    logger.info("Generating model trading signals...")
    
    # We pass the features matrix to prediction
    is_lstm = "lstm" in model.model_name.lower()
    seq_len = getattr(model, 'sequence_length', 60)
    
    signals = []
    confidences = []
    confidence_threshold = config.resolve_confidence().signal_threshold
    
    if is_lstm:
        # Pre-populate warm-up bars as 'NO_TRADE'
        signals.extend(["NO_TRADE"] * seq_len)
        confidences.extend([0.0] * seq_len)
        
        # Predict on sequences
        X = FeatureValidator.align_matrix(df_feats, schema).values.astype(np.float32)
        proba = model.predict_proba(X)
        
        # Determine signals
        for p in proba:
            p_no_trade, p_buy, p_sell = p[0], p[1], p[2]
            action = "NO_TRADE"
            confidence = max(p_no_trade, p_buy, p_sell)
            if p_buy >= confidence_threshold and p_buy > p_sell:
                action = "BUY"
                confidence = p_buy
            elif p_sell >= confidence_threshold and p_sell > p_buy:
                action = "SELL"
                confidence = p_sell
            signals.append(action)
            confidences.append(float(confidence))
    else:
        X = FeatureValidator.align_matrix(df_feats, schema).values.astype(np.float32)
        proba = model.predict_proba(X)
        for p in proba:
            p_no_trade, p_buy, p_sell = p[0], p[1], p[2]
            action = "NO_TRADE"
            confidence = max(p_no_trade, p_buy, p_sell)
            if p_buy >= confidence_threshold and p_buy > p_sell:
                action = "BUY"
                confidence = p_buy
            elif p_sell >= confidence_threshold and p_sell > p_buy:
                action = "SELL"
                confidence = p_sell
            signals.append(action)
            confidences.append(float(confidence))
    # Count raw signals
    raw_buys = sum(1 for s in signals if s == "BUY")
    raw_sells = sum(1 for s in signals if s == "SELL")
    logger.info(f"Model generated raw signals: BUY={raw_buys}, SELL={raw_sells}")

    # 6. Run Backtest
    backtester = Backtester(initial_balance=args.balance, strategy_mode=args.strategy_mode)
    results = backtester.run(df_feats, signals, confidences=confidences)

    # 7. Analyze Performance & Export Results
    analyzer = PerformanceAnalyzer(strategy_mode=args.strategy_mode)
    analyzer.export_report(results)

    # Print summary to console
    logger.info("======================================================================")
    logger.info("Backtest Summary Results:")
    logger.info(f"  Total Trades:               {results.total_trades}")
    logger.info(f"  Win Rate:                   {results.winrate * 100:.2f}%")
    logger.info(f"  Profit Factor:              {results.profit_factor:.2f}")
    logger.info(f"  Net Profit:                 ${results.net_profit:.2f}")
    logger.info(f"  Max Drawdown:               {results.max_drawdown_pct * 100:.2f}%")
    logger.info(f"  Sharpe Ratio:               {results.sharpe_ratio:.2f}")
    logger.info("======================================================================")


if __name__ == "__main__":
    main()
