"""
============================================
ML Training Orchestrator Script
============================================
CLI tool to train and evaluate ML trading models for XAUUSD M5.
Features: Data loading, Feature engineering, Label generation, Model training,
Overfitting checks, Hyperparameter tuning, Walk-forward analysis, and Comparison reporting.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Adjust path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import config, normalize_timeframe
from utils.logger import setup_logging, get_logger
from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from data.labeling import LabelGenerator
from models.model_trainer import ModelTrainer
from models.model_comparison import ModelComparison

# Setup logging
logger = get_logger()

SUPPORTED_TRAINING_MODELS = {"xgboost", "lightgbm", "catboost"}


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


def main():
    parser = argparse.ArgumentParser(description="Train ML models for XAUUSD Trading Bot")
    parser.add_argument("--symbol", type=str, default=None, help="Trading symbol (e.g. XAUUSD, USTEC_x100)")
    parser.add_argument("--timeframe", type=str, default=config.symbol.timeframe,
                        help="Trading timeframe (e.g. M5, M15)")
    parser.add_argument("--months", type=int, default=config.data.training_months, help="Months of training data")
    parser.add_argument("--csv", type=str, default=None, help="Path to historical CSV data (skips MT5 loading)")
    parser.add_argument("--models", nargs="+", default=config.model.models_to_train,
                        help="List of models to train: xgboost lightgbm catboost")
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning with Optuna")
    parser.add_argument("--tune-trials", type=int, default=None, help="Number of Optuna trials per model")
    parser.add_argument("--tune-models", nargs="+", default=None,
                        help="List of models to tune (defaults to all configured models)")
    parser.add_argument("--walk-forward", action="store_true", help="Enable walk-forward analysis validation")
    parser.add_argument("--balance", action="store_true", help="Balance training classes before fitting")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration for model training")
    
    args = parser.parse_args()

    setup_logging(debug=config.debug)
    logger.info("======================================================================")
    logger.info("Starting ML Training Pipeline")
    logger.info("======================================================================")

    # 1. Update config parameters based on arguments
    config.data.training_months = args.months
    requested_models = [str(model).strip().lower() for model in args.models]
    ignored_models = [model for model in requested_models if model not in SUPPORTED_TRAINING_MODELS]
    selected_models = [model for model in requested_models if model in SUPPORTED_TRAINING_MODELS]
    if ignored_models:
        logger.warning(f"Ignoring unsupported training models: {ignored_models}")
    if not selected_models:
        logger.error(f"No supported models selected. Available: {sorted(SUPPORTED_TRAINING_MODELS)}")
        sys.exit(1)
    config.model.models_to_train = selected_models
    config.set_timeframe(normalize_timeframe(args.timeframe))
    
    if args.gpu:
        logger.info("Enabling GPU acceleration for supported models...")
        # XGBoost GPU settings
        config.model.xgb_params["tree_method"] = "hist"
        config.model.xgb_params["device"] = "cuda"
        # CatBoost GPU settings
        config.model.catboost_params["task_type"] = "GPU"
        # LightGBM GPU settings (requires GPU compilation of LightGBM)
        config.model.lgbm_params["device"] = "gpu"

    config.set_symbol(args.symbol or config.symbol.symbol)

    if args.symbol:
        # Try to connect and fetch specifications from MT5 dynamically
        try:
            import MetaTrader5 as mt5
            if mt5.initialize():
                # Select the symbol in market watch first
                mt5.symbol_select(args.symbol, True)
                info = mt5.symbol_info(args.symbol)
                if info is not None:
                    config.symbol.point = info.point
                    config.symbol.digits = info.digits
                    config.symbol.contract_size = info.trade_contract_size
                    logger.info(f"Dynamically configured training specifications for {args.symbol}: "
                                f"Point={info.point}, Digits={info.digits}, ContractSize={info.trade_contract_size}")
                else:
                    logger.error(f"Symbol {args.symbol} not found in Market Watch. Make sure it matches broker spelling.")
                mt5.shutdown()
        except Exception as e:
            logger.warning(f"Could not dynamically query symbol properties from MT5: {e}")

    logger.info(f"Data directories set for: {config.symbol.symbol} {config.symbol.timeframe}")

    # Re-initialize logging to use per-symbol log directory
    setup_logging(log_dir=config.paths.logs_dir, debug=config.debug)

    # 2. Data Loading
    loader = DataLoader()
    if args.csv:
        try:
            df = loader.load_from_csv(args.csv)
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            sys.exit(1)
    else:
        try:
            df = loader.load_from_mt5(months=args.months)
            # Cache the data for future use
            symbol_lower = config.symbol.symbol.lower()
            tf = config.symbol.timeframe.lower()
            cache_path = os.path.join(config.paths.data_dir, f"{symbol_lower}_{tf}_{args.months}m.csv")
            loader.save_to_csv(df, cache_path)
        except Exception as e:
            logger.error(f"Failed to load data from MT5: {e}")
            logger.info("Attempting to fall back to cached CSV data...")
            cache_path = find_cached_csv(config.symbol.symbol, config.symbol.timeframe, args.months)
            if cache_path:
                logger.info(f"Using cached CSV fallback: {cache_path}")
                df = loader.load_from_csv(cache_path)
            else:
                logger.error("No cached CSV data found. Cannot proceed without data.")
                sys.exit(1)

    logger.info(f"Loaded dataset: {len(df)} rows")

    # 3. Feature Engineering
    fe = FeatureEngineer()
    df_feats = fe.add_all_features(df)

    # 4. Label Generation
    lg = LabelGenerator()
    df_labeled = lg.generate_labels(df_feats)
    
    # Show label distribution
    dist = lg.get_label_distribution(df_labeled)
    import json
    logger.info(f"Label class distribution:\n{json.dumps(dist, indent=4)}")

    # 5. Initialize Model Trainer
    trainer = ModelTrainer()
    trainer.engineered_feature_names = fe.get_feature_columns()
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = trainer.prepare_data(df_labeled)

    # Apply class balancing if requested
    if args.balance:
        logger.info("Balancing training classes...")
        # Since X_train is numpy array, we balance it.
        # Simple random undersampling to balance classes for simplicity
        # (Avoid importing imblearn if not strictly installed, fallback to manual balance)
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logger.info(f"SMOTE finished: Balanced Train shape X={X_train.shape}, y={y_train.shape}")
        except ImportError:
            logger.warning("imblearn not installed. Running simple manual undersampling...")
            # Simple undersampling logic
            min_class_size = min([np.sum(y_train == c) for c in [0, 1, 2]])
            indices = []
            for c in [0, 1, 2]:
                c_indices = np.where(y_train == c)[0]
                selected = np.random.choice(c_indices, min_class_size, replace=False)
                indices.extend(selected)
            np.random.shuffle(indices)
            X_train, y_train = X_train[indices], y_train[indices]
            logger.info(f"Manual balance shape: X={X_train.shape}, y={y_train.shape}")

    # 6. Optuna Hyperparameter Tuning
    if args.tune:
        logger.info("Running hyperparameter tuning...")
        try:
            models_to_tune = args.tune_models or config.model.models_to_train
            tunable_models = sorted(SUPPORTED_TRAINING_MODELS)
            models_to_tune = [m for m in models_to_tune if m in tunable_models]
            
            if models_to_tune:
                trainer.tune_all_models(
                    X_train, y_train, X_val, y_val,
                    model_names=models_to_tune,
                    trials=args.tune_trials
                )
            else:
                logger.warning("No tunable models selected for tuning.")
        except Exception as e:
            logger.error(f"Error during hyperparameter tuning: {e}", exc_info=True)

    # 7. Model Training
    logger.info("Training all configured models...")
    train_metrics = trainer.train_all_models(X_train, y_train, X_val, y_val)

    # 8. Model Evaluation
    logger.info("Evaluating models on test set...")
    eval_results = trainer.evaluate_all_models(X_test, y_test)

    # 9. Overfitting Detection
    logger.info("Running overfitting checks...")
    overfit_metrics = trainer.detect_overfitting(X_train, y_train, X_test, y_test)

    # 10. Walk-Forward Analysis
    if args.walk_forward:
        logger.info("Running Walk-Forward Cross Validation...")
        for model_name in args.models:
            wf_results = trainer.walk_forward_analysis(df_labeled, model_name)
            if wf_results:
                avg_prec = np.mean([r['trade_signal_precision'] for r in wf_results])
                logger.info(f"Walk-Forward {model_name} Avg Trade Precision: {avg_prec:.4f}")

    # 11. Model Comparison & Report Export
    mc = ModelComparison()
    comp_df = mc.compare_models(eval_results)
    
    # Save confusion matrices, ROC curves, PR curves
    mc.plot_confusion_matrices(eval_results)
    mc.plot_roc_curves(eval_results, y_test)
    mc.plot_precision_recall(eval_results, y_test)
    
    # Generate text report
    report = mc.generate_report(eval_results, comp_df)

    # 12. Save Best Model
    best_model = trainer.get_best_model(eval_results)
    trainer.save_all_models()
    trainer.save_best_model()

    # Export feature importance for tree model
    trainer.feature_importance_analysis()

    logger.info("======================================================================")
    logger.info("ML Training Pipeline Completed Successfully!")
    logger.info(f"Best Model Selected: {trainer.best_model_name}")
    logger.info(f"Report and charts saved to: {config.paths.backtest_dir}")
    logger.info("======================================================================")


if __name__ == "__main__":
    main()
