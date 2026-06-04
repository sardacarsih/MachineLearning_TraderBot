"""
============================================
Model Trainer Pipeline
============================================
Handles data preparation (chronological split), training of tree-based
models, hyperparameter tuning (Optuna), walk-forward analysis,
feature importance extraction, and overfitting detection.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, List, Optional

from config.settings import config
from utils.logger import get_logger
from models.base_model import BaseModel
from models.tree_models import create_model
from data.feature_schema import FeatureSchema, FeatureValidator

logger = get_logger()


class ModelTrainer:
    """
    Manages the training, evaluation, walk-forward validation, and tuning
    of ML models for the XAUUSD bot.
    """

    def __init__(self):
        self.models: Dict[str, BaseModel] = {}
        self.best_model: Optional[BaseModel] = None
        self.best_model_name: Optional[str] = None
        self.feature_names: List[str] = []
        self.engineered_feature_names: List[str] = []
        logger.info("ModelTrainer initialized")

    def prepare_data(
        self, df: pd.DataFrame
    ) -> Tuple[
        np.ndarray, np.ndarray,
        np.ndarray, np.ndarray,
        np.ndarray, np.ndarray,
        List[str]
    ]:
        """
        Splits features/labels chronologically into train/val/test splits (70/15/15).

        Args:
            df: Feature-engineered and labeled DataFrame.

        Returns:
            X_train, y_train, X_val, y_val, X_test, y_test, feature_names
        """
        logger.info("Preparing data for model training...")

        # Drop time and label related columns from features
        exclude_cols = ['time', 'label', 'label_name', 'close']
        self.feature_names = [c for c in df.columns if c not in exclude_cols]
        schema = FeatureSchema.from_feature_names(self.feature_names, self.engineered_feature_names)
        FeatureValidator.validate_dataframe(df, schema)

        # Extract features and targets
        X = df[self.feature_names].values.astype(np.float32)
        y = df['label'].values.astype(np.int32)

        n_samples = len(df)
        if n_samples == 0:
            raise ValueError("Cannot prepare data: DataFrame is empty")

        # ------------------------------------------------------------------
        # Lookahead-aware split: the labelling routine uses
        # ``label_lookahead_max`` forward bars to assign each bar's target.
        # If we keep the last `lookahead_max` rows of the training set, their
        # labels are derived from bars that fall inside the validation/test
        # window — a classic leakage. Trim the tail of every split boundary
        # so labels never look across the split.
        # ------------------------------------------------------------------
        lookahead = max(0, int(getattr(config.data, "label_lookahead_max", 0)))
        ratio_sum = (
            float(config.data.train_ratio)
            + float(config.data.val_ratio)
            + float(config.data.test_ratio)
        )
        if not np.isclose(ratio_sum, 1.0, atol=1e-3):
            logger.warning(
                f"Train/Val/Test ratios sum to {ratio_sum:.4f}, expected 1.0. "
                "Normalising boundaries proportionally."
            )

        raw_train_end = int(n_samples * config.data.train_ratio)
        raw_val_end = raw_train_end + int(n_samples * config.data.val_ratio)
        # Reserve `lookahead` bars between splits so labels at the boundary
        # don't peek across. We drop them from training and from val.
        train_end = max(1, raw_train_end - lookahead)
        val_end = max(train_end + 1, raw_val_end - lookahead)

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        logger.info(
            f"Data splits created (lookahead={lookahead}):\n"
            f"  Train: X={X_train.shape}, y={y_train.shape}\n"
            f"  Val:   X={X_val.shape}, y={y_val.shape}\n"
            f"  Test:  X={X_test.shape}, y={y_test.shape}"
        )
        return X_train, y_train, X_val, y_val, X_test, y_test, self.feature_names

    def train_all_models(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """
        Trains XGBoost, LightGBM, and CatBoost.

        Returns:
            Dictionary with training results for each model.
        """
        results = {}

        # Reset models dict
        self.models = {}

        # 1. Train tree models
        tree_model_names = ["xgboost", "lightgbm", "catboost"]
        for name in tree_model_names:
            if name in config.model.models_to_train:
                try:
                    model = create_model(name)
                    model.set_feature_schema(self.feature_names, self.engineered_feature_names)
                    train_metrics = model.train(X_train, y_train, X_val, y_val)
                    self.models[model.model_name] = model
                    results[model.model_name] = train_metrics
                except Exception as e:
                    logger.opt(exception=True).error("Failed to train tree model {}: {}", name, e)

        logger.info(f"Trained models: {list(self.models.keys())}")
        return results

    def evaluate_all_models(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluates all trained models on the test set.

        Returns:
            Dictionary of metrics for each model.
        """
        evaluation_results = {}
        for name, model in self.models.items():
            try:
                logger.info(f"Evaluating {name} on test set...")
                eval_metrics = model.evaluate(X_test, y_test)
                evaluation_results[name] = eval_metrics
            except Exception as e:
                logger.error(f"Failed to evaluate model {name}: {e}", exc_info=True)

        return evaluation_results

    def get_best_model(self, evaluation_results: Dict[str, Dict[str, Any]]) -> BaseModel:
        """
        Selects the best model based on precision of trade signals (BUY and SELL).
        Precision is crucial to avoid false entry signals.
        """
        best_score = -1.0
        best_name = None

        for name, results in evaluation_results.items():
            trade_prec = results.get("trade_signal_precision", 0.0)
            logger.info(f"Model {name} Trade Signal Precision: {trade_prec:.4f}")

            # Criteria: Best precision on trades (BUY & SELL combined)
            if trade_prec > best_score:
                best_score = trade_prec
                best_name = name

        if best_name is not None:
            self.best_model = self.models[best_name]
            self.best_model_name = best_name
            logger.info(f"Best model selected: {best_name} with trade precision: {best_score:.4f}")
            return self.best_model
        else:
            raise ValueError("No models were successfully evaluated to select the best model")

    def save_best_model(self, filepath: Optional[str] = None):
        """Saves the winning model to disk."""
        if self.best_model is None:
            raise ValueError("No best model has been identified to save.")

        if filepath is None:
            filename = f"selected_{self.best_model_name.lower().replace(' ', '_')}_model"
            filepath = os.path.join(config.paths.saved_models_dir, filename)

        self.best_model.save(filepath)
        logger.info(f"Selected model saved to {filepath}")

    def save_all_models(self, prefix: str = "candidate") -> Dict[str, str]:
        """Saves every trained model and returns model names mapped to paths."""
        if not self.models:
            raise ValueError("No trained models are available to save.")

        saved_paths = {}
        for name, model in self.models.items():
            filename_name = name.lower().replace(" ", "_")
            filepath = os.path.join(config.paths.saved_models_dir, f"{prefix}_{filename_name}_model")
            model.save(filepath)
            saved_paths[name] = filepath

        logger.info(f"Saved {len(saved_paths)} trained model candidates to {config.paths.saved_models_dir}")
        return saved_paths

    def hyperparameter_tune(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray, trials: int = None,
                            timeout: int = None) -> Dict[str, Any]:
        """
        Tunes hyperparameters for a specific model using Optuna.
        
        Features:
        - Expanded search spaces with regularization, structural, and class weighting params
        - MedianPruner to prune unpromising trials early
        - Composite scoring: 0.6 * trade_precision + 0.2 * f1_macro + 0.2 * (1 - overfit_gap)
        """
        trials = trials or config.model.optuna_trials
        timeout = timeout or config.model.optuna_timeout
        logger.info(f"Starting hyperparameter tuning for {model_name} with {trials} trials (timeout: {timeout}s)...")

        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna is not installed. Hyperparameter tuning skipped. Please install optuna.")
            return {}

        def objective(trial):
            # Define expanded search space based on model type
            if model_name.lower() in ["xgboost", "xgb"]:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                    "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                    "objective": "multi:softprob",
                    "num_class": 3,
                    "eval_metric": "mlogloss",
                    "random_state": 42,
                }
                model = create_model("xgboost", params)

            elif model_name.lower() in ["lightgbm", "lgbm"]:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "num_leaves": trial.suggest_int("num_leaves", 15, 255),
                    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 150),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                    "bagging_freq": trial.suggest_int("bagging_freq", 0, 10),
                    "objective": "multiclass",
                    "num_class": 3,
                    "metric": "multi_logloss",
                    "verbose": -1,
                    "random_state": 42,
                }
                model = create_model("lightgbm", params)

            elif model_name.lower() in ["catboost", "cb"]:
                grow_policy = trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"])
                params = {
                    "iterations": trial.suggest_int("iterations", 200, 1500),
                    "depth": trial.suggest_int("depth", 4, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 30.0, log=True),
                    "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
                    "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
                    "border_count": trial.suggest_int("border_count", 32, 255),
                    "grow_policy": grow_policy,
                    "loss_function": "MultiClass",
                    "eval_metric": "MultiClass",
                    "random_seed": 42,
                    "verbose": 0,
                }
                model = create_model("catboost", params)

            else:
                raise ValueError(f"Tuning for {model_name} is not currently supported.")

            model._feature_names = self.feature_names
            model.set_feature_schema(self.feature_names, self.engineered_feature_names)

            try:
                model.train(X_train, y_train, X_val, y_val)
                metrics = model.evaluate(X_val, y_val)
            except Exception as e:
                logger.debug(f"Trial failed: {e}")
                return 0.0

            # Composite scoring: balance trade precision, trade F1 (to prevent 0-recall models), and generalization
            trade_precision = metrics.get("trade_signal_precision", 0.0)
            
            buy_f1 = metrics.get("f1_per_class", [0.0, 0.0, 0.0])[1]
            sell_f1 = metrics.get("f1_per_class", [0.0, 0.0, 0.0])[2]
            trade_f1 = (buy_f1 + sell_f1) / 2.0

            # Quick overfit check: train accuracy vs val accuracy
            try:
                train_acc = np.mean(model.predict(X_train) == y_train)
                val_acc = metrics.get("accuracy", 0.0)
                overfit_gap = max(0.0, train_acc - val_acc)
            except Exception:
                overfit_gap = 0.0

            # Weighted composite: balanced between trade precision, trade F1, and overfit gap
            composite_score = (
                0.40 * trade_precision +
                0.40 * trade_f1 +
                0.20 * max(0.0, 1.0 - overfit_gap * 2)  # Penalize gaps > 0.5
            )

            return composite_score

        # Use MedianPruner to abort unpromising trials early
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
        study = optuna.create_study(direction="maximize", pruner=pruner)
        study.optimize(objective, n_trials=trials, timeout=timeout)

        logger.info(f"[{model_name}] Tuning finished after {len(study.trials)} trials.")
        logger.info(f"[{model_name}] Best composite score: {study.best_value:.4f}")
        logger.info(f"[{model_name}] Best parameters: {study.best_params}")
        return study.best_params

    def tune_all_models(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        model_names: List[str] = None,
        trials: int = None,
        timeout: int = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Tunes hyperparameters for multiple models and stores results.

        Args:
            X_train, y_train: Training data.
            X_val, y_val: Validation data.
            model_names: List of models to tune (defaults to all tree models in config).
            trials: Number of Optuna trials per model.
            timeout: Max seconds per model.

        Returns:
            Dictionary mapping model names to their best parameter sets.
        """
        tunable_models = ["xgboost", "lightgbm", "catboost"]
        model_names = model_names or [m for m in config.model.models_to_train if m in tunable_models]

        all_best_params = {}
        for model_name in model_names:
            logger.info(f"{'='*60}")
            logger.info(f"TUNING: {model_name.upper()}")
            logger.info(f"{'='*60}")
            try:
                best_params = self.hyperparameter_tune(
                    model_name, X_train, y_train, X_val, y_val,
                    trials=trials, timeout=timeout
                )
                if best_params:
                    all_best_params[model_name] = best_params
                    # Apply tuned params to config for subsequent training
                    self._apply_tuned_params(model_name, best_params)
            except Exception as e:
                logger.opt(exception=True).error("Tuning failed for {}: {}", model_name, e)

        logger.info(f"{'='*60}")
        logger.info(f"TUNING COMPLETE — {len(all_best_params)} models optimized")
        logger.info(f"{'='*60}")
        return all_best_params

    def _apply_tuned_params(self, model_name: str, best_params: Dict[str, Any]):
        """Applies Optuna's best params into the global config for subsequent training."""
        param_map = {
            "xgboost": config.model.xgb_params,
            "lightgbm": config.model.lgbm_params,
            "catboost": config.model.catboost_params,
        }
        target = param_map.get(model_name.lower())
        if target is not None:
            # Only update keys that exist in the search space (avoid overwriting fixed keys)
            for k, v in best_params.items():
                if k == "class_weight" and v == "None":
                    v = None
                target[k] = v
            logger.info(f"Applied {len(best_params)} tuned params to {model_name} config")

    def walk_forward_analysis(
        self, df: pd.DataFrame, model_name: str, k_splits: int = None
    ) -> List[Dict[str, Any]]:
        """
        Performs walk-forward cross validation with K splits on the dataset.
        Walk-forward is critical for validating financial time-series models.

        The training window ends ``label_lookahead_max`` bars before the test
        window starts, otherwise the last training rows would carry labels
        derived from bars inside the test set (label leakage).

        Args:
            df: Entire labeled dataframe.
            model_name: Name of model to evaluate.
            k_splits: Number of splits.

        Returns:
            List of metrics dictionaries for each split.
        """
        k_splits = k_splits or config.model.walk_forward_splits
        logger.info(f"Running walk-forward analysis for {model_name} with {k_splits} splits...")

        # Feature separation
        exclude_cols = ['time', 'label', 'label_name', 'close']
        feats = [c for c in df.columns if c not in exclude_cols]
        X = df[feats].values.astype(np.float32)
        y = df['label'].values.astype(np.int32)

        n_samples = len(df)
        split_size = n_samples // (k_splits + 1)
        if split_size <= 0:
            logger.warning("Not enough samples for walk-forward analysis; skipping.")
            return []

        # Reserve lookahead bars so labels at the train/test boundary cannot
        # peek into the test window. We trim the tail of every training window.
        lookahead = max(0, int(getattr(config.data, "label_lookahead_max", 0)))
        results = []

        for i in range(k_splits):
            # Expansive window approach: Train grows, Test moves forward
            train_end = split_size * (i + 1)
            test_end = train_end + split_size
            if test_end > n_samples:
                logger.info(
                    f"Walk-forward fold {i+1} would exceed available samples; stopping at fold {i}."
                )
                break

            # Trim `lookahead` bars off the end of the training window so its
            # labels stay inside the training domain.
            safe_train_end = max(1, train_end - lookahead)

            X_tr, y_tr = X[:safe_train_end], y[:safe_train_end]
            X_te, y_te = X[train_end:test_end], y[train_end:test_end]

            # Allocate 15% of training window for validation (for early stopping)
            if len(X_tr) < 50:
                logger.warning(
                    f"Skipping fold {i+1}: training window too small after lookahead trim ({len(X_tr)} rows)."
                )
                continue
            val_split = int(len(X_tr) * 0.85)
            val_split = max(1, min(val_split, len(X_tr) - 1))
            X_tr_fold, y_tr_fold = X_tr[:val_split], y_tr[:val_split]
            X_val_fold, y_val_fold = X_tr[val_split:], y_tr[val_split:]

            logger.info(
                f"Fold {i+1}/{k_splits}: Train={X_tr_fold.shape[0]}, Val={X_val_fold.shape[0]}, "
                f"Test={X_te.shape[0]} (lookahead_trim={lookahead})"
            )

            try:
                # Re-instantiate model
                model = create_model(model_name)

                model.set_feature_schema(feats, self.engineered_feature_names)
                model.train(X_tr_fold, y_tr_fold, X_val_fold, y_val_fold)
                metrics = model.evaluate(X_te, y_te)

                fold_result = {
                    "fold": i + 1,
                    "accuracy": metrics["accuracy"],
                    "precision_macro": metrics["precision_macro"],
                    "recall_macro": metrics["recall_macro"],
                    "f1_macro": metrics["f1_macro"],
                    "trade_signal_precision": metrics["trade_signal_precision"],
                }
                results.append(fold_result)
                logger.info(f"Fold {i+1} Results: Acc={fold_result['accuracy']:.4f}, Trade Prec={fold_result['trade_signal_precision']:.4f}")
            except Exception as e:
                logger.error(f"Error in walk-forward fold {i+1}: {e}", exc_info=True)

        return results

    def feature_importance_analysis(self, out_path: Optional[str] = None) -> Dict[str, float]:
        """
        Plots and returns feature importances for the best model.
        Only applicable to tree-based models.
        """
        if self.best_model is None:
            raise ValueError("No best model trained.")

        importances = self.best_model.get_feature_importance()
        if importances is None:
            logger.info("Feature importance is not available for this model type.")
            return {}

        # Plot top 20 features
        top_n = 20
        sorted_imp = list(importances.items())[:top_n]
        features, scores = zip(*sorted_imp)

        plt.figure(figsize=(12, 8))
        sns.barplot(x=list(scores), y=list(features), palette="viridis")
        plt.title(f"Top {top_n} Feature Importances — {self.best_model_name}")
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.tight_layout()

        if out_path is None:
            out_path = os.path.join(config.paths.backtest_dir, "feature_importance.png")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
        plt.close()
        logger.info(f"Feature importance plot saved to {out_path}")

        return importances

    def detect_overfitting(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Compares training performance vs test performance to detect overfitting.
        """
        overfitting_metrics = {}
        for name, model in self.models.items():
            try:
                train_acc = np.mean(model.predict(X_train) == y_train)
                test_acc = np.mean(model.predict(X_test) == y_test)

                diff = train_acc - test_acc
                overfitting_metrics[name] = {
                    "train_accuracy": float(train_acc),
                    "test_accuracy": float(test_acc),
                    "difference": float(diff),
                    "overfitted": bool(diff > 0.15)  # Threshold 15%
                }
                logger.info(
                    f"{name} Overfitting check: Train Acc={train_acc:.4f}, "
                    f"Test Acc={test_acc:.4f}, Diff={diff:.4f} "
                    f"({'OVERFITTED' if diff > 0.15 else 'OK'})"
                )
            except Exception as e:
                logger.error(f"Failed to check overfitting for {name}: {e}")

        return overfitting_metrics
