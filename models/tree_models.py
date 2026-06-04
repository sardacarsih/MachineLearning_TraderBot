"""
============================================
Tree-Based ML Models
============================================
Concrete implementations: XGBoost, LightGBM, Random Forest, CatBoost.
All inherit from BaseModel with early stopping and feature importance.
"""

import os
import numpy as np
import joblib
from typing import Dict, Any, Optional, List

from models.base_model import BaseModel
from config.settings import config
from utils.logger import get_logger

logger = get_logger()


# ============================================
# XGBoost Model
# ============================================
class XGBoostModel(BaseModel):
    """XGBoost gradient boosting classifier for trade signal prediction."""

    def __init__(self, params: Dict = None):
        super().__init__("XGBoost")
        self.params = self._sanitize_params(params or config.model.xgb_params.copy())
        # Pop early_stopping_rounds so it isn't passed twice; we'll feed it
        # explicitly to the XGBClassifier constructor (which is the supported
        # 2.x API — passing it via .fit() is no longer supported).
        self._early_stopping_rounds = self.params.pop("early_stopping_rounds", 50)
        # XGBoost 2.x also forbids combining callbacks with early_stopping_rounds.
        self._has_validation_callback = False

    @staticmethod
    def _sanitize_params(params: Dict) -> Dict:
        """Remove binary-only XGBoost params when using multiclass training."""
        clean_params = params.copy()
        objective = clean_params.get("objective", "")
        if str(objective).startswith("multi:"):
            clean_params.pop("scale_pos_weight", None)
        return clean_params

    def train(self, X_train, y_train, X_val=None, y_val=None) -> Dict:
        from xgboost import XGBClassifier

        logger.info(f"Training XGBoost with {X_train.shape[0]} samples...")

        # Pass early_stopping_rounds at construction time (XGBoost 2.x API).
        constructor_kwargs = dict(self.params)
        if X_val is not None and y_val is not None and self._early_stopping_rounds:
            constructor_kwargs["early_stopping_rounds"] = int(self._early_stopping_rounds)
            self._has_validation_callback = True
        else:
            self._has_validation_callback = False

        self._model = XGBClassifier(**constructor_kwargs)

        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False

        self._model.fit(X_train, y_train, **fit_params)
        self._is_trained = True

        # Training metrics
        train_pred = self._model.predict(X_train)
        from sklearn.metrics import accuracy_score
        self._training_metrics = {
            "train_accuracy": float(accuracy_score(y_train, train_pred)),
            "best_iteration": getattr(self._model, 'best_iteration', -1),
        }
        logger.info(f"XGBoost training complete: {self._training_metrics}")
        return self._training_metrics

    def predict(self, X) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self._model.predict_proba(X)

    def save(self, filepath: str):
        self._ensure_dir(filepath)
        payload = {
            'model': self._model,
            'params': self.params,
        }
        payload.update(self._artifact_payload())
        joblib.dump(payload, filepath)
        logger.info(f"XGBoost model saved to {filepath}")

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self._model = data['model']
        self.params = data.get('params', {})
        self._load_feature_schema_from_artifact(data)
        self._training_metrics = data.get('training_metrics', {})
        self._is_trained = True
        logger.info(f"XGBoost model loaded from {filepath}")

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if self._model is None:
            return None
        importances = self._model.feature_importances_
        names = self._feature_names or [f"f_{i}" for i in range(len(importances))]
        return dict(sorted(zip(names, importances), key=lambda x: -x[1]))


# ============================================
# LightGBM Model
# ============================================
class LightGBMModel(BaseModel):
    """LightGBM gradient boosting classifier."""

    def __init__(self, params: Dict = None):
        super().__init__("LightGBM")
        self.params = params or config.model.lgbm_params.copy()
        # LightGBM still accepts early_stopping_rounds as a fit() callback, but
        # we keep the same constructor-style pop pattern for consistency.
        self._early_stopping_rounds = self.params.pop("early_stopping_rounds", 50)

    def train(self, X_train, y_train, X_val=None, y_val=None) -> Dict:
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation

        logger.info(f"Training LightGBM with {X_train.shape[0]} samples...")

        self._model = LGBMClassifier(**self.params)

        fit_params = {}
        callbacks = [log_evaluation(period=0)]
        if X_val is not None and y_val is not None and self._early_stopping_rounds:
            fit_params["eval_set"] = [(X_val, y_val)]
            callbacks.append(early_stopping(self._early_stopping_rounds))

        fit_params["callbacks"] = callbacks
        self._model.fit(X_train, y_train, **fit_params)
        self._is_trained = True

        train_pred = self._model.predict(X_train)
        from sklearn.metrics import accuracy_score
        self._training_metrics = {
            "train_accuracy": float(accuracy_score(y_train, train_pred)),
            "best_iteration": getattr(self._model, 'best_iteration_', -1),
        }
        logger.info(f"LightGBM training complete: {self._training_metrics}")
        return self._training_metrics

    def predict(self, X) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self._model.predict_proba(X)

    def save(self, filepath: str):
        self._ensure_dir(filepath)
        payload = {
            'model': self._model,
            'params': self.params,
        }
        payload.update(self._artifact_payload())
        joblib.dump(payload, filepath)
        logger.info(f"LightGBM model saved to {filepath}")

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self._model = data['model']
        self.params = data.get('params', {})
        self._load_feature_schema_from_artifact(data)
        self._training_metrics = data.get('training_metrics', {})
        self._is_trained = True
        logger.info(f"LightGBM model loaded from {filepath}")

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if self._model is None:
            return None
        importances = self._model.feature_importances_
        names = self._feature_names or [f"f_{i}" for i in range(len(importances))]
        return dict(sorted(zip(names, importances), key=lambda x: -x[1]))


# ============================================
# Random Forest Model
# ============================================
class RandomForestModel(BaseModel):
    """Random Forest classifier."""

    def __init__(self, params: Dict = None):
        super().__init__("RandomForest")
        self.params = params or config.model.rf_params.copy()
        if self.params.get("class_weight") == "None":
            self.params["class_weight"] = None

    def train(self, X_train, y_train, X_val=None, y_val=None) -> Dict:
        from sklearn.ensemble import RandomForestClassifier

        logger.info(f"Training Random Forest with {X_train.shape[0]} samples...")

        self._model = RandomForestClassifier(**self.params)
        self._model.fit(X_train, y_train)
        self._is_trained = True

        train_pred = self._model.predict(X_train)
        from sklearn.metrics import accuracy_score
        self._training_metrics = {
            "train_accuracy": float(accuracy_score(y_train, train_pred)),
            "n_estimators": self.params.get("n_estimators", 500),
        }
        logger.info(f"Random Forest training complete: {self._training_metrics}")
        return self._training_metrics

    def predict(self, X) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self._model.predict_proba(X)

    def save(self, filepath: str):
        self._ensure_dir(filepath)
        payload = {
            'model': self._model,
            'params': self.params,
        }
        payload.update(self._artifact_payload())
        joblib.dump(payload, filepath)
        logger.info(f"Random Forest model saved to {filepath}")

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self._model = data['model']
        self.params = data.get('params', {})
        self._load_feature_schema_from_artifact(data)
        self._training_metrics = data.get('training_metrics', {})
        self._is_trained = True
        logger.info(f"Random Forest model loaded from {filepath}")

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if self._model is None:
            return None
        importances = self._model.feature_importances_
        names = self._feature_names or [f"f_{i}" for i in range(len(importances))]
        return dict(sorted(zip(names, importances), key=lambda x: -x[1]))


# ============================================
# CatBoost Model
# ============================================
class CatBoostModel(BaseModel):
    """CatBoost gradient boosting classifier."""

    def __init__(self, params: Dict = None):
        super().__init__("CatBoost")
        self.params = params or config.model.catboost_params.copy()
        self._early_stopping_rounds = self.params.pop("early_stopping_rounds", 50)

    def train(self, X_train, y_train, X_val=None, y_val=None) -> Dict:
        from catboost import CatBoostClassifier

        logger.info(f"Training CatBoost with {X_train.shape[0]} samples...")

        self._model = CatBoostClassifier(**self.params)

        fit_params = {}
        if X_val is not None and y_val is not None and self._early_stopping_rounds:
            fit_params["eval_set"] = (X_val, y_val)
            fit_params["early_stopping_rounds"] = int(self._early_stopping_rounds)

        self._model.fit(X_train, y_train, **fit_params)
        self._is_trained = True

        train_pred = self._model.predict(X_train).flatten().astype(int)
        from sklearn.metrics import accuracy_score
        self._training_metrics = {
            "train_accuracy": float(accuracy_score(y_train, train_pred)),
            "best_iteration": getattr(self._model, 'best_iteration_', -1),
        }
        logger.info(f"CatBoost training complete: {self._training_metrics}")
        return self._training_metrics

    def predict(self, X) -> np.ndarray:
        return self._model.predict(X).flatten().astype(int)

    def predict_proba(self, X) -> np.ndarray:
        return self._model.predict_proba(X)

    def save(self, filepath: str):
        self._ensure_dir(filepath)
        payload = {
            'model': self._model,
            'params': self.params,
        }
        payload.update(self._artifact_payload())
        joblib.dump(payload, filepath)
        logger.info(f"CatBoost model saved to {filepath}")

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self._model = data['model']
        self.params = data.get('params', {})
        self._load_feature_schema_from_artifact(data)
        self._training_metrics = data.get('training_metrics', {})
        self._is_trained = True
        logger.info(f"CatBoost model loaded from {filepath}")

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if self._model is None:
            return None
        importances = self._model.feature_importances_
        names = self._feature_names or [f"f_{i}" for i in range(len(importances))]
        return dict(sorted(zip(names, importances), key=lambda x: -x[1]))


# ============================================
# Model Factory
# ============================================
def create_model(model_name: str, params: Dict = None) -> BaseModel:
    """
    Factory function to create model instances by name.

    Args:
        model_name: One of 'xgboost', 'lightgbm', 'random_forest', 'catboost'.
        params: Optional custom parameters.

    Returns:
        BaseModel instance.
    """
    model_map = {
        "xgboost": XGBoostModel,
        "lightgbm": LightGBMModel,
        "random_forest": RandomForestModel,
        "catboost": CatBoostModel,
    }

    if model_name not in model_map:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(model_map.keys())}"
        )

    return model_map[model_name](params)
