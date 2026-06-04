"""
============================================
Abstract Base Model
============================================
Base class for all ML models with common evaluation and serialization.
"""

import os
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from config.settings import config
from utils.logger import get_logger
from data.feature_schema import FeatureSchema

logger = get_logger()


class BaseModel(ABC):
    """
    Abstract base class for trading ML models.

    All model implementations (XGBoost, LightGBM, RF, CatBoost, LSTM)
    must inherit from this class and implement the abstract methods.
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = None
        self._is_trained = False
        self._feature_names: List[str] = []
        self._feature_schema: Optional[FeatureSchema] = None
        self._training_metrics: Dict[str, Any] = {}

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """
        Train the model.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features (for early stopping).
            y_val: Validation labels.

        Returns:
            Dictionary with training metrics.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature matrix.

        Returns:
            Array of predicted class labels.
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Array of shape (n_samples, n_classes) with probabilities.
        """
        pass

    @abstractmethod
    def save(self, filepath: str):
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self, filepath: str):
        """Load model from disk."""
        pass

    @abstractmethod
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores,
            or None if not available.
        """
        pass

    def set_feature_names(self, names: List[str]):
        """Store feature names for importance analysis."""
        self._feature_names = list(names)

    def set_feature_schema(
        self,
        feature_names: List[str],
        engineered_feature_names: Optional[List[str]] = None,
    ):
        """Store the complete feature schema for artifact and live validation."""
        self._feature_names = list(feature_names)
        self._feature_schema = FeatureSchema.from_feature_names(
            feature_names=self._feature_names,
            engineered_feature_names=engineered_feature_names or [],
            symbol=config.symbol.symbol,
            timeframe=config.symbol.timeframe,
        )

    def _artifact_payload(self) -> Dict[str, Any]:
        """Common metadata persisted with all model artifacts."""
        schema = self._feature_schema or FeatureSchema.from_feature_names(self._feature_names)
        return {
            "feature_names": self._feature_names,
            "feature_count": len(self._feature_names),
            "feature_schema": schema.to_dict(),
            "raw_columns": schema.raw_columns,
            "engineered_feature_names": schema.engineered_feature_names,
            "label_mapping": schema.label_mapping,
            "symbol": schema.symbol,
            "timeframe": schema.timeframe,
            "schema_version": schema.schema_version,
            "training_metrics": self._training_metrics,
        }

    def _load_feature_schema_from_artifact(self, artifact: Dict[str, Any]):
        """Load and require model feature schema metadata."""
        schema = FeatureSchema.from_artifact(artifact)
        self._feature_names = schema.feature_names
        self._feature_schema = schema

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model on test data.

        Args:
            X_test: Test features.
            y_test: Test labels.

        Returns:
            Dictionary with:
                - accuracy: Overall accuracy
                - precision_macro: Macro-averaged precision
                - recall_macro: Macro-averaged recall
                - f1_macro: Macro-averaged F1
                - precision_per_class: Per-class precision
                - recall_per_class: Per-class recall
                - f1_per_class: Per-class F1
                - confusion_matrix: Confusion matrix
                - classification_report: Full text report
        """
        if not self._is_trained:
            raise RuntimeError(f"Model {self._model_name} is not trained yet")

        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        labels = [0, 1, 2]  # NO_TRADE, BUY, SELL
        target_names = ["NO_TRADE", "BUY", "SELL"]

        results = {
            "model_name": self._model_name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision_macro": float(precision_score(
                y_test, y_pred, average='macro', zero_division=0
            )),
            "recall_macro": float(recall_score(
                y_test, y_pred, average='macro', zero_division=0
            )),
            "f1_macro": float(f1_score(
                y_test, y_pred, average='macro', zero_division=0
            )),
            "precision_per_class": precision_score(
                y_test, y_pred, average=None, labels=labels, zero_division=0
            ).tolist(),
            "recall_per_class": recall_score(
                y_test, y_pred, average=None, labels=labels, zero_division=0
            ).tolist(),
            "f1_per_class": f1_score(
                y_test, y_pred, average=None, labels=labels, zero_division=0
            ).tolist(),
            "confusion_matrix": confusion_matrix(
                y_test, y_pred, labels=labels
            ).tolist(),
            "classification_report": classification_report(
                y_test, y_pred, labels=labels,
                target_names=target_names, zero_division=0
            ),
            "predictions": y_pred,
            "probabilities": y_proba,
        }

        # Calculate BUY/SELL specific precision (most important for trading)
        buy_precision = results["precision_per_class"][1]
        sell_precision = results["precision_per_class"][2]
        results["trade_signal_precision"] = (buy_precision + sell_precision) / 2

        logger.info(
            f"{self._model_name} evaluation: "
            f"Accuracy={results['accuracy']:.4f}, "
            f"F1={results['f1_macro']:.4f}, "
            f"Trade Precision={results['trade_signal_precision']:.4f}"
        )

        return results

    def _ensure_dir(self, filepath: str):
        """Create directory for filepath if it doesn't exist."""
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
