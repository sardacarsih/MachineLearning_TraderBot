"""
============================================
LSTM Model (Stub)
============================================

Sequence-based classifier. Currently a stub — full implementation is
intentionally deferred because the trading pipeline ships with tree-based
classifiers (XGBoost, LightGBM, CatBoost) which dominate tabular
feature engineering of OHLCV bars.

The stub is present so that:

1. ``scripts.backtest_run.load_saved_model`` can detect a saved LSTM
   artifact (by name) without crashing on an unresolved import.
2. Live code that branches on ``is_lstm = "lstm" in model_name.lower()``
   can still load the model class and call ``load()`` against a future
   artifact.

The methods raise :class:`NotImplementedError` with a clear message
explaining that the LSTM training path is not yet wired in. Tree
classifiers (XGBoost / LightGBM / CatBoost) remain the production
default.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from models.base_model import BaseModel
from config.settings import config
from utils.logger import get_logger

logger = get_logger()


class LSTMModel(BaseModel):
    """Sequence classifier stub. Not yet implemented."""

    DEFAULT_SEQUENCE_LENGTH = 60

    def __init__(self, params: Optional[Dict] = None):
        super().__init__("LSTM")
        self.params = params or {}
        self.sequence_length: int = int(
            self.params.get("sequence_length", self.DEFAULT_SEQUENCE_LENGTH)
        )
        self._model = None
        logger.warning(
            "LSTMModel instantiated but not implemented. Tree classifiers "
            "(XGBoost / LightGBM / CatBoost) are the production default."
        )

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------
    def train(self, X_train, y_train, X_val=None, y_val=None) -> Dict[str, Any]:
        raise NotImplementedError(
            "LSTM training is not implemented in this build. "
            "Use XGBoost, LightGBM, or CatBoost for tabular OHLCV features."
        )

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError("LSTM inference is not implemented in this build.")

    def predict_proba(self, X) -> np.ndarray:
        raise NotImplementedError("LSTM inference is not implemented in this build.")

    def save(self, filepath: str) -> None:
        # Persist metadata only so reload still works for inspection.
        self._ensure_dir(filepath)
        payload: Dict[str, Any] = {
            "model": None,
            "params": self.params,
            "sequence_length": self.sequence_length,
        }
        payload.update(self._artifact_payload())
        import joblib
        joblib.dump(payload, filepath)
        logger.info(f"LSTM stub saved (no trained weights) to {filepath}")

    def load(self, filepath: str) -> None:
        import joblib
        data = joblib.load(filepath)
        self.params = data.get("params", {})
        self.sequence_length = int(
            data.get("sequence_length", self.DEFAULT_SEQUENCE_LENGTH)
        )
        self._load_feature_schema_from_artifact(data)
        self._training_metrics = data.get("training_metrics", {})
        # We do NOT mark the model as trained — every inference call will
        # raise NotImplementedError until a real implementation lands.
        self._is_trained = False
        logger.info(
            f"LSTM stub loaded from {filepath}; inference calls will raise "
            "NotImplementedError until a real implementation is added."
        )

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        return None
