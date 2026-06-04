"""
============================================
ML Signal Generator
============================================
Takes a trained ML model and generates trade signals (BUY, SELL, NO_TRADE)
based on model prediction probabilities and confidence thresholds.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from config.settings import config
from utils.logger import get_logger, TradingLogger
from models.base_model import BaseModel
from data.feature_schema import FeatureSchema, FeatureSchemaError, FeatureValidator

logger = get_logger()


class SignalGenerator:
    """
    Generates trading signals by running real-time feature matrices through
    a trained ML model and applying confidence filters.
    """

    def __init__(self, model: BaseModel, confidence_threshold: Optional[float] = None):
        """
        Initialize the SignalGenerator.

        Args:
            model: An instance of a trained BaseModel (XGBoost, LSTM, etc.).
            confidence_threshold: Confidence threshold for executing trades (default from config).
        """
        self.model = model
        self.confidence_threshold = confidence_threshold or config.model.confidence_threshold
        logger.info(f"SignalGenerator initialized with model '{model.model_name}' and confidence threshold {self.confidence_threshold:.2f}")

    def generate_signal(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs ML prediction on the latest data.

        Args:
            features_df: DataFrame containing the latest feature-engineered rows.

        Returns:
            Dictionary with signal data:
                {
                    'action': 'BUY' | 'SELL' | 'NO_TRADE',
                    'confidence': float,
                    'model_probabilities': list,
                    'features': dict (latest features values)
                }
        """
        # Ensure model is trained
        if not getattr(self.model, '_is_trained', False):
            raise ValueError(f"Model '{self.model.model_name}' is not trained.")

        model_features = getattr(self.model, '_feature_names', None)
        if not model_features:
            raise FeatureSchemaError(
                f"Model '{self.model.model_name}' has no saved feature_names. "
                "Refusing live inference."
            )

        schema = getattr(self.model, '_feature_schema', None)
        if schema is None:
            schema = FeatureSchema.from_feature_names(model_features)

        try:
            latest_df = FeatureValidator.align_matrix(features_df, schema)
        except FeatureSchemaError as e:
            logger.error(f"Feature schema validation failed before inference: {e}")
            return {
                "action": "NO_TRADE",
                "confidence": 0.0,
                "probabilities": [1.0, 0.0, 0.0],
                "features": {},
                "reason": f"feature_schema_error: {e}",
            }

        # Check data length depending on model type
        is_lstm = "lstm" in self.model.model_name.lower()
        seq_len = getattr(self.model, 'sequence_length', 60)

        if is_lstm:
            if len(latest_df) < seq_len:
                logger.warning(f"Insufficient data for LSTM signal. Required: {seq_len}, Available: {len(latest_df)}")
                signal = {"action": "NO_TRADE", "confidence": 0.0, "probabilities": [1.0, 0.0, 0.0], "features": {}}
                TradingLogger.signal_log(f"Signal generated: {signal['action']} | Conf: {signal['confidence']:.4f} | Reason: Insufficient sequence length")
                return signal
            # Slice the last 'sequence_length' rows
            input_data = latest_df.iloc[-seq_len:].values.astype(np.float32)
        else:
            if len(latest_df) < 1:
                raise ValueError("DataFrame is empty, cannot generate signal.")
            # Slice the last row (current closed candle)
            input_data = latest_df.iloc[-1:].astype(np.float32)

        # Predict probabilities
        try:
            # predict_proba returns shape (n_sequences, 3) or (1, 3)
            proba = self.model.predict_proba(input_data)
            # Take the last prediction
            last_proba = proba[-1]
        except Exception as e:
            logger.error(f"Error predicting probabilities: {e}", exc_info=True)
            return {"action": "NO_TRADE", "confidence": 0.0, "probabilities": [1.0, 0.0, 0.0], "features": {}}

        p_no_trade, p_buy, p_sell = last_proba[0], last_proba[1], last_proba[2]

        action = "NO_TRADE"
        confidence = p_no_trade

        # Compare probabilities against threshold
        if p_buy >= self.confidence_threshold and p_buy > p_sell:
            action = "BUY"
            confidence = p_buy
        elif p_sell >= self.confidence_threshold and p_sell > p_buy:
            action = "SELL"
            confidence = p_sell
        else:
            # Default to NO_TRADE, select max of the three or p_no_trade
            max_idx = np.argmax(last_proba)
            if max_idx == 0:
                action = "NO_TRADE"
                confidence = p_no_trade
            elif max_idx == 1:
                action = "NO_TRADE"  # Below threshold
                confidence = p_buy
            else:
                action = "NO_TRADE"  # Below threshold
                confidence = p_sell

        # Latest features values for debugging
        latest_features_dict = features_df[model_features].iloc[-1].to_dict()

        signal = {
            "action": action,
            "confidence": float(confidence),
            "probabilities": last_proba.tolist(),
            "features": latest_features_dict
        }

        # Log signal using signal_log
        log_msg = (
            f"Signal Generated: {action} | Confidence: {confidence:.4f} | "
            f"Probas: [NO_TRADE: {p_no_trade:.4f}, BUY: {p_buy:.4f}, SELL: {p_sell:.4f}]"
        )
        logger.info(log_msg)
        TradingLogger.signal_log(log_msg)

        return signal
