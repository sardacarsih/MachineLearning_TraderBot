from copy import deepcopy
import unittest

import numpy as np
import pandas as pd

from config.settings import config
from data.feature_schema import FeatureSchema
from strategy.signal_generator import SignalGenerator


class DummyModel:
    model_name = "dummy"

    def __init__(self, probabilities):
        self._is_trained = True
        self._feature_names = ["f1"]
        self._feature_schema = FeatureSchema.from_feature_names(["f1"])
        self._probabilities = probabilities

    def predict_proba(self, _):
        return np.array([self._probabilities], dtype=np.float32)


class SignalGeneratorConfidenceTests(unittest.TestCase):
    def setUp(self):
        self._model_confidence_threshold = config.model.confidence_threshold
        self._confidence_config = deepcopy(config.confidence)
        self._symbol_name = config.symbol.symbol
        self._timeframe = config.symbol.timeframe
        self._mt5_timeframe = config.symbol.mt5_timeframe

        config.model.confidence_threshold = 0.50
        config.confidence.clear()
        config.symbol.symbol = "XAUUSD"
        config.symbol.timeframe = "M5"
        config.symbol.mt5_timeframe = 5

    def tearDown(self):
        config.model.confidence_threshold = self._model_confidence_threshold
        config.confidence = self._confidence_config
        config.symbol.symbol = self._symbol_name
        config.symbol.timeframe = self._timeframe
        config.symbol.mt5_timeframe = self._mt5_timeframe

    def test_signal_generator_uses_symbol_timeframe_threshold(self):
        config.confidence.load_from_mapping({
            "by_symbol_timeframe": {
                "XAUUSD": {
                    "M5": {"signal_threshold": 0.60},
                },
            },
        })
        model = DummyModel([0.10, 0.56, 0.34])
        generator = SignalGenerator(model)

        signal = generator.generate_signal(pd.DataFrame([{"f1": 1.0}]))

        self.assertEqual(generator.confidence_threshold, 0.60)
        self.assertEqual(signal["action"], "NO_TRADE")
        self.assertAlmostEqual(signal["confidence"], 0.56)


if __name__ == "__main__":
    unittest.main()
