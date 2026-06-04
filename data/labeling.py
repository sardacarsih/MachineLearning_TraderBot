"""
============================================
Label Generation Module
============================================
Creates 3-class labels (BUY/SELL/NO_TRADE) based on forward-looking
price movement relative to ATR-based risk/reward targets.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from collections import Counter

from config.settings import config
from utils.logger import get_logger

logger = get_logger()

# Label constants
NO_TRADE = 0
BUY = 1
SELL = 2
LABEL_NAMES = {NO_TRADE: "NO_TRADE", BUY: "BUY", SELL: "SELL"}


class LabelGenerator:
    """
    Generates forward-looking labels for ML classification.

    For each bar, looks ahead N candles to determine if price moves
    sufficiently in one direction to justify a trade with RR >= 1:1.5.

    Labels:
        0 = NO_TRADE (sideways/noise)
        1 = BUY  (price goes up >= TP before hitting SL)
        2 = SELL (price goes down >= TP before hitting SL)
    """

    def __init__(self, cfg=None):
        """
        Initialize label generator.

        Args:
            cfg: DataConfig instance. Uses global config if None.
        """
        self.cfg = cfg or config.data
        self.lookahead_min = self.cfg.label_lookahead_min
        self.lookahead_max = self.cfg.label_lookahead_max
        self.rr_ratio = self.cfg.reward_risk_ratio
        self.atr_multiplier = self.cfg.atr_sl_multiplier

    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate BUY/SELL/NO_TRADE labels for each bar.

        Logic:
        - SL distance = ATR * atr_multiplier
        - TP distance = SL * reward_risk_ratio
        - Look ahead 5-10 candles:
          - BUY if high reaches entry + TP before low hits entry - SL
          - SELL if low reaches entry - TP before high hits entry + SL
          - NO_TRADE otherwise

        Args:
            df: DataFrame with 'close', 'high', 'low', 'atr' columns.

        Returns:
            DataFrame with 'label' (int) and 'label_name' (str) columns added.
        """
        if 'atr' not in df.columns:
            raise ValueError(
                "DataFrame must have 'atr' column. "
                "Run FeatureEngineer.add_all_features() first."
            )

        logger.info(
            f"Generating labels with lookahead={self.lookahead_min}-"
            f"{self.lookahead_max}, RR=1:{self.rr_ratio}, "
            f"ATR_mult={self.atr_multiplier}"
        )

        n = len(df)
        labels = np.full(n, NO_TRADE, dtype=np.int32)

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        atr = df['atr'].values

        for i in range(n - self.lookahead_max):
            entry_price = close[i]
            current_atr = atr[i]

            # Skip if ATR is too small (no volatility)
            if current_atr < config.symbol.point * 1.0:
                continue

            sl_distance = current_atr * self.atr_multiplier
            tp_distance = sl_distance * self.rr_ratio

            # BUY targets
            buy_tp = entry_price + tp_distance
            buy_sl = entry_price - sl_distance

            # SELL targets
            sell_tp = entry_price - tp_distance
            sell_sl = entry_price + sl_distance

            # Look ahead window
            start = i + 1
            end = min(i + 1 + self.lookahead_max, n)

            buy_hit = False
            sell_hit = False
            buy_stopped = False
            sell_stopped = False

            for j in range(start, end):
                # Check BUY scenario
                if not buy_stopped and not buy_hit:
                    if low[j] <= buy_sl:
                        buy_stopped = True
                    if high[j] >= buy_tp:
                        buy_hit = True

                # Check SELL scenario
                if not sell_stopped and not sell_hit:
                    if high[j] >= sell_sl:
                        sell_stopped = True
                    if low[j] <= sell_tp:
                        sell_hit = True

            # Assign label (BUY takes priority if both hit)
            if buy_hit and not buy_stopped:
                labels[i] = BUY
            elif sell_hit and not sell_stopped:
                labels[i] = SELL
            # else: stays NO_TRADE

        df = df.copy()
        df['label'] = labels
        df['label_name'] = df['label'].map(LABEL_NAMES)

        # Remove last rows that couldn't be labeled (no lookahead)
        valid_end = n - self.lookahead_max
        df = df.iloc[:valid_end].reset_index(drop=True)

        # Log distribution
        dist = self.get_label_distribution(df)
        logger.info(f"Label distribution: {dist}")

        return df

    def get_label_distribution(self, df: pd.DataFrame) -> Dict[str, dict]:
        """
        Get label class distribution.

        Args:
            df: DataFrame with 'label' column.

        Returns:
            Dictionary with count and percentage for each class.
        """
        if 'label' not in df.columns:
            return {}

        counts = Counter(df['label'].values)
        total = len(df)

        distribution = {}
        for label_int, label_name in LABEL_NAMES.items():
            count = counts.get(label_int, 0)
            pct = count / total * 100 if total > 0 else 0
            distribution[label_name] = {
                "count": count,
                "percentage": round(pct, 2),
            }

        return distribution

    def balance_classes(self, df: pd.DataFrame,
                        method: str = "undersample") -> pd.DataFrame:
        """
        Balance class distribution to improve model training.

        Args:
            df: DataFrame with 'label' column.
            method: 'undersample' (reduce majority) or 'oversample' (increase minority).

        Returns:
            Balanced DataFrame.
        """
        if 'label' not in df.columns:
            raise ValueError("DataFrame must have 'label' column")

        logger.info(f"Balancing classes using {method} method")

        if method == "undersample":
            return self._undersample(df)
        elif method == "oversample":
            return self._oversample(df)
        elif method == "smote":
            return self._smote_balance(df)
        else:
            raise ValueError(f"Unknown balancing method: {method}")

    def _undersample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Undersample majority classes to match minority class size."""
        class_counts = df['label'].value_counts()
        min_count = class_counts.min()

        balanced_dfs = []
        for label in class_counts.index:
            class_df = df[df['label'] == label]
            sampled = class_df.sample(n=min_count, random_state=42)
            balanced_dfs.append(sampled)

        result = pd.concat(balanced_dfs).sort_index().reset_index(drop=True)
        logger.info(
            f"Undersampled: {len(df)} -> {len(result)} rows "
            f"({min_count} per class)"
        )
        return result

    def _oversample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Oversample minority classes to match majority class size."""
        class_counts = df['label'].value_counts()
        max_count = class_counts.max()

        balanced_dfs = []
        for label in class_counts.index:
            class_df = df[df['label'] == label]
            if len(class_df) < max_count:
                oversampled = class_df.sample(
                    n=max_count, replace=True, random_state=42
                )
                balanced_dfs.append(oversampled)
            else:
                balanced_dfs.append(class_df)

        result = pd.concat(balanced_dfs).sort_index().reset_index(drop=True)
        logger.info(
            f"Oversampled: {len(df)} -> {len(result)} rows "
            f"({max_count} per class)"
        )
        return result

    def _smote_balance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply SMOTE (Synthetic Minority Over-sampling Technique).
        Requires: pip install imbalanced-learn
        """
        try:
            from imblearn.over_sampling import SMOTE

            feature_cols = [c for c in df.columns
                           if c not in ['time', 'label', 'label_name']]
            X = df[feature_cols].values
            y = df['label'].values

            smote = SMOTE(random_state=42, k_neighbors=5)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            result = pd.DataFrame(X_resampled, columns=feature_cols)
            result['label'] = y_resampled
            result['label_name'] = result['label'].map(LABEL_NAMES)

            logger.info(
                f"SMOTE applied: {len(df)} -> {len(result)} rows"
            )
            return result

        except ImportError:
            logger.warning(
                "imbalanced-learn not installed. Falling back to oversampling. "
                "Install with: pip install imbalanced-learn"
            )
            return self._oversample(df)
