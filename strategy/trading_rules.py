"""
============================================
Trading Rules Engine
============================================
Implements rule-based entry confirmations and filters to validate ML signals.
Checks trend alignment (EMAs), pullbacks, RSI momentum, breakouts, and market conditions.
"""

from typing import Dict, Any, Tuple

from config.settings import config
from utils.logger import get_logger

logger = get_logger()


class TradingRules:
    """
    Combines rule-based checks with ML signals to ensure trades are only executed
    under favorable technical and market conditions.
    """

    def __init__(self):
        logger.info("TradingRules engine initialized")

    def check_buy_conditions(self, features: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate rule-based BUY setup.

        Checks:
            1. EMA Trend Alignment: EMA20 > EMA50 > EMA200
            2. Pullback Confirmation: Low price pulled back to or below EMA20/50
            3. RSI Bullish Zone: RSI is between 30 and 70 (not overbought)
            4. RSI Momentum: RSI is rising (rsi_momentum > 0) OR Stochastic RSI cross
            5. Breakout Range: Price in upper range (breakout_position >= 0.5)

        Args:
            features: Dictionary containing latest technical feature values.

        Returns:
            Tuple: (is_confirmed: bool, reason: str)
        """
        # 1. EMA alignment
        is_aligned = features.get('ema_alignment_bullish', 0) == 1
        if not is_aligned:
            # Fallback to manual check
            ema20 = features.get('ema_20')
            ema50 = features.get('ema_50')
            ema200 = features.get('ema_200')
            if ema20 and ema50 and ema200:
                is_aligned = ema20 > ema50 > ema200

        if not is_aligned:
            return False, "EMA trend alignment is not bullish (EMA20 > EMA50 > EMA200)"

        # 2. Retrace / Pullback check (price low is near or below EMA20 or EMA50)
        low = features.get('low')
        ema20 = features.get('ema_20')
        ema50 = features.get('ema_50')
        close = features.get('close')

        # Price shouldn't be too far from EMA20
        dist_ema20 = features.get('dist_ema_20', 999.0)
        
        pulled_back = False
        if low and ema20 and ema50:
            # Low is below EMA20 or within 0.05% of it, or low is below EMA50
            pulled_back = (low <= ema20 * 1.002) or (low <= ema50 * 1.002) or (dist_ema20 <= 0.1)

        if not pulled_back:
            return False, f"Price has not retraced to EMA20/50 support (low: {low}, ema20: {ema20})"

        # 3. RSI Zone check
        rsi = features.get('rsi')
        if rsi is not None:
            if rsi > config.features.rsi_overbought:
                return False, f"RSI is overbought: {rsi:.2f}"
            if rsi < config.features.rsi_oversold:
                return False, f"RSI is oversold (waiting for momentum): {rsi:.2f}"

        # 4. RSI Momentum check
        rsi_momentum = features.get('rsi_momentum', 0.0)
        stoch_rsi_k = features.get('stoch_rsi_k', 0.0)
        stoch_rsi_d = features.get('stoch_rsi_d', 0.0)
        
        momentum_bullish = (rsi_momentum > 0) or (stoch_rsi_k > stoch_rsi_d)
        if not momentum_bullish:
            return False, "RSI momentum is not rising"

        # 5. Breakout Range Check
        breakout_pos = features.get('breakout_position', 0.0)
        is_breakout = features.get('breakout_high', 0) == 1
        
        # Price should not be at the absolute bottom of the range
        if breakout_pos < 0.4 and not is_breakout:
            return False, f"Price is in the lower part of range (breakout_position: {breakout_pos:.2f})"

        return True, "BUY conditions fully confirmed"

    def check_sell_conditions(self, features: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate rule-based SELL setup.

        Checks:
            1. EMA Trend Alignment: EMA20 < EMA50 < EMA200
            2. Pullback Confirmation: High price pulled back to or above EMA20/50
            3. RSI Bearish Zone: RSI is between 30 and 70 (not oversold)
            4. RSI Momentum: RSI is falling (rsi_momentum < 0) OR Stochastic RSI cross
            5. Breakout Range: Price in lower range (breakout_position <= 0.5)

        Args:
            features: Dictionary containing latest technical feature values.

        Returns:
            Tuple: (is_confirmed: bool, reason: str)
        """
        # 1. EMA alignment
        is_aligned = features.get('ema_alignment_bearish', 0) == 1
        if not is_aligned:
            ema20 = features.get('ema_20')
            ema50 = features.get('ema_50')
            ema200 = features.get('ema_200')
            if ema20 and ema50 and ema200:
                is_aligned = ema20 < ema50 < ema200

        if not is_aligned:
            return False, "EMA trend alignment is not bearish (EMA20 < EMA50 < EMA200)"

        # 2. Retrace / Pullback check
        high = features.get('high')
        ema20 = features.get('ema_20')
        ema50 = features.get('ema_50')
        dist_ema20 = features.get('dist_ema_20', -999.0)

        pulled_back = False
        if high and ema20 and ema50:
            pulled_back = (high >= ema20 * 0.998) or (high >= ema50 * 0.998) or (dist_ema20 >= -0.1)

        if not pulled_back:
            return False, f"Price has not retraced to EMA20/50 resistance (high: {high}, ema20: {ema20})"

        # 3. RSI Zone check
        rsi = features.get('rsi')
        if rsi is not None:
            if rsi < config.features.rsi_oversold:
                return False, f"RSI is oversold: {rsi:.2f}"
            if rsi > config.features.rsi_overbought:
                return False, f"RSI is overbought (waiting for momentum): {rsi:.2f}"

        # 4. RSI Momentum check
        rsi_momentum = features.get('rsi_momentum', 0.0)
        stoch_rsi_k = features.get('stoch_rsi_k', 0.0)
        stoch_rsi_d = features.get('stoch_rsi_d', 0.0)
        
        momentum_bearish = (rsi_momentum < 0) or (stoch_rsi_k < stoch_rsi_d)
        if not momentum_bearish:
            return False, "RSI momentum is not falling"

        # 5. Breakout Range Check
        breakout_pos = features.get('breakout_position', 0.0)
        is_breakout = features.get('breakout_low', 0) == 1

        if breakout_pos > 0.6 and not is_breakout:
            return False, f"Price is in the upper part of range (breakout_position: {breakout_pos:.2f})"

        return True, "SELL conditions fully confirmed"

    def check_no_trade_conditions(self, features: Dict[str, Any], spread: float) -> Tuple[bool, str]:
        """
        Check conditions that suggest we should sit on our hands.

        Checks:
            1. High Spread: Current spread exceeds max allowed
            2. Low Volatility: ATR is too small
            3. Ranging/Sideways Market: ADX is extremely low

        Args:
            features: Dictionary containing latest technical feature values.
            spread: Current spread in points.

        Returns:
            Tuple: (should_avoid: bool, reason: str)
        """
        # 1. Spread filter
        if config.filters.spread_filter_enabled:
            if spread > config.risk.max_spread:
                return True, f"High Spread: {spread} points (Max: {config.risk.max_spread})"

        # 2. Volatility filter
        if config.filters.volatility_filter_enabled:
            atr = features.get('atr', 0.0)
            if atr < config.filters.min_atr_threshold:
                return True, f"Low Volatility: ATR is {atr:.4f} (Min: {config.filters.min_atr_threshold})"
            if atr > config.filters.max_atr_threshold:
                return True, f"Extreme Volatility: ATR is {atr:.4f} (Max: {config.filters.max_atr_threshold})"

        # 3. Ranging filter
        if config.filters.ranging_filter_enabled:
            adx = features.get('adx', 0.0)
            if adx < config.filters.adx_ranging_threshold:
                return True, f"Ranging Market: ADX is {adx:.2f} (Min Trend Strength: {config.filters.adx_ranging_threshold})"

        return False, "Market conditions are tradeable"

    def validate_signal(self, ml_signal: str, features: Dict[str, Any], spread: float) -> Tuple[bool, str]:
        """
        Combines the ML signal with rule confirmations and market condition filters.

        Args:
            ml_signal: Output of ML model ('BUY', 'SELL', 'NO_TRADE').
            features: Dictionary containing latest technical feature values.
            spread: Current spread in points.

        Returns:
            Tuple: (is_valid: bool, reason: str)
        """
        if ml_signal == "NO_TRADE":
            return False, "ML model predicted NO_TRADE"

        # Check market filters first (spread, volatility, ranging)
        should_avoid, avoid_reason = self.check_no_trade_conditions(features, spread)
        if should_avoid:
            return False, avoid_reason

        # Check signal specific rules
        if ml_signal == "BUY":
            is_ok, reason = self.check_buy_conditions(features)
            return is_ok, reason
        elif ml_signal == "SELL":
            is_ok, reason = self.check_sell_conditions(features)
            return is_ok, reason

        return False, f"Unknown ML signal: {ml_signal}"
