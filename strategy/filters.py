"""
============================================
Trade Filters
============================================
Checks market filters: spreads, volatility, session hours, ranging conditions,
and news schedules to avoid trading under unfavorable external conditions.
"""

from datetime import datetime, time
from typing import List, Tuple, Dict, Any, Optional

from config.settings import config
from utils.logger import get_logger

logger = get_logger()


class TradeFilters:
    """
    Evaluates market conditions (volatility, spreads, trading session, news schedules)
    to confirm if the bot should be allowed to trade at the current moment.
    """

    def __init__(self):
        logger.info("TradeFilters initialized")

    def check_spread(self, current_spread: float, max_spread: float) -> bool:
        """Returns True if the spread is acceptable, False if it is too high."""
        is_ok = current_spread <= max_spread
        if not is_ok:
            logger.debug(f"Spread filter failed: current={current_spread}, max={max_spread}")
        return is_ok

    def check_volatility(self, atr: float, min_atr: float, max_atr: float) -> bool:
        """Returns True if ATR is between min and max volatility levels."""
        is_ok = min_atr <= atr <= max_atr
        if not is_ok:
            logger.debug(f"Volatility filter failed: atr={atr:.4f}, min={min_atr}, max={max_atr}")
        return is_ok

    def check_session(self, current_time: datetime) -> bool:
        """
        Check if the current UTC time falls within permitted trading sessions.
        London: 07:00 - 16:00 UTC
        New York: 12:00 - 21:00 UTC
        Asia: 00:00 - 08:00 UTC
        """
        if not config.filters.session_filter_enabled:
            return True

        current_hour = current_time.hour
        current_minute = current_time.minute
        time_float = current_hour + current_minute / 60.0

        is_london = 7.0 <= time_float < 16.0
        is_newyork = 12.0 <= time_float < 21.0
        is_asia = 0.0 <= time_float < 8.0

        allowed = config.filters.allowed_sessions

        # Check if the active sessions match allowed settings
        session_active = False
        if "london" in allowed and is_london:
            session_active = True
        if "newyork" in allowed and is_newyork:
            session_active = True
        if "asia" in allowed and is_asia:
            session_active = True

        if not session_active:
            logger.debug(f"Session filter failed: time={current_time.strftime('%H:%M')} (not in allowed sessions {allowed})")
        
        return session_active

    def check_ranging_market(self, adx: float) -> bool:
        """Returns True if the market is trending (ADX >= threshold), False if ranging."""
        threshold = config.filters.adx_ranging_threshold
        is_trending = adx >= threshold
        if not is_trending:
            logger.debug(f"Ranging market filter failed: ADX={adx:.2f} (threshold: {threshold})")
        return is_trending

    def check_news_time(self, current_time: datetime, news_times: List[datetime]) -> bool:
        """
        Check if current time falls within the window surrounding high-impact news.

        Args:
            current_time: Current time in UTC.
            news_times: List of high-impact news event datetimes in UTC.

        Returns:
            True if it's safe to trade (not near news), False if too close.
        """
        if not config.filters.news_filter_enabled or not news_times:
            return True

        avoid_before = config.filters.news_avoid_minutes_before
        avoid_after = config.filters.news_avoid_minutes_after

        for news_t in news_times:
            time_diff = (current_time - news_t).total_seconds() / 60.0
            # If current_time is in [news_t - avoid_before, news_t + avoid_after]
            if -avoid_before <= time_diff <= avoid_after:
                logger.warning(
                    f"News filter active! Too close to news event at {news_t.strftime('%Y-%m-%d %H:%M')}. "
                    f"Diff: {time_diff:.1f} minutes."
                )
                return False

        return True

    def apply_all_filters(
        self, current_data: Dict[str, Any], current_time: datetime,
        current_spread: float, news_times: Optional[List[datetime]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Runs all filters.

        Args:
            current_data: Dictionary of latest technical indicator values (atr, adx, etc.).
            current_time: Current time in UTC.
            current_spread: Current symbol spread in points.
            news_times: Optional list of news event datetimes.

        Returns:
            Tuple: (can_trade: bool, fail_reasons: list of strings)
        """
        reasons = []
        news_times = news_times or []

        # 1. Spread Check
        if config.filters.spread_filter_enabled:
            max_spread = config.risk.max_spread
            if not self.check_spread(current_spread, max_spread):
                reasons.append(f"Spread too high ({current_spread} > {max_spread} pts)")

        # 2. Volatility Check
        if config.filters.volatility_filter_enabled:
            atr = current_data.get("atr", 0.0)
            min_atr = config.filters.min_atr_threshold
            max_atr = config.filters.max_atr_threshold
            if not self.check_volatility(atr, min_atr, max_atr):
                reasons.append(f"Volatility out of bounds (ATR={atr:.4f}, bounds=[{min_atr}, {max_atr}])")

        # 3. Session Check
        if config.filters.session_filter_enabled:
            if not self.check_session(current_time):
                reasons.append(f"Outside allowed trading session hours")

        # 4. Ranging Check
        if config.filters.ranging_filter_enabled:
            adx = current_data.get("adx", 0.0)
            if not self.check_ranging_market(adx):
                reasons.append(f"Ranging market (ADX={adx:.2f} < {config.filters.adx_ranging_threshold})")

        # 5. News Check
        if config.filters.news_filter_enabled and news_times:
            if not self.check_news_time(current_time, news_times):
                reasons.append("Too close to high-impact news event")

        can_trade = len(reasons) == 0
        return can_trade, reasons
