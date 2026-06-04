"""
Reusable terminal banner for the trading bot.
"""

from config.settings import config


BANNER_BORDER = "=" * 85

BANNER_TEMPLATE = r"""
{border}
  ____            _                   _   _                           _ _   __  __ _
 |  _ \  __ _  __| | __ _ _ __   __ _| | | | __ _ _ __ _   _  __ _ __| (_) |  \/  | |
 | | | |/ _` |/ _` |/ _` | '_ \ / _` | |_| |/ _` | '__| | | |/ _` / _` | | | |\/| | |
 | |_| | (_| | (_| | (_| | | | | (_| |  _  | (_| | |  | |_| | (_| (_ | | | | |  | | |___
 |____/ \__,_|\__,_|\__,_|_| |_|\__, |_| |_|\__,_|_|   \__, |\__,_\__,_|_|_|_|  |_|_____|
                                |___/                  |___/
                         [EML Trading Bot | {symbol} {timeframe} | AI Live Engine]
{border}
"""


def render_banner(symbol: str = None, timeframe: str = None) -> str:
    """Render the fixed ASCII application banner."""
    symbol = symbol or config.symbol.symbol
    timeframe = timeframe or config.symbol.timeframe
    return BANNER_TEMPLATE.format(
        border=BANNER_BORDER,
        symbol=str(symbol).upper(),
        timeframe=str(timeframe).upper(),
    )
