"""
======================================================================
XAUUSD M5 Machine Learning Trading Bot — Unified Entry Point
======================================================================
Main entry script routing to specialized sub-modules:
1. train      — Data fetch, labeling, training, and model selection.
2. backtest   — Run backtesting using historical data and a saved model.
3. compare    — Train/backtest multiple timeframes and select the winner.
4. live       — Live execution loop on MetaTrader 5 broker feed.
5. paper      — Real-time paper trading loop with virtual accounts.
6. analyze-live — Analyze live/paper trade logs and realized performance.
"""

import sys
import os

# Adjust path to import from root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logging
from utils.banner import render_banner
from config.settings import config


def _get_arg_value(argv, option_name):
    """Return a CLI option value without consuming argv."""
    prefix = f"{option_name}="
    for index, arg in enumerate(argv):
        if arg.startswith(prefix):
            return arg.split("=", 1)[1]
        if arg == option_name and index + 1 < len(argv):
            next_arg = argv[index + 1]
            if not next_arg.startswith("--"):
                return next_arg
    return None


def get_banner_context(argv):
    """Read symbol/timeframe early so the banner matches the selected pair."""
    symbol = _get_arg_value(argv, "--symbol") or config.symbol.symbol
    timeframe = _get_arg_value(argv, "--timeframe") or config.symbol.timeframe
    return symbol, timeframe


def show_banner(symbol=None, timeframe=None):
    """Prints the application banner to stdout."""
    print(render_banner(symbol=symbol, timeframe=timeframe))


def should_show_plain_banner(argv):
    """Avoid a one-time banner print when the Rich live dashboard owns the screen."""
    if not argv:
        return True
    command = argv[0].lower()
    if command != "live":
        return True
    return "--no-dashboard" in argv


def main():
    args = sys.argv[1:]
    banner_symbol, banner_timeframe = get_banner_context(args)
    plain_banner = should_show_plain_banner(args)
    if plain_banner:
        show_banner(banner_symbol, banner_timeframe)
    
    # Configure logging
    setup_logging(debug=config.debug, quiet_console=not plain_banner)
    
    if len(sys.argv) < 2:
        print("Usage: python main.py [train | backtest | compare | live | paper | analyze-live] --help")
        print("\nCommands:")
        print("  train      Run model training pipeline")
        print("  backtest   Execute historical backtester")
        print("  compare    Train/backtest M5 and M15, then select best timeframe/model")
        print("  live       Launch live trading loop on MT5")
        print("  paper      Launch real-time paper trading loop")
        print("  analyze-live Analyze live/paper logs and realized performance")
        sys.exit(1)

    command = sys.argv[1].lower()
    
    # Slice off the first command argument to pass remaining parameters to sub-scripts
    sys.argv.pop(1)

    if command == "train":
        from scripts.train import main as train_main
        train_main()
    elif command == "backtest":
        from scripts.backtest_run import main as backtest_main
        backtest_main()
    elif command == "compare":
        from scripts.train_compare_timeframes import main as compare_main
        compare_main()
    elif command == "live":
        from scripts.live_trade import main as live_main
        live_main()
    elif command == "paper":
        from scripts.paper_trade import main as paper_main
        paper_main()
    elif command == "analyze-live":
        from scripts.analyze_live_trade import main as analyze_live_main
        analyze_live_main()
    else:
        print(f"Unknown command: '{command}'")
        print("Available: train, backtest, compare, live, paper, analyze-live")
        sys.exit(1)


if __name__ == "__main__":
    main()
