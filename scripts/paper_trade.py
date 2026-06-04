"""
============================================
Paper Trading Execution Wrapper
============================================
Launches the live trading orchestrator forced in paper trading mode.
Allows verification of model signals and risk logic against real-time broker feeds
using simulated orders and virtual account tracking.
"""

import os
import sys
import argparse

# Adjust path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.mt5_config import apply_mt5_config_from_yaml, mt5_config
from utils.logger import setup_logging
from scripts.live_trade import run_trading_bot
from config.settings import normalize_timeframe

def main():
    parser = argparse.ArgumentParser(description="Run Paper Trading Simulation for XAUUSD M5")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model file/directory")
    parser.add_argument("--symbol", type=str, default=None, help="Trading symbol (e.g. XAUUSD, USTEC_x100)")
    parser.add_argument("--timeframe", type=str, default=None, help="Trading timeframe (e.g. M5, M15)")
    parser.add_argument("--config", type=str, default=None, help="Path to custom MT5 credentials yaml")
    parser.add_argument("--strategy-mode", choices=["ml", "hybrid"], default="hybrid",
                        help="Entry validation mode. 'ml' uses model signals after safety filters; "
                             "'hybrid' also requires technical rule confirmation.")
    
    args = parser.parse_args()

    # Load custom yaml configurations if provided
    if args.config:
        apply_mt5_config_from_yaml(args.config)

    # Dynamic symbol specification retrieval
    from config.settings import config
    if args.timeframe:
        config.set_timeframe(normalize_timeframe(args.timeframe))

    config.set_symbol(args.symbol or config.symbol.symbol)

    if args.symbol:
        try:
            import MetaTrader5 as mt5
            if mt5.initialize():
                if mt5_config.login > 0:
                    mt5.login(login=mt5_config.login, password=mt5_config.password, server=mt5_config.server)
                mt5.symbol_select(args.symbol, True)
                info = mt5.symbol_info(args.symbol)
                if info is not None:
                    config.symbol.point = info.point
                    config.symbol.digits = info.digits
                    config.symbol.contract_size = info.trade_contract_size
                mt5.shutdown()
        except Exception as e:
            pass

    # Force paper trading mode
    run_trading_bot(args.model, paper_trading=True, strategy_mode=args.strategy_mode, dashboard_enabled=False)


if __name__ == "__main__":
    main()
