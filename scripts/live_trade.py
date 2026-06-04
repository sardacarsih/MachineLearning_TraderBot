"""
============================================
Live Trading Orchestrator Loop
============================================
Connects to MetaTrader 5, loads a trained model, and runs a real-time trading loop.
Calculates features and generates entry signals on candle closes.
Manages trailing stops and open trades on every tick loop cycle.
Handles risk manager limits and auto-reconnects on terminal drop.
"""

import os
import sys
import time
import argparse
import signal
import math
from datetime import datetime

# Adjust path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import config, normalize_timeframe, timeframe_to_minutes
from config.mt5_config import mt5_config, apply_mt5_config_from_yaml
from utils.logger import setup_logging, get_logger, TradingLogger
from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from strategy.signal_generator import SignalGenerator
from strategy.risk_manager import RiskManager
from strategy.trading_rules import TradingRules
from strategy.filters import TradeFilters
from mt5.connector import MT5Connector
from mt5.order_executor import OrderExecutor
from mt5.account_manager import AccountManager
from mt5.currency import CurrencyConverter
from scripts.backtest_run import load_saved_model
from utils.live_dashboard import RichLiveDashboard

logger = get_logger()

# Global flag for graceful shutdown
running = True
MAX_ORDER_COMMENT_LEN = 31
ACCOUNT_SUMMARY_WAIT_SECONDS = 60


def model_expects_higher_timeframe(model) -> bool:
    """Return whether the loaded model schema requires HTF features."""
    schema = getattr(model, "_feature_schema", None)
    feature_names = getattr(schema, "feature_names", None) or getattr(model, "_feature_names", [])
    return any(str(name).startswith("htf_") for name in feature_names)


def build_live_features(raw_df, include_higher_timeframe: bool = False):
    """Build features from a raw dataframe with a fresh feature engineer."""
    return FeatureEngineer(include_higher_timeframe=include_higher_timeframe).add_all_features(raw_df.copy())


def feature_window_size(include_higher_timeframe: bool = False) -> int:
    """Use enough raw history for stable slow indicators in live inference."""
    base_window = max(300, int(config.features.ema_slow) * 3)
    if not include_higher_timeframe:
        return base_window

    base_minutes = timeframe_to_minutes(config.symbol.timeframe)
    h1_ema_window = int(math.ceil(config.features.ema_slow * 60 / base_minutes))
    h4_regime_window = int(math.ceil(50 * 240 / base_minutes))
    return max(base_window, h1_ema_window, h4_regime_window, 2400 if config.symbol.timeframe == "M5" else 0)


def passes_min_candle_size(latest_data: dict) -> tuple[bool, str]:
    """Reject near-flat closed candles that are usually spread/noise dominated."""
    high = float(latest_data.get("high", 0.0) or 0.0)
    low = float(latest_data.get("low", 0.0) or 0.0)
    atr = float(latest_data.get("atr", 0.0) or 0.0)
    candle_range = high - low
    min_by_point = config.symbol.point * 5
    min_by_atr = atr * 0.05 if atr > 0 else 0.0
    min_range = max(min_by_point, min_by_atr)
    if candle_range < min_range:
        return False, f"Candle range too small ({candle_range:.5f} < {min_range:.5f})"
    return True, "OK"


def format_order_comment(timeframe: str, strategy_mode: str, model_name: str) -> str:
    """Build a broker-safe order comment while preserving timeframe first."""
    raw_comment = f"{timeframe}_{strategy_mode.upper()}_{model_name}"
    return raw_comment[:MAX_ORDER_COMMENT_LEN]


def format_block_reason(reason) -> str:
    """Format dashboard block reasons into one compact line."""
    if isinstance(reason, (list, tuple)):
        text = "; ".join(str(item) for item in reason if str(item).strip())
    else:
        text = str(reason or "").strip()
    return text or "No reason provided"


def update_dashboard_signal_status(
    dashboard,
    action: str,
    candle_time,
    reason: str,
    confidence=None,
    probabilities=None,
) -> None:
    """Show the current candle's signal lifecycle even when inference is blocked."""
    if not dashboard:
        return
    dashboard.update_signal({
        "action": action,
        "signal_time": str(candle_time) if candle_time is not None else None,
        "timeframe": config.symbol.timeframe,
        "confidence": confidence,
        "probabilities": probabilities or [],
        "reason": format_block_reason(reason),
    })


def account_login_matches(account_info) -> bool:
    """Validate that MT5 is connected to the configured account."""
    expected_login = int(mt5_config.login or 0)
    if expected_login <= 0:
        return True

    actual_login = int(getattr(account_info, "login", 0) or 0)
    return actual_login == expected_login


def get_reliable_account_summary(account_manager: AccountManager, connector: MT5Connector):
    """
    Return a broker-backed account summary only after MT5 is connected to the
    configured login. Live mode must not fall back to virtual balances.
    """
    if not connector.is_connected():
        return None, "MT5 is not connected"

    account_info = connector.get_account_info()
    if account_info is None:
        return None, "MT5 account_info is unavailable"

    if not account_login_matches(account_info):
        expected_login = int(mt5_config.login or 0)
        actual_login = int(getattr(account_info, "login", 0) or 0)
        return (
            None,
            f"MT5 account mismatch: expected login {expected_login}, connected login {actual_login}",
        )

    summary = account_manager.get_account_summary()
    summary["login"] = int(getattr(account_info, "login", 0) or 0)
    summary["currency"] = str(getattr(account_info, "currency", "") or "")
    return summary, "OK"


def wait_for_reliable_account_summary(account_manager: AccountManager, connector: MT5Connector):
    """Wait briefly for a live broker summary before initializing risk limits."""
    deadline = time.time() + ACCOUNT_SUMMARY_WAIT_SECONDS
    last_reason = "Account summary not checked yet"

    while running and time.time() < deadline:
        summary, reason = get_reliable_account_summary(account_manager, connector)
        if summary is not None:
            logger.info(
                f"Live account summary accepted. Login: {summary.get('login')}, "
                f"Balance: {summary['balance']:.2f}, Equity: {summary['equity']:.2f}"
            )
            return summary

        last_reason = reason
        logger.warning(f"Waiting for reliable MT5 account summary: {reason}")
        time.sleep(2)

    raise RuntimeError(f"Could not get reliable MT5 account summary: {last_reason}")


def signal_handler(signum, frame):
    """Handles Ctrl+C and exit signals for graceful shutdown."""
    global running
    logger.info("Graceful shutdown signal received. Cleaning up...")
    running = False


def run_trading_bot(
    model_path: str,
    paper_trading: bool,
    strategy_mode: str = "hybrid",
    dashboard_enabled: bool = True,
):
    """
    Executes the main live trading loop.
    """
    global running
    strategy_mode = (strategy_mode or "hybrid").strip().lower()
    if strategy_mode not in {"ml", "hybrid"}:
        raise ValueError("strategy_mode must be 'ml' or 'hybrid'")
    config.strategy_mode = strategy_mode
    
    # Register shutdown handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Force paper trading override if requested
    if paper_trading:
        mt5_config.paper_trading = True
        logger.info("Overriding config to PAPER TRADING mode.")
    else:
        mt5_config.paper_trading = False
        mt5_config.trading_enabled = True
        logger.warning("Bot is running in LIVE TRADING mode! Real orders will be sent to the broker.")

    # Initialize components
    connector = MT5Connector()
    
    # Try connecting
    if not connector.connect():
        logger.error("Could not establish connection to MT5. Exiting.")
        sys.exit(1)

    account_manager = AccountManager(connector)
    order_executor = OrderExecutor(connector, account_manager=account_manager)
    currency_converter = CurrencyConverter(connector)
    risk_manager = RiskManager()
    trading_rules = TradingRules()
    trade_filters = TradeFilters()
    loader = DataLoader()

    # Load model and set up signal generator
    try:
        model = load_saved_model(model_path)
        signal_generator = SignalGenerator(model)
        include_higher_timeframe = model_expects_higher_timeframe(model)
        logger.info(f"Higher timeframe live features enabled: {include_higher_timeframe}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        connector.disconnect()
        sys.exit(1)

    # Initialize daily risk limits from broker-backed account data only.
    try:
        summary = wait_for_reliable_account_summary(account_manager, connector)
    except RuntimeError as exc:
        logger.error(str(exc))
        connector.disconnect()
        sys.exit(1)

    risk_manager.reset_daily(summary["balance"])

    dashboard = None
    if dashboard_enabled:
        dashboard = RichLiveDashboard(
            symbol=config.symbol.symbol,
            timeframe=config.symbol.timeframe,
            mode="PAPER" if paper_trading else "LIVE",
            strategy_mode=strategy_mode,
            htf_enabled=include_higher_timeframe,
        )
        dashboard.__enter__()
        dashboard.update_connection(connector.is_connected(), heartbeat_ok=True)
        dashboard.update_account(summary)

    logger.info("======================================================================")
    logger.info(f"Bot started for {config.symbol.symbol} {config.symbol.timeframe}. Running active loop...")
    logger.info(f"Strategy Mode: {strategy_mode.upper()}")
    logger.info("======================================================================")

    last_bar_time = None
    last_signal_key = None
    last_trade_time = None
    loop_interval = 1.0  # Run real-time monitoring loop every 1 second
    raw_window = feature_window_size(include_higher_timeframe)
    
    # State caching & throttlers
    cached_atr = None
    last_candle_check_time = 0.0
    candle_check_interval = 10.0  # Check for candle close every 10 seconds

    while running:
        try:
            # 1. Connection check (heartbeat)
            heartbeat_ok = connector.heartbeat()
            if dashboard:
                dashboard.update_connection(connector.is_connected(), heartbeat_ok=heartbeat_ok)
            if not heartbeat_ok:
                logger.error("MT5 heartbeat failed and reconnection attempts exhausted. Sleeping...")
                time.sleep(30)
                continue

            # Fetch broker-backed account summary. In live mode this must not
            # fall back to the virtual balance, and it must match the YAML login.
            acct_summary, acct_reason = get_reliable_account_summary(account_manager, connector)
            if acct_summary is None:
                logger.error(f"Live account summary blocked: {acct_reason}")
                if dashboard:
                    update_dashboard_signal_status(
                        dashboard,
                        "BLOCKED",
                        last_bar_time,
                        acct_reason,
                    )
                time.sleep(5)
                continue

            balance = acct_summary["balance"]
            equity = acct_summary["equity"]
            if dashboard:
                dashboard.update_account(acct_summary)

            # Reset daily risk limits on calendar day changes (adjust for 1s loop)
            now_utc = datetime.utcnow()
            if now_utc.hour == 0 and now_utc.minute == 0 and now_utc.second < 2:
                risk_manager.reset_daily(balance)

            # ----------------------------------------------------
            # Real-Time Position Management (Run every 1-second loop cycle)
            # ----------------------------------------------------
            open_positions = order_executor.get_open_positions(config.symbol.symbol)
            if dashboard:
                dashboard.update_positions(open_positions)
            
            # Fetch latest tick prices
            tick = connector._mt5.symbol_info_tick(config.symbol.symbol) if connector.is_connected() else None
            if dashboard:
                spread_pts = None
                if tick:
                    try:
                        symbol_info = connector.get_symbol_info(config.symbol.symbol)
                        point = float(getattr(symbol_info, "point", 0.0) or 0.0)
                        if point > 0:
                            spread_pts = float(tick.ask - tick.bid) / point
                    except Exception:
                        spread_pts = None
                dashboard.update_market(tick=tick, spread_points=spread_pts, cached_atr=cached_atr)
            
            if tick and open_positions:
                # Fetch ATR if not cached yet
                if cached_atr is None:
                    try:
                        rates_df = loader.get_latest_bars(n=raw_window)
                        rates_df = build_live_features(rates_df.iloc[:-1].copy(), include_higher_timeframe=False)
                        cached_atr = rates_df["atr"].iloc[-1] if "atr" in rates_df.columns else 1.5
                    except Exception as e:
                        logger.warning(f"Could not compute initial ATR for trailing stop, using default: {e}")
                        cached_atr = 1.5

                for pos in open_positions:
                    ticket = pos["ticket"] if mt5_config.paper_trading else getattr(pos, "ticket")
                    pos_type = pos["type"] if mt5_config.paper_trading else ("BUY" if getattr(pos, "type") == connector._mt5.POSITION_TYPE_BUY else "SELL")
                    entry_p = pos["open_price"] if mt5_config.paper_trading else getattr(pos, "price_open")
                    sl = pos["sl"] if mt5_config.paper_trading else getattr(pos, "sl")
                    tp = pos["tp"] if mt5_config.paper_trading else getattr(pos, "tp")
                    vol = pos["volume"] if mt5_config.paper_trading else getattr(pos, "volume")

                    # Check trailing stop using cached ATR (dynamic re-evaluation)
                    px = tick.bid if pos_type == "BUY" else tick.ask
                    new_sl = risk_manager.calculate_trailing_stop(entry_p, px, sl, cached_atr, pos_type, tp)
                    
                    if new_sl is not None:
                        logger.info(f"Trailing Stop triggered for position {ticket}. Modifying SL: {sl} -> {new_sl}")
                        if order_executor.modify_position(ticket, new_sl, tp):
                            # Update local list representation immediately
                            if mt5_config.paper_trading:
                                pos["sl"] = new_sl

                    # For paper trading, check if SL or TP is hit manually in loop
                    if mt5_config.paper_trading:
                        # Check BUY exits
                        if pos_type == "BUY":
                            if tick.bid <= sl:
                                logger.info(f"[PAPER] SL hit for ticket {ticket}")
                                order_executor.close_position(ticket, exit_reason="SL")
                                risk_manager.update_trade_result(vol * 100 * (sl - entry_p))
                            elif tick.bid >= tp:
                                logger.info(f"[PAPER] TP hit for ticket {ticket}")
                                order_executor.close_position(ticket, exit_reason="TP")
                                risk_manager.update_trade_result(vol * 100 * (tp - entry_p))
                        # Check SELL exits
                        else:
                            if tick.ask >= sl:
                                logger.info(f"[PAPER] SL hit for ticket {ticket}")
                                order_executor.close_position(ticket, exit_reason="SL")
                                risk_manager.update_trade_result(vol * 100 * (entry_p - sl))
                            elif tick.ask <= tp:
                                logger.info(f"[PAPER] TP hit for ticket {ticket}")
                                order_executor.close_position(ticket, exit_reason="TP")
                                risk_manager.update_trade_result(vol * 100 * (entry_p - tp))

            # ----------------------------------------------------
            # Candle Close & Signal Evaluation (Throttled Check)
            # ----------------------------------------------------
            current_time = time.time()
            if current_time - last_candle_check_time >= candle_check_interval:
                last_candle_check_time = current_time

                # Fetch latest bars to construct features
                hist_df = loader.get_latest_bars(n=raw_window)
                logger.debug(
                    f"Fetched raw live bars: rows={len(hist_df)}, cols={list(hist_df.columns)}"
                )
                
                # The last bar in hist_df is the active in-progress bar
                # The last closed bar is index -2
                closed_bar = hist_df.iloc[-2]
                closed_bar_time = closed_bar['time']

                # Check if a new bar has closed
                if last_bar_time is None:
                    last_bar_time = closed_bar_time
                    logger.info(f"Initialized candle track. Current closed bar: {last_bar_time}")
                    if dashboard:
                        dashboard.update_candle(closed_bar_time)
                    update_dashboard_signal_status(
                        dashboard,
                        "WAITING",
                        closed_bar_time,
                        "Initialized candle tracking; waiting for the next closed candle",
                    )
                
                elif closed_bar_time != last_bar_time:
                    logger.info(f"New {config.symbol.timeframe} candle closed! Time: {closed_bar_time}")
                    last_bar_time = closed_bar_time
                    if dashboard:
                        dashboard.update_candle(closed_bar_time)

                    # 1. Run Risk checks before checking entry signals
                    if not risk_manager.can_trade(balance):
                        reason = getattr(risk_manager, "last_block_reason", "") or "Trading halted by RiskManager limits"
                        logger.warning(reason)
                        update_dashboard_signal_status(
                            dashboard,
                            "BLOCKED",
                            closed_bar_time,
                            reason,
                        )
                        continue

                    # 2. Extract features on entire dataset
                    closed_raw_df = hist_df.iloc[:-1].copy()
                    feat_df = build_live_features(
                        closed_raw_df,
                        include_higher_timeframe=include_higher_timeframe,
                    )
                    logger.info(
                        f"Live feature frame built: shape={feat_df.shape}, "
                        f"latest_feature_time={feat_df['time'].iloc[-1] if 'time' in feat_df.columns and len(feat_df) else 'N/A'}"
                    )
                    
                    # Update cached ATR for trailing stop
                    cached_atr = feat_df["atr"].iloc[-1] if "atr" in feat_df.columns else 1.5
                    if dashboard:
                        dashboard.update_market(tick=tick, spread_points=None, cached_atr=cached_atr)
                    
                    # 3. Apply general filters (news, session, etc.)
                    latest_data = feat_df.iloc[-1].to_dict()  # latest completed bar features
                    if dashboard:
                        dashboard.update_trends(latest_data)
                    spread_pts = order_executor.get_current_spread(config.symbol.symbol)
                    
                    can_trade, reasons = trade_filters.apply_all_filters(
                        latest_data, datetime.utcnow(), spread_pts
                    )

                    if not can_trade:
                        logger.info(f"Filters blocked trading. Reasons: {reasons}")
                        update_dashboard_signal_status(
                            dashboard,
                            "BLOCKED",
                            closed_bar_time,
                            reasons,
                        )
                        continue

                    candle_ok, candle_reason = passes_min_candle_size(latest_data)
                    if not candle_ok:
                        logger.info(f"Minimum candle size filter blocked trading. Reason: {candle_reason}")
                        update_dashboard_signal_status(
                            dashboard,
                            "BLOCKED",
                            closed_bar_time,
                            candle_reason,
                        )
                        continue

                    # 4. Generate Signal
                    signal_out = signal_generator.generate_signal(feat_df)
                    signal_out["signal_time"] = str(closed_bar_time)
                    signal_out["timeframe"] = config.symbol.timeframe
                    if dashboard:
                        dashboard.update_signal(signal_out)
                    action = signal_out["action"]
                    confidence = signal_out["confidence"]

                    if action in ["BUY", "SELL"]:
                        signal_key = (config.symbol.symbol, closed_bar_time, action)
                        if signal_key == last_signal_key:
                            logger.info(f"Duplicate signal blocked for {signal_key}")
                            update_dashboard_signal_status(
                                dashboard,
                                "BLOCKED",
                                closed_bar_time,
                                f"Duplicate signal blocked for {action}",
                                confidence=confidence,
                                probabilities=signal_out.get("probabilities"),
                            )
                            continue

                        if tick:
                            entry_price = tick.ask if action == "BUY" else tick.bid
                            can_open, open_reason = order_executor.can_open_averaging_entry(
                                config.symbol.symbol,
                                action,
                                entry_price,
                            )
                            if not can_open:
                                logger.info(open_reason)
                                update_dashboard_signal_status(
                                    dashboard,
                                    "BLOCKED",
                                    closed_bar_time,
                                    open_reason,
                                    confidence=confidence,
                                    probabilities=signal_out.get("probabilities"),
                                )
                                continue

                        if last_trade_time is not None:
                            minutes_since_trade = (datetime.utcnow() - last_trade_time).total_seconds() / 60.0
                            if minutes_since_trade < config.risk.min_trade_interval:
                                cooldown_reason = (
                                    f"Cooldown blocked trade: {minutes_since_trade:.2f} minutes since last trade "
                                    f"< {config.risk.min_trade_interval} minutes"
                                )
                                logger.info(cooldown_reason)
                                update_dashboard_signal_status(
                                    dashboard,
                                    "BLOCKED",
                                    closed_bar_time,
                                    cooldown_reason,
                                    confidence=confidence,
                                    probabilities=signal_out.get("probabilities"),
                                )
                                continue

                        # 5. Validate Signal according to strategy mode
                        if strategy_mode == "hybrid":
                            is_valid, reason = trading_rules.validate_signal(action, latest_data, spread_pts)
                        else:
                            is_valid, reason = True, "ML signal accepted after safety filters"
                        
                        if is_valid:
                            logger.info(
                                f"Signal VALIDATED ({strategy_mode.upper()}). "
                                f"Action: {action} | Conf: {confidence:.4f}"
                            )
                            
                            # Calculate SL & lot sizing
                            sl_price = risk_manager.calculate_sl(closed_bar['close'], latest_data.get('atr', 1.5), action)
                            sl_dist = abs(closed_bar['close'] - sl_price)
                            tp_price = risk_manager.calculate_tp(closed_bar['close'], sl_dist, action)
                            account_risk_amount = balance * config.risk.max_risk_per_trade
                            risk_amount_quote, account_ccy, quote_ccy, conversion_rate = (
                                currency_converter.convert_risk_amount(account_risk_amount)
                            )
                            lot_size = risk_manager.calculate_position_size(
                                balance,
                                sl_dist,
                                risk_amount_quote=risk_amount_quote,
                                account_currency=account_ccy,
                                risk_quote_currency=quote_ccy,
                                conversion_rate=conversion_rate,
                            )
                            lot_size = risk_manager.apply_confidence_lot_multiplier(lot_size, confidence)

                            # Place Order
                            success, ticket_or_err = order_executor.open_position(
                                symbol=config.symbol.symbol,
                                order_type=action,
                                lot=lot_size,
                                sl=sl_price,
                                tp=tp_price,
                                comment=format_order_comment(config.symbol.timeframe, strategy_mode, model.model_name)
                            )
                            
                            if success:
                                last_signal_key = signal_key
                                last_trade_time = datetime.utcnow()
                                TradingLogger.trade_log(
                                    f"[SIGNAL TRADE LINK] Ticket: {ticket_or_err} | "
                                    f"Action: {action} | Confidence: {confidence:.4f} | "
                                    f"SignalTime: {closed_bar_time} | Entry: {closed_bar['close']:.5f} | "
                                    f"SL: {sl_price:.5f} | TP: {tp_price:.5f}"
                                )
                                logger.info(f"Trade successfully placed! Ticket: {ticket_or_err}")
                            else:
                                logger.error(f"Failed to place trade: {ticket_or_err}")
                                update_dashboard_signal_status(
                                    dashboard,
                                    "BLOCKED",
                                    closed_bar_time,
                                    f"Order placement failed: {ticket_or_err}",
                                    confidence=confidence,
                                    probabilities=signal_out.get("probabilities"),
                                )
                        else:
                            logger.info(f"Signal REJECTED by {strategy_mode} validation. Reason: {reason}")
                            update_dashboard_signal_status(
                                dashboard,
                                "BLOCKED",
                                closed_bar_time,
                                f"{strategy_mode.upper()} validation rejected signal: {reason}",
                                confidence=confidence,
                                probabilities=signal_out.get("probabilities"),
                            )

        except Exception as e:
            logger.error(f"Error in main loop cycle: {e}", exc_info=True)

        # Sleep before next cycle
        time.sleep(loop_interval)

    # Disconnect on loop exit
    connector.disconnect()
    logger.info("Bot execution halted. Goodbye.")
    if dashboard:
        dashboard.__exit__(None, None, None)


def main():
    parser = argparse.ArgumentParser(description="Run Live Trading Bot for configured symbol/timeframe")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model directory/file")
    parser.add_argument("--symbol", type=str, default=None, help="Trading symbol (e.g. XAUUSD, USTEC_x100)")
    parser.add_argument("--timeframe", type=str, default=config.symbol.timeframe,
                        help="Trading timeframe (e.g. M5, M15)")
    parser.add_argument("--config", type=str, default=None, help="Path to custom MT5 credentials yaml")
    parser.add_argument("--live", action="store_true", help="Run in live trading mode (default is paper trading)")
    parser.add_argument("--dashboard", dest="dashboard", action="store_true", default=True,
                        help="Enable Rich live terminal dashboard (default)")
    parser.add_argument("--no-dashboard", dest="dashboard", action="store_false",
                        help="Disable Rich dashboard and use plain console logging")
    parser.add_argument("--strategy-mode", choices=["ml", "hybrid"], default="hybrid",
                        help="Entry validation mode. 'ml' uses model signals after safety filters; "
                             "'hybrid' also requires technical rule confirmation.")
    
    args = parser.parse_args()

    # Load custom yaml configurations if provided
    if args.config:
        apply_mt5_config_from_yaml(args.config)

    config.set_timeframe(normalize_timeframe(args.timeframe))

    config.set_symbol(args.symbol or config.symbol.symbol)

    # Dynamic symbol specification retrieval
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
                    config.symbol.min_lot = float(getattr(info, "volume_min", config.symbol.min_lot) or config.symbol.min_lot)
                    config.symbol.max_lot = float(getattr(info, "volume_max", config.symbol.max_lot) or config.symbol.max_lot)
                    config.symbol.lot_step = float(getattr(info, "volume_step", config.symbol.lot_step) or config.symbol.lot_step)
                    logger.info(f"Dynamically configured live trade specifications for {args.symbol}: "
                                f"Point={info.point}, Digits={info.digits}, "
                                f"ContractSize={info.trade_contract_size}, "
                                f"MinLot={config.symbol.min_lot}, MaxLot={config.symbol.max_lot}, "
                                f"LotStep={config.symbol.lot_step}")
                else:
                    logger.error(f"Symbol {args.symbol} not found in Market Watch. Make sure it matches broker spelling.")
                mt5.shutdown()
        except Exception as e:
            logger.warning(f"Could not dynamically query symbol properties from MT5: {e}")

    setup_logging(log_dir=config.paths.logs_dir, debug=config.debug, quiet_console=args.dashboard)
    
    paper_mode = not args.live
    run_trading_bot(args.model, paper_mode, strategy_mode=args.strategy_mode, dashboard_enabled=args.dashboard)


if __name__ == "__main__":
    main()
