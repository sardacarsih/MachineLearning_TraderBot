"""
============================================
Order Executor
============================================
Handles order placement, modifications, position closes, and execution status.
Includes Paper Trading simulator mode for testing without broker execution.
"""

import time
import random
import math
from typing import Dict, Any, List, Optional, Tuple, Union

from config.mt5_config import mt5_config
from config.settings import config
from utils.logger import get_logger, TradingLogger
from mt5.connector import MT5Connector

logger = get_logger()


class OrderExecutor:
    """
    Executes trades on MetaTrader 5 or simulates them in paper trading mode.
    """

    def __init__(self, connector: MT5Connector, account_manager: Optional["AccountManager"] = None):
        """
        Args:
            connector: MT5Connector instance for terminal/symbol access.
            account_manager: Optional AccountManager so paper-mode closes can
                update the virtual balance and realised PnL. When None, paper
                mode logs the trade but does not mutate balance state.
        """
        self.connector = connector
        self.account_manager = account_manager
        self.paper_mode = mt5_config.paper_trading

        # Paper trading virtual tracker
        if self.paper_mode:
            from mt5.paper_db import PaperDBManager
            self._db = PaperDBManager()
            db_positions = self._db.get_positions()
            self._virtual_positions = {pos["ticket"]: pos for pos in db_positions}
            self._virtual_ticket_counter = self._db.get_next_ticket(100000)
            logger.info(f"Loaded {len(self._virtual_positions)} virtual positions from persistent SQLite DB.")
        else:
            self._db = None
            self._virtual_positions = {}
            self._virtual_ticket_counter = 100000

        logger.info(
            f"OrderExecutor initialized. Paper Trading Mode: {self.paper_mode}, "
            f"account_manager_wired={account_manager is not None}"
        )


    def _get_filling_type(self, symbol: str = None) -> int:
        """Auto-detect the filling type supported by the broker for the given symbol.
        
        Queries symbol_info().filling_mode bitmask to determine which filling
        types are allowed, and picks the first supported one in order:
        FOK → IOC → RETURN.
        
        Falls back to the config-specified filling type if symbol info is unavailable.
        """
        import MetaTrader5 as mt5
        
        # Safe lookup of symbol filling constants not defined in some MetaTrader5 python versions
        symbol_filling_fok = getattr(mt5, "SYMBOL_FILLING_FOK", 1)
        symbol_filling_ioc = getattr(mt5, "SYMBOL_FILLING_IOC", 2)
        
        if symbol and self.connector.is_connected():
            info = mt5.symbol_info(symbol)
            if info is not None:
                filling_mode = info.filling_mode
                # Check bitmask for supported filling types (try in order)
                if filling_mode & symbol_filling_fok:
                    logger.debug(f"Symbol {symbol}: using FILLING_FOK (supported)")
                    return mt5.ORDER_FILLING_FOK
                elif filling_mode & symbol_filling_ioc:
                    logger.debug(f"Symbol {symbol}: using FILLING_IOC (supported)")
                    return mt5.ORDER_FILLING_IOC
                else:
                    # RETURN is always allowed for exchange symbols, use as fallback
                    logger.debug(f"Symbol {symbol}: using FILLING_RETURN (fallback)")
                    return mt5.ORDER_FILLING_RETURN

        # Fallback: use config value
        fill_map = {
            "FILLING_FOK": mt5.ORDER_FILLING_FOK,
            "FILLING_IOC": mt5.ORDER_FILLING_IOC,
            "FILLING_RETURN": mt5.ORDER_FILLING_RETURN
        }
        return fill_map.get(mt5_config.filling_type, mt5.ORDER_FILLING_IOC)

    def get_current_spread(self, symbol: str) -> float:
        """Get current spread in points."""
        if self.paper_mode:
            # Try to get real tick spread even in paper trading if MT5 is connected
            if self.connector.is_connected():
                import MetaTrader5 as mt5
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    return float(tick.ask - tick.bid) / self.connector.get_symbol_info(symbol).point
            return 15.0  # Default 1.5 pips for paper trading when offline

        import MetaTrader5 as mt5
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise ValueError(f"Failed to fetch tick info for {symbol}")
        
        point = self.connector.get_symbol_info(symbol).point
        return float((tick.ask - tick.bid) / point)

    def check_slippage(self, requested_price: float, filled_price: float, symbol: str) -> float:
        """Returns slippage in points."""
        point = self.connector.get_symbol_info(symbol).point
        slippage_pts = abs(requested_price - filled_price) / point
        return round(slippage_pts, 1)

    def _volume_decimals(self, step: float) -> int:
        step_text = f"{step:.10f}".rstrip("0").rstrip(".")
        return len(step_text.split(".")[1]) if "." in step_text else 0

    def _floor_to_step(self, lot: float, step: float) -> float:
        if step <= 0:
            return float(lot)
        decimals = self._volume_decimals(step)
        return round(math.floor((float(lot) + 1e-12) / step) * step, decimals)

    def _fit_volume_bounds(self, symbol: str, lot: float) -> Tuple[Optional[float], str]:
        info = self.connector.get_symbol_info(symbol) if self.connector.is_connected() else None
        min_lot = float(getattr(info, "volume_min", config.symbol.min_lot) or config.symbol.min_lot)
        max_lot = float(getattr(info, "volume_max", config.symbol.max_lot) or config.symbol.max_lot)
        step = float(getattr(info, "volume_step", config.symbol.lot_step) or config.symbol.lot_step)

        effective_max = min(max_lot, float(mt5_config.safety_max_lot))
        if effective_max < min_lot:
            return None, (
                f"No valid lot size: safety_max_lot {mt5_config.safety_max_lot} "
                f"is below broker min lot {min_lot}"
            )

        safe_lot = min(float(lot), effective_max)
        safe_lot = self._floor_to_step(safe_lot, step)
        if safe_lot < min_lot:
            return None, f"Safe lot {safe_lot} is below broker min lot {min_lot}"

        if safe_lot < float(lot):
            logger.warning(f"Lot clipped for {symbol}: requested={lot}, safe={safe_lot}")
        return safe_lot, "OK"

    def _order_check_passes(self, mt5, request: Dict[str, Any]) -> Tuple[bool, str]:
        order_check = getattr(mt5, "order_check", None)
        if order_check is None:
            return True, "order_check unavailable"

        result = order_check(request)
        if result is None:
            return False, f"order_check returned None, last_error={mt5.last_error()}"

        accepted = {
            code for code in (
                getattr(mt5, "TRADE_RETCODE_DONE", None),
                getattr(mt5, "TRADE_RETCODE_PLACED", None),
                getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", None),
            )
            if code is not None
        }
        retcode = getattr(result, "retcode", None)
        comment = getattr(result, "comment", "")
        if retcode in accepted:
            return True, f"order_check accepted: retcode={retcode}, comment={comment!r}"

        normalized_comment = str(comment or "").strip().lower()
        if retcode == 0 and normalized_comment in {"done", "ok", "success", "successful"}:
            logger.info(
                f"order_check accepted broker non-standard success retcode: "
                f"retcode={retcode}, comment={comment!r}"
            )
            return True, f"order_check accepted non-standard success: retcode={retcode}, comment={comment!r}"

        return False, f"order_check failed: retcode={retcode}, comment={comment!r}"

    def _margin_fits(self, mt5, action_type: int, symbol: str, lot: float, price: float) -> Tuple[bool, str]:
        account = mt5.account_info()
        if account is None:
            return False, "account_info unavailable"

        margin = mt5.order_calc_margin(action_type, symbol, float(lot), float(price))
        if margin is None:
            return False, f"order_calc_margin returned None, last_error={mt5.last_error()}"

        free_margin = float(getattr(account, "margin_free", 0.0))
        usable_margin = float(getattr(account, "margin_free", 0.0)) * (1.0 - float(mt5_config.margin_reserve_pct))
        if float(margin) > usable_margin:
            return False, (
                f"required_margin={float(margin):.2f} > usable_free_margin={usable_margin:.2f} "
                f"(free_margin={free_margin:.2f}, reserve={mt5_config.margin_reserve_pct:.2%})"
            )

        return True, f"required_margin={float(margin):.2f}, usable_free_margin={usable_margin:.2f}"

    def _find_margin_safe_lot(
        self,
        mt5,
        symbol: str,
        action_type: int,
        price: float,
        requested_lot: float,
        request_template: Dict[str, Any],
    ) -> Tuple[Optional[float], str]:
        info = self.connector.get_symbol_info(symbol)
        min_lot = float(getattr(info, "volume_min", config.symbol.min_lot) or config.symbol.min_lot)
        step = float(getattr(info, "volume_step", config.symbol.lot_step) or config.symbol.lot_step)
        lot = self._floor_to_step(requested_lot, step)
        last_reason = "not checked"

        while lot >= min_lot:
            request = dict(request_template)
            request["volume"] = float(lot)
            margin_ok, margin_reason = self._margin_fits(mt5, action_type, symbol, lot, price)
            if margin_ok:
                check_ok, check_reason = self._order_check_passes(mt5, request)
            else:
                check_ok, check_reason = False, "order_check skipped because margin did not fit"

            if margin_ok and check_ok:
                if lot < requested_lot:
                    logger.warning(f"Lot downsized by margin/order check: requested={requested_lot}, safe={lot}")
                return lot, "OK"

            last_reason = f"lot={lot}: {margin_reason}; {check_reason}"
            lot = self._floor_to_step(lot - step, step)

        return None, f"No lot >= broker min {min_lot} passed margin/order checks. Last check: {last_reason}"

    def _get_all_symbol_positions(self, symbol: str) -> List[Any]:
        """Return all open positions for a symbol, regardless of magic number."""
        if self.paper_mode:
            return [pos for pos in self._virtual_positions.values() if pos["symbol"] == symbol]

        if not self.connector.is_connected():
            return []

        import MetaTrader5 as mt5
        positions = mt5.positions_get(symbol=symbol)
        return list(positions or [])

    def _position_direction(self, pos: Any) -> Optional[str]:
        raw_type = pos.get("type") if isinstance(pos, dict) else getattr(pos, "type", None)
        if isinstance(raw_type, str):
            direction = raw_type.upper()
            return direction if direction in {"BUY", "SELL"} else None

        if self.paper_mode:
            return None

        import MetaTrader5 as mt5
        if raw_type == getattr(mt5, "POSITION_TYPE_BUY", 0):
            return "BUY"
        if raw_type == getattr(mt5, "POSITION_TYPE_SELL", 1):
            return "SELL"
        return None

    def _position_open_price(self, pos: Any) -> float:
        if isinstance(pos, dict):
            return float(pos["open_price"])
        return float(getattr(pos, "price_open"))

    def _position_volume(self, pos: Any) -> float:
        if isinstance(pos, dict):
            return float(pos["volume"])
        return float(getattr(pos, "volume"))

    def can_open_averaging_entry(
        self,
        symbol: str,
        order_type: str,
        entry_price: float,
    ) -> Tuple[bool, str]:
        """
        Enforce per-symbol max positions and averaging-only add-on entries.

        The first symbol entry is allowed. Later entries must match the existing
        direction and enter at a worse price than the volume-weighted average.
        """
        symbol_positions = self._get_all_symbol_positions(symbol)
        position_count = len(symbol_positions)
        max_positions = int(config.risk.max_open_positions)

        if position_count >= max_positions:
            return False, (
                f"Max open positions blocked trade for {symbol}: "
                f"{position_count} open >= limit {max_positions}"
            )

        if not symbol_positions:
            return True, "First symbol position allowed"

        position_directions = [self._position_direction(pos) for pos in symbol_positions]
        if any(direction is None for direction in position_directions):
            return False, f"Averaging blocked for {symbol}: unknown existing position direction"

        directions = set(position_directions)
        if len(directions) != 1:
            return False, f"Averaging blocked for {symbol}: mixed or unknown existing position directions"

        existing_direction = next(iter(directions))
        if existing_direction != order_type:
            return False, (
                f"Averaging blocked for {symbol}: requested {order_type} but "
                f"existing direction is {existing_direction}"
            )

        total_volume = sum(self._position_volume(pos) for pos in symbol_positions)
        if total_volume <= 0:
            return False, f"Averaging blocked for {symbol}: existing position volume is zero"

        average_price = sum(
            self._position_open_price(pos) * self._position_volume(pos)
            for pos in symbol_positions
        ) / total_volume

        if order_type == "BUY" and float(entry_price) < average_price:
            return True, f"BUY averaging allowed: entry {entry_price:.5f} < average {average_price:.5f}"
        if order_type == "SELL" and float(entry_price) > average_price:
            return True, f"SELL averaging allowed: entry {entry_price:.5f} > average {average_price:.5f}"

        return False, (
            f"Averaging blocked for {symbol}: {order_type} entry {entry_price:.5f} "
            f"is not worse than average {average_price:.5f}"
        )

    def open_position(
        self, symbol: str, order_type: str, lot: float,
        sl: float, tp: float, comment: str = ""
    ) -> Tuple[bool, Union[int, str]]:
        """
        Open a market position (BUY or SELL).

        Args:
            symbol: Trading symbol (e.g. XAUUSD).
            order_type: 'BUY' or 'SELL'.
            lot: Volume in lots.
            sl: Stop loss price.
            tp: Take profit price.
            comment: Optional trade comment.

        Returns:
            Tuple: (success: bool, ticket_or_error: int or str)
        """
        comment = comment or mt5_config.order_comment
        
        if order_type not in ["BUY", "SELL"]:
            return False, f"Invalid order type: {order_type}"

        # ----------------------------------------------------
        # Paper Trading Simulation Mode
        # ----------------------------------------------------
        if self.paper_mode:
            lot, reason = self._fit_volume_bounds(symbol, lot)
            if lot is None:
                return False, reason

            # Fetch current prices for entry
            entry_price = 2000.00
            if self.connector.is_connected():
                import MetaTrader5 as mt5
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    entry_price = tick.ask if order_type == "BUY" else tick.bid

            can_open, reason = self.can_open_averaging_entry(symbol, order_type, entry_price)
            if not can_open:
                TradingLogger.trade_log(f"[TRADE SKIPPED] {symbol} {order_type} | Reason: {reason}")
                logger.warning(f"Trade skipped: {reason}")
                return False, f"Trade skipped: {reason}"

            self._virtual_ticket_counter += 1
            ticket = self._virtual_ticket_counter

            pos_dict = {
                "ticket": ticket,
                "symbol": symbol,
                "type": order_type,
                "volume": lot,
                "open_price": entry_price,
                "sl": sl,
                "tp": tp,
                "comment": comment,
                "time": time.time()
            }
            self._virtual_positions[ticket] = pos_dict
            if self._db:
                self._db.save_position(pos_dict)
            
            msg = (
                f"[PAPER TRADE OPEN] Ticket: {ticket} | {symbol} {order_type} "
                f"{lot} lots at {entry_price:.2f} | SL: {sl:.2f}, TP: {tp:.2f}"
            )
            TradingLogger.trade_log(msg)
            logger.info(msg)
            return True, ticket


        # ----------------------------------------------------
        # Live/Demo MetaTrader 5 Order Execution
        # ----------------------------------------------------
        if not self.connector.is_connected():
            return False, "MT5 terminal not connected"

        import MetaTrader5 as mt5

        # Get tick data
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return False, f"Cannot get tick price for {symbol}"

        # Determine price & order type constant
        if order_type == "BUY":
            price = tick.ask
            action_type = mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid
            action_type = mt5.ORDER_TYPE_SELL

        can_open, reason = self.can_open_averaging_entry(symbol, order_type, price)
        if not can_open:
            TradingLogger.trade_log(f"[TRADE SKIPPED] {symbol} {order_type} | Reason: {reason}")
            logger.warning(f"Trade skipped: {reason}")
            return False, f"Trade skipped: {reason}"

        # Create request dictionary
        lot, reason = self._fit_volume_bounds(symbol, lot)
        if lot is None:
            return False, reason

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot),
            "type": action_type,
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": int(mt5_config.max_deviation),
            "magic": int(mt5_config.magic_number),
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_type(symbol),
        }

        lot, reason = self._find_margin_safe_lot(mt5, symbol, action_type, price, lot, request)
        if lot is None:
            msg = f"Trade skipped: {reason}"
            TradingLogger.trade_log(f"[TRADE SKIPPED] {symbol} {order_type} | Reason: {reason}")
            logger.warning(msg)
            return False, msg
        request["volume"] = float(lot)

        logger.info(f"Sending MT5 order request: {order_type} {lot} lots at {price:.2f}...")
        
        result = mt5.order_send(request)
        
        if result is None:
            err = mt5.last_error()
            return False, f"order_send returned None. Error: {err}"

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            err_desc = self._get_error_description(result.retcode)
            msg = f"Order failed. Code: {result.retcode} ({err_desc})"
            TradingLogger.trade_log(f"[TRADE FAILED] {symbol} {order_type} {lot} lots | Reason: {err_desc}")
            return False, msg

        # Calculate slippage
        slippage = self.check_slippage(price, result.price, symbol)
        
        success_msg = (
            f"[TRADE OPENED] Ticket: {result.order} | {symbol} {order_type} "
            f"{result.volume} lots at {result.price:.2f} (Requested: {price:.2f}, Slippage: {slippage} pts) | "
            f"SL: {result.request.sl:.2f}, TP: {result.request.tp:.2f}"
        )
        TradingLogger.trade_log(success_msg)
        logger.info(success_msg)

        return True, result.order

    def close_position(self, ticket: int, exit_reason: Optional[str] = None) -> bool:
        """
        Closes a specific open position.

        Args:
            ticket: Position ticket number.
            exit_reason: Optional close reason label, e.g. 'TP' or 'SL'.
        """
        # ----------------------------------------------------
        # Paper Trading Simulation Mode
        # ----------------------------------------------------
        if self.paper_mode:
            if ticket not in self._virtual_positions:
                logger.error(f"Cannot close: ticket {ticket} not found in paper trading positions")
                return False
            pos = self._virtual_positions.pop(ticket)
            if self._db:
                self._db.delete_position(ticket)

            # Fetch close price
            close_price = pos["open_price"]
            if self.connector.is_connected():
                import MetaTrader5 as mt5
                tick = mt5.symbol_info_tick(pos["symbol"])
                if tick:
                    close_price = tick.bid if pos["type"] == "BUY" else tick.ask

            # XAUUSD PNL = Lot * ContractSize * (Close - Open)
            pnl_direction = 1 if pos["type"] == "BUY" else -1
            contract_size = config.symbol.contract_size
            gross_pnl = pos["volume"] * contract_size * (close_price - pos["open_price"]) * pnl_direction

            # Apply round-trip commission
            commission = pos["volume"] * config.backtest.commission_per_lot
            net_pnl = gross_pnl - commission

            # Mutate virtual account so the paper book keeps a real ledger.
            if self.account_manager is not None:
                self.account_manager.apply_paper_pnl(net_pnl)
            else:
                logger.warning(
                    "OrderExecutor closed paper position but no account_manager is wired; "
                    "virtual balance not updated."
                )

            msg = (
                f"[PAPER TRADE CLOSE] Ticket: {ticket} | {pos['symbol']} {pos['type']} "
                f"closed at {close_price:.2f} (Entry: {pos['open_price']:.2f}) | "
                f"Gross PnL: {gross_pnl:.2f}, Net PnL: {net_pnl:.2f} (Comm: {commission:.2f})"
            )
            if exit_reason:
                msg += f" | ExitReason: {str(exit_reason).upper()}"
            TradingLogger.trade_log(msg)
            logger.info(msg)
            return True

        # ----------------------------------------------------
        # Live/Demo MetaTrader 5 Order Execution
        # ----------------------------------------------------
        if not self.connector.is_connected():
            return False

        import MetaTrader5 as mt5

        # Fetch open positions to get volume and symbol
        positions = mt5.positions_get(ticket=ticket)
        if not positions or len(positions) == 0:
            logger.error(f"Cannot close position {ticket}: not found or already closed")
            return False

        pos = positions[0]
        symbol = pos.symbol
        volume = pos.volume
        pos_type = pos.type

        # Fetch tick
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return False

        # Set close details (reverse transaction)
        if pos_type == mt5.POSITION_TYPE_BUY:
            close_price = tick.bid
            action_type = mt5.ORDER_TYPE_SELL
        else:
            close_price = tick.ask
            action_type = mt5.ORDER_TYPE_BUY

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": action_type,
            "position": int(ticket),
            "price": float(close_price),
            "deviation": int(mt5_config.max_deviation),
            "magic": int(mt5_config.magic_number),
            "comment": f"Close {ticket}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_type(symbol),
        }

        result = mt5.order_send(request)
        if result is None:
            logger.error(f"Failed to close position {ticket}. order_send returned None")
            return False

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            err_desc = self._get_error_description(result.retcode)
            logger.error(f"Failed to close position {ticket}. Code: {result.retcode} ({err_desc})")
            return False

        msg = (
            f"[TRADE CLOSED] Ticket: {ticket} | {symbol} closed at {result.price:.2f} "
            f"(Entry: {pos.price_open:.2f}, Profit: {pos.profit + pos.swap:.2f})"
        )
        TradingLogger.trade_log(msg)
        logger.info(msg)
        return True

    def close_all_positions(self, symbol: str) -> bool:
        """Close all open positions for a symbol."""
        open_positions = self.get_open_positions(symbol)
        if not open_positions:
            return True

        logger.info(f"Closing all open positions ({len(open_positions)}) for {symbol}...")
        success = True
        for pos in open_positions:
            ticket = pos.get("ticket") if self.paper_mode else getattr(pos, "ticket")
            if not self.close_position(ticket):
                success = False

        return success

    def modify_position(self, ticket: int, sl: float, tp: float) -> bool:
        """
        Modifies SL/TP for an open position.
        """
        # ----------------------------------------------------
        # Paper Trading Simulation Mode
        # ----------------------------------------------------
        if self.paper_mode:
            if ticket not in self._virtual_positions:
                return False
            self._virtual_positions[ticket]["sl"] = sl
            self._virtual_positions[ticket]["tp"] = tp
            if self._db:
                self._db.save_position(self._virtual_positions[ticket])
            logger.info(f"[PAPER TRADE MODIFY] Ticket: {ticket} updated. New SL: {sl:.2f}, TP: {tp:.2f}")
            return True

        # ----------------------------------------------------
        # Live/Demo MetaTrader 5 Order Execution
        # ----------------------------------------------------
        if not self.connector.is_connected():
            return False

        import MetaTrader5 as mt5

        # Check if position exists
        positions = mt5.positions_get(ticket=ticket)
        if not positions or len(positions) == 0:
            return False
        
        pos = positions[0]

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": int(ticket),
            "sl": float(sl),
            "tp": float(tp)
        }

        result = mt5.order_send(request)
        if result is None:
            logger.error(f"Modify position failed for ticket {ticket}: order_send returned None")
            return False

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            err_desc = self._get_error_description(result.retcode)
            logger.error(f"Modify position failed for ticket {ticket}. Code: {result.retcode} ({err_desc})")
            return False

        logger.info(f"[TRADE MODIFIED] Ticket: {ticket} | SL: {sl:.2f}, TP: {tp:.2f}")
        return True

    def get_open_positions(self, symbol: str) -> List[Any]:
        """
        Returns list of all open positions for a symbol.
        """
        if self.paper_mode:
            # Filter in-memory dict
            return [pos for pos in self._virtual_positions.values() if pos["symbol"] == symbol]

        if not self.connector.is_connected():
            return []

        import MetaTrader5 as mt5
        # Fetch by symbol
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            return []

        # Filter by magic number to only manage our bot's trades
        bot_positions = [
            p for p in positions
            if getattr(p, 'magic', 0) == mt5_config.magic_number
        ]
        return bot_positions

    def get_position_by_ticket(self, ticket: int) -> Optional[Any]:
        """Gets a specific position by its ticket ID."""
        if self.paper_mode:
            return self._virtual_positions.get(ticket)

        if not self.connector.is_connected():
            return None

        import MetaTrader5 as mt5
        positions = mt5.positions_get(ticket=ticket)
        if positions and len(positions) > 0:
            return positions[0]
        return None

    def _get_error_description(self, retcode: int) -> str:
        """Translate MT5 trade return code into a human-readable string."""
        import MetaTrader5 as mt5

        errors = {}

        def add(name: str, description: str):
            if hasattr(mt5, name):
                errors[getattr(mt5, name)] = description

        add("TRADE_RETCODE_REQUOTE", "Requote: price changed")
        add("TRADE_RETCODE_REJECT", "Trade request rejected")
        add("TRADE_RETCODE_CANCEL", "Trade request canceled")
        add("TRADE_RETCODE_PLACED", "Order placed")
        add("TRADE_RETCODE_DONE_PARTIAL", "Trade partially completed")
        add("TRADE_RETCODE_ERROR", "Trade server error")
        add("TRADE_RETCODE_TIMEOUT", "Operation timed out")
        add("TRADE_RETCODE_INVALID", "Invalid request format")
        add("TRADE_RETCODE_INVALID_VOLUME", "Invalid lot size volume")
        add("TRADE_RETCODE_INVALID_PRICE", "Invalid price")
        add("TRADE_RETCODE_INVALID_STOPS", "Invalid stop loss or take profit")
        add("TRADE_RETCODE_TRADE_DISABLED", "Trading disabled")
        add("TRADE_RETCODE_MARKET_CLOSED", "Market is closed")
        add("TRADE_RETCODE_NO_MONEY", "Insufficient margin/balance")
        add("TRADE_RETCODE_PRICE_CHANGED", "Price changed")
        add("TRADE_RETCODE_PRICE_OFF", "No prices available (off quotes)")
        add("TRADE_RETCODE_INVALID_EXPIRATION", "Invalid order expiration")
        add("TRADE_RETCODE_ORDER_CHANGED", "Order state changed")
        add("TRADE_RETCODE_TOO_MANY_REQUESTS", "Too many trade requests")
        add("TRADE_RETCODE_NO_CHANGES", "No changes in request")
        add("TRADE_RETCODE_SERVER_DISABLES_AT", "Autotrading disabled by server")
        add("TRADE_RETCODE_CLIENT_DISABLES_AT", "Autotrading disabled by client terminal")
        add("TRADE_RETCODE_LOCKED", "Trade request locked")
        add("TRADE_RETCODE_FROZEN", "Order or position frozen")
        add("TRADE_RETCODE_INVALID_FILL", "Invalid filling type")
        add("TRADE_RETCODE_CONNECTION", "Terminal connection lost")
        add("TRADE_RETCODE_ONLY_REAL", "Operation only supported on real accounts")
        add("TRADE_RETCODE_LIMIT_ORDERS", "Trade order limit reached")
        add("TRADE_RETCODE_LIMIT_VOLUME", "Trade volume limit reached")
        add("TRADE_RETCODE_INVALID_ORDER", "Invalid order type")
        add("TRADE_RETCODE_POSITION_CLOSED", "Position already closed")
        add("TRADE_RETCODE_INVALID_CLOSE_VOLUME", "Invalid close volume")
        add("TRADE_RETCODE_CLOSE_ORDER_EXIST", "Close order already exists")
        add("TRADE_RETCODE_LIMIT_POSITIONS", "Open position limit reached")
        add("TRADE_RETCODE_REJECT_CANCEL", "Cancel request rejected")
        add("TRADE_RETCODE_LONG_ONLY", "Only long positions are allowed")
        add("TRADE_RETCODE_SHORT_ONLY", "Only short positions are allowed")
        add("TRADE_RETCODE_CLOSE_ONLY", "Only position closing is allowed")
        add("TRADE_RETCODE_FIFO_CLOSE", "Position must be closed by FIFO rule")

        return errors.get(retcode, f"Unknown MT5 error code: {retcode}")
