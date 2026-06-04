"""
============================================
Event-Driven Backtester
============================================
Runs historical backtests using engineered features and model signals.
Tracks open positions, SL/TP execution, trailing stops, commission, and slippage.
Calculates portfolio performance metrics (Sharpe, Sortino, Calmar, Max Drawdown).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from config.settings import config
from utils.logger import get_logger
from strategy.risk_manager import RiskManager
from strategy.trading_rules import TradingRules

logger = get_logger()


@dataclass
class BacktestResult:
    """Dataclass holding all backtest results and statistics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    winrate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    expectancy: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    equity_curve: pd.Series = field(default_factory=pd.Series)
    trade_log: List[Dict[str, Any]] = field(default_factory=list)


class Backtester:
    """
    Simulates live execution on historical data to evaluate strategy performance.
    """

    def __init__(
        self, initial_balance: float = None,
        commission_per_lot: float = None,
        slippage_points: float = None,
        strategy_mode: str = None
    ):
        """
        Initialize the Backtester.

        Args:
            initial_balance: Starting account balance (default from config).
            commission_per_lot: Round-trip fee in dollars per lot.
            slippage_points: Slippage execution cost in price points.
            strategy_mode: "hybrid" validates ML signals with technical rules;
                "ml" uses ML signals directly after safety filters.
        """
        self.initial_balance = initial_balance or config.backtest.initial_balance
        self.commission_per_lot = commission_per_lot or config.backtest.commission_per_lot
        self.slippage_points = slippage_points or config.backtest.slippage_points
        self.strategy_mode = self._normalize_strategy_mode(strategy_mode or config.strategy_mode)

        self.risk_manager = RiskManager()
        self.trading_rules = TradingRules()

        self.reset()

    @staticmethod
    def _normalize_strategy_mode(strategy_mode: str) -> str:
        """Normalize and validate strategy mode labels."""
        mode = (strategy_mode or "hybrid").strip().lower()
        if mode not in {"ml", "hybrid"}:
            raise ValueError("strategy_mode must be 'ml' or 'hybrid'")
        return mode

    def reset(self):
        """Reset the backtester state."""
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.trade_log: List[Dict[str, Any]] = []
        self.open_positions: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = []
        self.times: List[pd.Timestamp] = []
        self.results = BacktestResult()

    def run(
        self,
        df: pd.DataFrame,
        signals: List[str],
        confidences: Optional[List[float]] = None,
    ) -> BacktestResult:
        """
        Run the historical backtest simulation.

        Args:
            df: Historical features DataFrame (including open, high, low, close, volume).
            signals: List of model-generated signals ('BUY', 'SELL', 'NO_TRADE')
                     matching the indices of df.
            confidences: Optional model confidence values matching the indices
                         of df. When omitted, confidence lot scaling is skipped.

        Returns:
            BacktestResult object containing statistics.
        """
        self.reset()
        logger.info(
            f"Starting backtest simulation over {len(df)} bars "
            f"(strategy_mode={self.strategy_mode})..."
        )

        point = config.symbol.point
        contract_size = config.symbol.contract_size

        for t in range(len(df)):
            bar = df.iloc[t]
            bar_time = bar['time']
            bar_open = bar['open']
            bar_high = bar['high']
            bar_low = bar['low']
            bar_close = bar['close']
            bar_atr = bar.get('atr', 1.0)
            
            # 1. Update/check open positions (SL/TP check)
            self._manage_open_positions(bar_high, bar_low, bar_close, bar_time, bar_atr, point, contract_size)

            # Calculate current floating equity
            open_pnl = self._calculate_open_pnl(bar_close, contract_size)
            self.equity = self.balance + open_pnl
            self.equity_curve.append(self.equity)
            self.times.append(bar_time)

            # 2. Check for new entries if we are below max open positions
            if len(self.open_positions) < config.risk.max_open_positions:
                signal = signals[t]
                confidence = confidences[t] if confidences is not None else None
                
                if signal in ["BUY", "SELL"]:
                    features_dict = bar.to_dict()
                    spread = features_dict.get("spread", 15.0)

                    is_valid, reason = self._validate_entry_signal(signal, features_dict, spread)
                    
                    if is_valid:
                        self._execute_entry(signal, bar_close, bar_atr, bar_time, point, confidence)

        # Close any remaining open positions at the end of the backtest
        if self.open_positions:
            last_bar = df.iloc[-1]
            self._close_all_positions(last_bar['close'], last_bar['time'], contract_size)
            # Re-update final equity curve
            self.equity = self.balance
            self.equity_curve[-1] = self.equity

        # Compile final results
        self._calculate_metrics()
        return self.results

    def _validate_entry_signal(
        self, signal: str, features: Dict[str, Any], spread: float
    ) -> Tuple[bool, str]:
        """Apply strategy-mode specific validation to a model signal."""
        if self.strategy_mode == "hybrid":
            return self.trading_rules.validate_signal(signal, features, spread)

        should_avoid, avoid_reason = self.trading_rules.check_no_trade_conditions(features, spread)
        if should_avoid:
            return False, avoid_reason
        return True, "ML signal accepted after safety filters"

    def _manage_open_positions(
        self, high: float, low: float, close: float,
        current_time: pd.Timestamp, atr: float, point: float, contract_size: float
    ):
        """Update open positions, checking if SL/TP are hit or trailing stops should adjust."""
        remaining_positions = []
        slippage_cost = self.slippage_points * point

        for pos in self.open_positions:
            direction = pos['type']
            sl = pos['sl']
            tp = pos['tp']
            entry_price = pos['entry_price']
            vol = pos['volume']

            closed = False
            close_price = 0.0
            reason = ""

            # Check SL and TP execution
            if direction == "BUY":
                # Check Stop Loss (Conservative: Check SL first if both could be hit)
                if low <= sl:
                    closed = True
                    # Fill at Stop Loss price minus slippage
                    close_price = sl - slippage_cost
                    reason = "SL"
                elif high >= tp:
                    closed = True
                    close_price = tp - slippage_cost
                    reason = "TP"
            else:  # SELL
                if high >= sl:
                    closed = True
                    close_price = sl + slippage_cost
                    reason = "SL"
                elif low <= tp:
                    closed = True
                    close_price = tp + slippage_cost
                    reason = "TP"

            if closed:
                # Calculate trade profit/loss
                pnl_dir = 1 if direction == "BUY" else -1
                gross_pnl = vol * contract_size * (close_price - entry_price) * pnl_dir
                
                # Apply transaction commission fee
                commission = vol * self.commission_per_lot
                net_pnl = gross_pnl - commission

                # Update account balance
                self.balance += net_pnl
                
                # Record closed trade
                trade_record = {
                    "ticket": pos["ticket"],
                    "open_time": pos["open_time"],
                    "close_time": current_time,
                    "symbol": config.symbol.symbol,
                    "type": direction,
                    "volume": vol,
                    "entry_price": entry_price,
                    "close_price": close_price,
                    "sl": sl,
                    "tp": tp,
                    "gross_pnl": gross_pnl,
                    "commission": commission,
                    "net_pnl": net_pnl,
                    "close_reason": reason,
                    "duration_bars": len(self.equity_curve) - pos["open_bar_idx"]
                }
                self.trade_log.append(trade_record)
            else:
                # Not closed, check for Trailing Stop adjustment
                new_sl = self.risk_manager.calculate_trailing_stop(entry_price, close, sl, atr, direction, tp)
                if new_sl is not None:
                    pos['sl'] = new_sl  # Trail the stop loss
                remaining_positions.append(pos)

        self.open_positions = remaining_positions

    def _calculate_open_pnl(self, current_price: float, contract_size: float) -> float:
        """Calculate total unrealized floating profit and loss."""
        open_pnl = 0.0
        for pos in self.open_positions:
            direction = pos['type']
            entry_price = pos['entry_price']
            vol = pos['volume']
            pnl_dir = 1 if direction == "BUY" else -1
            
            gross_pnl = vol * contract_size * (current_price - entry_price) * pnl_dir
            commission = vol * self.commission_per_lot
            open_pnl += (gross_pnl - commission)
        return open_pnl

    def _execute_entry(
        self,
        direction: str,
        entry_price: float,
        atr: float,
        current_time: pd.Timestamp,
        point: float,
        confidence: Optional[float] = None,
    ):
        """Creates and opens a new trade position."""
        # Calculate SL & TP
        sl = self.risk_manager.calculate_sl(entry_price, atr, direction)
        
        # Calculate SL distance in price
        sl_distance = abs(entry_price - sl)
        
        tp = self.risk_manager.calculate_tp(entry_price, sl_distance, direction)

        # Calculate dynamic lot size (1% risk of current balance)
        lot = self.risk_manager.calculate_position_size(self.balance, sl_distance)
        lot = self.risk_manager.apply_confidence_lot_multiplier(lot, confidence)

        # Apply slippage on entry
        slippage_cost = self.slippage_points * point
        adjusted_entry = entry_price + slippage_cost if direction == "BUY" else entry_price - slippage_cost

        # Record position details
        ticket = 200000 + len(self.trade_log) + len(self.open_positions) + 1
        new_pos = {
            "ticket": ticket,
            "open_time": current_time,
            "open_bar_idx": len(self.equity_curve),
            "type": direction,
            "volume": lot,
            "entry_price": adjusted_entry,
            "sl": sl,
            "tp": tp
        }
        self.open_positions.append(new_pos)

    def _close_all_positions(self, current_price: float, current_time: pd.Timestamp, contract_size: float):
        """Force close all positions at current price (end of backtest)."""
        for pos in self.open_positions:
            direction = pos['type']
            entry_price = pos['entry_price']
            vol = pos['volume']

            pnl_dir = 1 if direction == "BUY" else -1
            gross_pnl = vol * contract_size * (current_price - entry_price) * pnl_dir
            commission = vol * self.commission_per_lot
            net_pnl = gross_pnl - commission

            self.balance += net_pnl

            trade_record = {
                "ticket": pos["ticket"],
                "open_time": pos["open_time"],
                "close_time": current_time,
                "symbol": config.symbol.symbol,
                "type": direction,
                "volume": vol,
                "entry_price": entry_price,
                "close_price": current_price,
                "sl": pos["sl"],
                "tp": pos["tp"],
                "gross_pnl": gross_pnl,
                "commission": commission,
                "net_pnl": net_pnl,
                "close_reason": "FORCE_CLOSE",
                "duration_bars": len(self.equity_curve) - pos["open_bar_idx"]
            }
            self.trade_log.append(trade_record)
        self.open_positions = []

    def get_equity_curve(self) -> pd.Series:
        """Returns the final equity curve Series."""
        return pd.Series(self.equity_curve, index=self.times)

    def get_trade_log(self) -> pd.DataFrame:
        """Returns a DataFrame containing all completed trades."""
        return pd.DataFrame(self.trade_log)

    def _calculate_metrics(self):
        """Calculates and populates all backtest statistics."""
        if not self.trade_log:
            self.results = BacktestResult(
                equity_curve=pd.Series([self.initial_balance], index=[pd.Timestamp.now()])
            )
            return

        trade_df = pd.DataFrame(self.trade_log)
        total_trades = len(trade_df)
        wins = trade_df[trade_df['net_pnl'] > 0]
        losses = trade_df[trade_df['net_pnl'] <= 0]
        
        winning_trades = len(wins)
        losing_trades = len(losses)

        winrate = winning_trades / total_trades if total_trades > 0 else 0.0

        gross_profit = wins['net_pnl'].sum() if winning_trades > 0 else 0.0
        gross_loss = abs(losses['net_pnl'].sum()) if losing_trades > 0 else 0.0
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 1.0)
        net_profit = trade_df['net_pnl'].sum()

        average_win = wins['net_pnl'].mean() if winning_trades > 0 else 0.0
        average_loss = losses['net_pnl'].mean() if losing_trades > 0 else 0.0

        # Win rate * average win - loss rate * average loss
        loss_rate = losing_trades / total_trades if total_trades > 0 else 0.0
        expectancy = (winrate * average_win) + (loss_rate * average_loss)

        # Equity metrics
        eq_series = pd.Series(self.equity_curve, index=self.times)
        
        # Max drawdown
        cum_max = eq_series.cummax()
        drawdowns = cum_max - eq_series
        max_drawdown = drawdowns.max()
        
        drawdown_pcts = drawdowns / cum_max
        max_drawdown_pct = drawdown_pcts.max()

        # Sharpe ratio (from daily returns)
        daily_eq = eq_series.resample('D').last().ffill()
        daily_returns = daily_eq.pct_change().dropna()
        
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            # Annualized Sharpe (assuming 252 trading days)
            sharpe_ratio = float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252))
            
            # Sortino ratio (downside deviation only)
            downside_returns = daily_returns[daily_returns < 0]
            downside_std = downside_returns.std()
            if downside_std > 0:
                sortino_ratio = float((daily_returns.mean() / downside_std) * np.sqrt(252))
            else:
                sortino_ratio = 0.0
        else:
            sharpe_ratio = 0.0
            sortino_ratio = 0.0

        # Calmar Ratio: Annualized Return / Max Drawdown Pct
        years = (eq_series.index[-1] - eq_series.index[0]).days / 365.25
        if years > 0 and max_drawdown_pct > 0:
            cagr = (self.balance / self.initial_balance) ** (1 / years) - 1
            calmar_ratio = cagr / max_drawdown_pct
        else:
            calmar_ratio = 0.0

        self.results = BacktestResult(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            winrate=winrate,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            expectancy=expectancy,
            average_win=average_win,
            average_loss=average_loss,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            net_profit=net_profit,
            equity_curve=eq_series,
            trade_log=self.trade_log
        )
