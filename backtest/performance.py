"""
Backtest performance reporting and plots.
"""

import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config.settings import config
from utils.logger import get_logger

logger = get_logger()


class PerformanceAnalyzer:
    """Exports text reports and charts for a BacktestResult."""

    def __init__(self, output_dir: str = None, strategy_mode: str = "hybrid"):
        self.output_dir = output_dir or config.paths.backtest_dir
        self.strategy_mode = (strategy_mode or "hybrid").strip().lower()
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"PerformanceAnalyzer initialized, saving reports/plots to {self.output_dir}")

    def export_report(self, results):
        """Write the text report and all standard performance charts."""
        report_path = os.path.join(self.output_dir, "performance_report.txt")
        trade_df = pd.DataFrame(results.trade_log)

        close_reasons = Counter()
        avg_hold = 0.0
        if not trade_df.empty:
            close_reasons = Counter(trade_df.get("close_reason", []))
            if "duration_bars" in trade_df:
                avg_hold = float(trade_df["duration_bars"].mean())

        lines = [
            "=" * 70,
            "XAUUSD Trading Bot - Backtest Performance Report",
            "=" * 70,
            "",
            "Run Configuration:",
            f"  Symbol:                     {config.symbol.symbol}",
            f"  Timeframe:                  {config.symbol.timeframe}",
            f"  Strategy Mode:              {self.strategy_mode.upper()}",
            "",
            "Account Summary:",
            f"  Initial Balance:            ${config.backtest.initial_balance:.2f}",
            f"  Net Profit:                 ${results.net_profit:.2f}",
            f"  Ending Balance:             ${config.backtest.initial_balance + results.net_profit:.2f}",
            f"  Total Trades:               {results.total_trades}",
            f"  Winning Trades:             {results.winning_trades} ({results.winrate * 100:.2f}%)",
            f"  Losing Trades:              {results.losing_trades} ({(1 - results.winrate) * 100:.2f}%)",
            f"  Profit Factor:              {results.profit_factor:.2f}",
            f"  Trade Expectancy:           ${results.expectancy:.2f}",
            "",
            "Performance Statistics:",
            f"  Max Drawdown ($):           ${results.max_drawdown:.2f}",
            f"  Max Drawdown (%):           {results.max_drawdown_pct * 100:.2f}%",
            f"  Sharpe Ratio (Daily):       {results.sharpe_ratio:.2f}",
            f"  Sortino Ratio (Daily):      {results.sortino_ratio:.2f}",
            f"  Calmar Ratio:               {results.calmar_ratio:.2f}",
            "",
            "Trade Averages:",
            f"  Average Win:                ${results.average_win:.2f}",
            f"  Average Loss:               ${results.average_loss:.2f}",
            f"  Win / Loss Ratio:           {abs(results.average_win / results.average_loss) if results.average_loss else 0.0:.2f}",
            f"  Gross Profit:               ${results.gross_profit:.2f}",
            f"  Gross Loss:                 ${results.gross_loss:.2f}",
            "",
            "Close Reason Breakdown:",
        ]

        for reason, count in close_reasons.items():
            pct = count / results.total_trades * 100 if results.total_trades else 0.0
            lines.append(f"  {reason:<15} {count:<5} ({pct:.2f}%)")
        lines.extend([
            "",
            f"  Average Hold Time:          {avg_hold:.1f} {config.symbol.timeframe} bars",
        ])

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Performance report exported to {report_path}")

        self.plot_equity_curve(results)
        self.plot_drawdown(results)
        self.plot_monthly_returns(results)
        self.plot_trade_distribution(results)
        self.plot_cumulative_pnl(results)
        return report_path

    def plot_equity_curve(self, results):
        eq = results.equity_curve
        if eq.empty:
            return None
        path = os.path.join(self.output_dir, "equity_curve.png")
        plt.figure(figsize=(12, 6))
        plt.plot(eq.index, eq.values, label="Equity")
        plt.title("Equity Curve")
        plt.xlabel("Time")
        plt.ylabel("Equity")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Equity curve plot saved to {path}")
        return path

    def plot_drawdown(self, results):
        eq = results.equity_curve
        if eq.empty:
            return None
        path = os.path.join(self.output_dir, "drawdown.png")
        drawdown_pct = (eq.cummax() - eq) / eq.cummax() * 100
        plt.figure(figsize=(12, 5))
        plt.fill_between(drawdown_pct.index, drawdown_pct.values, color="red", alpha=0.35)
        plt.title("Drawdown")
        plt.xlabel("Time")
        plt.ylabel("Drawdown (%)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Drawdown plot saved to {path}")
        return path

    def plot_monthly_returns(self, results):
        eq = results.equity_curve
        if eq.empty:
            return None
        path = os.path.join(self.output_dir, "monthly_returns.png")
        monthly = eq.resample("ME").last().pct_change().dropna() * 100
        if monthly.empty:
            return None
        heatmap_df = monthly.to_frame("return")
        heatmap_df["year"] = heatmap_df.index.year
        heatmap_df["month"] = heatmap_df.index.strftime("%b")
        pivot = heatmap_df.pivot(index="year", columns="month", values="return")
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])
        plt.figure(figsize=(12, 4))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0)
        plt.title("Monthly Returns (%)")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Monthly returns plot saved to {path}")
        return path

    def plot_trade_distribution(self, results):
        trade_df = pd.DataFrame(results.trade_log)
        if trade_df.empty or "net_pnl" not in trade_df:
            return None
        path = os.path.join(self.output_dir, "trade_distribution.png")
        plt.figure(figsize=(10, 5))
        sns.histplot(trade_df["net_pnl"], bins=50, kde=True)
        plt.axvline(0, color="black", linestyle="--", linewidth=1)
        plt.title("Trade P&L Distribution")
        plt.xlabel("Net P&L")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Trade distribution plot saved to {path}")
        return path

    def plot_cumulative_pnl(self, results):
        trade_df = pd.DataFrame(results.trade_log)
        if trade_df.empty or "net_pnl" not in trade_df:
            return None
        path = os.path.join(self.output_dir, "cumulative_pnl.png")
        cumulative = trade_df["net_pnl"].cumsum()
        x = trade_df["close_time"] if "close_time" in trade_df else cumulative.index
        plt.figure(figsize=(12, 5))
        plt.plot(x, cumulative.values)
        plt.title("Cumulative Trade P&L")
        plt.xlabel("Time")
        plt.ylabel("Cumulative P&L")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Cumulative P&L plot saved to {path}")
        return path
