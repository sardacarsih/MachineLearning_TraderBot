"""
============================================
Timeframe Training and Backtest Comparator
============================================
Runs training and backtesting for multiple timeframes, then selects the best
symbol/timeframe/model combination using backtest quality.
"""

import argparse
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Adjust path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import config, normalize_timeframe

SUPPORTED_TRAINING_MODELS = {"xgboost", "lightgbm", "catboost"}


@dataclass
class ComparisonResult:
    symbol: str
    timeframe: str
    model_name: str
    model_path: str
    total_trades: int
    profit_factor: float
    max_drawdown_pct: float
    net_profit: float
    winrate: float
    trade_precision: float
    f1_buy: float
    f1_sell: float
    valid: bool
    reason: str


def _run_command(args: List[str], cwd: Path):
    print("\n" + "=" * 78)
    print("Running: " + " ".join(args))
    print("=" * 78)
    subprocess.run(args, cwd=str(cwd), check=True)


def _safe_float(value: str) -> float:
    cleaned = value.replace("$", "").replace(",", "").replace("%", "").strip()
    return float(cleaned)


def _extract(pattern: str, text: str, default: str = "0") -> str:
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else default


def _parse_model_report(path: Path, model_name: str) -> tuple[float, float, float]:
    if not path.exists():
        return 0.0, 0.0, 0.0

    text = path.read_text(encoding="utf-8", errors="ignore")
    section_match = re.search(
        rf"Model:\s*{re.escape(model_name)}(?P<section>.*?)(?:\n=+\n|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    section = section_match.group("section") if section_match else text

    trade_precision = _safe_float(_extract(r"Trade Signal Precision:\s*([\d.]+)", section))
    f1_values = re.search(r"F1-Score per Class \(0,1,2\):\s*\[(.*?)\]", section, re.IGNORECASE)
    if not f1_values:
        return trade_precision, 0.0, 0.0

    values = [
        _safe_float(v.strip().strip("'\""))
        for v in f1_values.group(1).split(",")
        if v.strip()
    ]
    f1_buy = values[1] if len(values) > 1 else 0.0
    f1_sell = values[2] if len(values) > 2 else 0.0
    return trade_precision, f1_buy, f1_sell


def _parse_performance_report(path: Path) -> tuple[int, float, float, float, float]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    total_trades = int(_safe_float(_extract(r"Total Trades:\s*([\d,]+)", text)))
    profit_factor = _safe_float(_extract(r"Profit Factor:\s*([\d.,]+)", text))
    max_drawdown_pct = _safe_float(_extract(r"Max Drawdown \(%\):\s*([\d.,]+)%", text)) / 100.0
    net_profit = _safe_float(_extract(r"Net Profit:\s*([$\-\d.,]+)", text))
    winrate = _safe_float(_extract(r"Winning Trades:\s*\d+\s*\(([\d.,]+)%\)", text)) / 100.0
    return total_trades, profit_factor, max_drawdown_pct, net_profit, winrate


def _model_key(model_name: str) -> str:
    name = model_name.strip().lower().replace("-", "_")
    aliases = {
        "xgb": "xgboost",
        "lgb": "lightgbm",
        "lgbm": "lightgbm",
        "rf": "random_forest",
        "randomforest": "random_forest",
        "random_forest": "randomforest",
        "cat": "catboost",
    }
    return aliases.get(name, name)


def _find_model_candidates(saved_models_dir: Path, requested_models: List[str]) -> List[Path]:
    candidates = []
    for requested in requested_models:
        key = _model_key(requested)
        patterns = [
            f"candidate_{key}_model",
            f"selected_{key}_model",
        ]
        for pattern in patterns:
            path = saved_models_dir / pattern
            if path.exists():
                candidates.append(path)
                break

    if not candidates:
        raise FileNotFoundError(f"No trained model candidates found in {saved_models_dir}")
    return candidates


def _model_name_from_path(model_path: Path) -> str:
    name = model_path.name.lower()
    if "xgboost" in name or "xgb" in name:
        return "XGBoost"
    if "lightgbm" in name or "lgb" in name:
        return "LightGBM"
    if "random" in name or "rf" in name:
        return "RandomForest"
    if "catboost" in name or "cat" in name:
        return "CatBoost"
    return model_path.name


def _result_is_valid(
    total_trades: int,
    profit_factor: float,
    max_drawdown_pct: float,
    winrate: float,
) -> tuple[bool, str]:
    failures = []
    if total_trades < config.backtest.min_trades:
        failures.append(f"trades {total_trades} < minimum {config.backtest.min_trades}")
    if not math.isfinite(profit_factor) or profit_factor <= 0:
        failures.append("profit factor is invalid")
    elif profit_factor < config.backtest.min_profit_factor:
        failures.append(f"profit factor {profit_factor:.2f} < minimum {config.backtest.min_profit_factor:.2f}")
    if not math.isfinite(max_drawdown_pct) or max_drawdown_pct < 0 or max_drawdown_pct > 1:
        failures.append("drawdown is invalid")
    elif max_drawdown_pct > config.backtest.max_drawdown_pct:
        failures.append(
            f"drawdown {max_drawdown_pct * 100:.2f}% > maximum {config.backtest.max_drawdown_pct * 100:.2f}%"
        )
    if winrate < config.backtest.min_winrate:
        failures.append(f"winrate {winrate * 100:.2f}% < minimum {config.backtest.min_winrate * 100:.2f}%")
    if failures:
        return False, "; ".join(failures)
    return True, "valid"


def _rank_key(result: ComparisonResult):
    return (
        result.valid,
        result.net_profit,
        result.profit_factor,
        -result.max_drawdown_pct,
        result.winrate,
        result.total_trades,
    )


def _write_report(results: List[ComparisonResult], output_path: Path):
    ranked = sorted(results, key=_rank_key, reverse=True)
    winner: Optional[ComparisonResult] = ranked[0] if ranked else None

    report_symbol = results[0].symbol if results else "Symbol"
    lines = [
        "=" * 78,
        f"{report_symbol} Timeframe + Model Comparison Report",
        "=" * 78,
        "",
    ]

    if winner:
        if winner.valid:
            lines.extend([
                f"Winner: {winner.symbol} {winner.timeframe} {winner.model_name}",
                f"Model Path: {winner.model_path}",
                "Reason: ranked by valid backtest, net profit, profit factor, drawdown, winrate, then trade count.",
                "",
            ])
        else:
            lines.extend([
                "Winner: NONE",
                "Reason: no model/timeframe met the validity rules.",
                f"Best Candidate: {winner.symbol} {winner.timeframe} {winner.model_name}",
                f"Candidate Model Path: {winner.model_path}",
                "",
            ])

    lines.append("Summary:")
    header = (
        "Symbol Timeframe Model        Valid Trades ProfitFactor Drawdown% "
        "NetProfit Winrate% TradePrecision F1_BUY F1_SELL Reason"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for result in ranked:
        lines.append(
            f"{result.symbol:<6} {result.timeframe:<9} {result.model_name:<12} "
            f"{str(result.valid):<5} {result.total_trades:<6} "
            f"{result.profit_factor:<12.2f} {result.max_drawdown_pct * 100:<9.2f} "
            f"{result.net_profit:<9.2f} {result.winrate * 100:<8.2f} "
            f"{result.trade_precision:<14.4f} {result.f1_buy:<6.4f} "
            f"{result.f1_sell:<7.4f} {result.reason}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nComparison report saved to: {output_path}")
    if winner:
        if winner.valid:
            print(f"Winner: {winner.symbol} {winner.timeframe} {winner.model_name}")
        else:
            print(f"Winner: NONE; best candidate is {winner.symbol} {winner.timeframe} {winner.model_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Train/backtest selected timeframe(s) and select the best model/timeframe."
    )
    parser.add_argument("--symbol", type=str, default="XAUUSD", help="Trading symbol")
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=[config.symbol.timeframe],
        help="Timeframes to compare. Defaults to the configured timeframe only.",
    )
    parser.add_argument("--months", type=int, default=config.data.training_months, help="Months of data")
    parser.add_argument("--models", nargs="+", default=config.model.models_to_train, help="Models to train")
    parser.add_argument("--tune", action="store_true", help="Enable Optuna tuning during training")
    parser.add_argument("--tune-trials", type=int, default=None, help="Optuna trials per tuned model")
    parser.add_argument("--walk-forward", action="store_true", help="Enable walk-forward validation")
    parser.add_argument("--class-balance", action="store_true", help="Balance classes during training")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration for supported models")
    parser.add_argument("--balance", type=float, default=config.backtest.initial_balance, help="Backtest balance")
    parser.add_argument("--max-spread", type=float, default=None,
                        help="Maximum allowed spread in broker points for each backtest")
    parser.add_argument("--strategy-mode", choices=["ml", "hybrid"], default="ml",
                        help="Backtest entry validation mode used for ranking.")
    args = parser.parse_args()
    requested_models = [str(model).strip().lower() for model in args.models]
    ignored_models = [model for model in requested_models if model not in SUPPORTED_TRAINING_MODELS]
    selected_models = [model for model in requested_models if model in SUPPORTED_TRAINING_MODELS]
    if ignored_models:
        print(f"Ignoring unsupported training models: {ignored_models}")
    if not selected_models:
        raise ValueError(f"No supported models selected. Available: {sorted(SUPPORTED_TRAINING_MODELS)}")

    root = Path(__file__).resolve().parents[1]
    python = sys.executable
    results: List[ComparisonResult] = []

    for timeframe in [normalize_timeframe(tf) for tf in args.timeframes]:
        train_cmd = [
            python, "main.py", "train",
            "--symbol", args.symbol,
            "--timeframe", timeframe,
            "--months", str(args.months),
            "--models", *selected_models,
        ]
        if args.tune:
            train_cmd.append("--tune")
        if args.tune_trials is not None:
            train_cmd.extend(["--tune-trials", str(args.tune_trials)])
        if args.walk_forward:
            train_cmd.append("--walk-forward")
        if args.class_balance:
            train_cmd.append("--balance")
        if args.gpu:
            train_cmd.append("--gpu")

        _run_command(train_cmd, root)

        saved_models_dir = root / "saved_models" / args.symbol.upper() / timeframe
        model_paths = _find_model_candidates(saved_models_dir, selected_models)
        backtest_dir = root / "backtest" / args.symbol.upper() / timeframe
        performance_dir = backtest_dir / args.strategy_mode

        for model_path in model_paths:
            backtest_cmd = [
                python, "main.py", "backtest",
                "--symbol", args.symbol,
                "--timeframe", timeframe,
                "--model", str(model_path),
                "--months", str(args.months),
                "--balance", str(args.balance),
                "--strategy-mode", args.strategy_mode,
            ]
            if args.max_spread is not None:
                backtest_cmd.extend(["--max-spread", str(args.max_spread)])
            _run_command(backtest_cmd, root)

            model_name = _model_name_from_path(model_path)
            total_trades, profit_factor, max_dd, net_profit, winrate = _parse_performance_report(
                performance_dir / "performance_report.txt"
            )
            trade_precision, f1_buy, f1_sell = _parse_model_report(
                backtest_dir / "model_comparison_report.txt", model_name
            )
            valid, reason = _result_is_valid(total_trades, profit_factor, max_dd, winrate)

            results.append(ComparisonResult(
                symbol=args.symbol.upper(),
                timeframe=timeframe,
                model_name=model_name,
                model_path=str(model_path),
                total_trades=total_trades,
                profit_factor=profit_factor,
                max_drawdown_pct=max_dd,
                net_profit=net_profit,
                winrate=winrate,
                trade_precision=trade_precision,
                f1_buy=f1_buy,
                f1_sell=f1_sell,
                valid=valid,
                reason=reason,
            ))

    output_path = root / "backtest" / args.symbol.upper() / "timeframe_model_comparison_report.txt"
    _write_report(results, output_path)


if __name__ == "__main__":
    main()
