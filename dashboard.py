"""
Trading bot web dashboard.

This Flask app serves a compact control and monitoring UI for existing
symbol/timeframe artifacts produced by the training and backtest scripts.
"""

import ast
import os
import re
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.mt5_config import mt5_config
from config.settings import config, normalize_timeframe


app = Flask(__name__)

BASE_DIR = Path(config.paths.base_dir).resolve()
BACKTEST_ROOT = BASE_DIR / "backtest"
LOGS_ROOT = BASE_DIR / "logs"
SAVED_MODELS_ROOT = BASE_DIR / "saved_models"
DEFAULT_SYMBOL = "XAUUSD"
DEFAULT_TIMEFRAME = "M5"
SUPPORTED_TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D1")
SUPPORTED_STRATEGY_MODES = ("hybrid", "ml")
SUPPORTED_MODEL_KEYS = ("xgboost", "lightgbm", "catboost")
SUPPORTED_CREDENTIAL_FILES = ("auto", "credentials_xauusd.yaml", "credentials_ustec.yaml", "credentials.yaml")
MODEL_LABELS = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}

running_process = None
process_lock = threading.Lock()
process_log = []
process_type = None


def _safe_symbol(symbol):
    raw = (symbol or DEFAULT_SYMBOL).strip().upper()
    if not re.fullmatch(r"[A-Z0-9_.-]{1,40}", raw):
        raise ValueError("Invalid symbol.")
    return raw


def resolve_context(symbol=None, timeframe=None):
    return {
        "symbol": _safe_symbol(symbol or request.args.get("symbol")),
        "timeframe": normalize_timeframe(timeframe or request.args.get("timeframe") or DEFAULT_TIMEFRAME),
    }


def _inside_root(path, root):
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _has_files(path, names=None):
    if not path.exists() or not path.is_dir():
        return False
    if names:
        direct_match = any((path / name).exists() for name in names)
        child_match = any(
            (child / name).exists()
            for child in path.iterdir()
            if child.is_dir()
            for name in names
        )
        return direct_match or child_match
    return any(item.is_file() and item.name != ".gitkeep" for item in path.iterdir()) or any(
        child.is_dir() and any(item.is_file() and item.name != ".gitkeep" for item in child.iterdir())
        for child in path.iterdir()
    )


def context_artifact_dir(root, ctx):
    candidate = root / ctx["symbol"] / ctx["timeframe"]
    if not _inside_root(candidate, root):
        raise ValueError("Invalid artifact context.")
    return candidate


def artifact_dir(root, ctx, expected_files=None, strict_context=False):
    symbol_tf_dir = root / ctx["symbol"] / ctx["timeframe"]
    if strict_context:
        candidates = [
            symbol_tf_dir / "hybrid",
            symbol_tf_dir / "ml",
            symbol_tf_dir,
        ]
    else:
        candidates = [
            symbol_tf_dir / "hybrid",
            symbol_tf_dir / "ml",
            symbol_tf_dir,
            root / ctx["symbol"],
            root,
        ]
    for candidate in candidates:
        if _inside_root(candidate, root) and _has_files(candidate, expected_files):
            return candidate
    return symbol_tf_dir if strict_context else candidates[0]


def file_meta(path):
    if not path or not path.exists():
        return None
    stat = path.stat()
    return {
        "file": path.name,
        "path": str(path),
        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    }


def parse_float(value):
    if value is None:
        return None
    return float(str(value).replace("$", "").replace("%", "").replace(",", "").strip())


def parse_performance_report(report_path):
    if not report_path.exists():
        return None

    try:
        content = report_path.read_text(encoding="utf-8", errors="ignore")

        def search_val(pattern, is_float=True, is_int=False):
            match = re.search(pattern, content)
            if not match:
                return None
            val = match.group(1)
            if is_int:
                return int(val)
            if is_float:
                return parse_float(val)
            return val.strip()

        metrics = {
            "strategy_mode": search_val(r"Strategy Mode:\s+([A-Z]+)", is_float=False),
            "initial_balance": search_val(r"Initial Balance:\s+(\$?[\d.,]+)"),
            "net_profit": search_val(r"Net Profit:\s+(\$?[-\d.,]+)"),
            "ending_balance": search_val(r"Ending Balance:\s+(\$?[\d.,]+)"),
            "total_trades": search_val(r"Total Trades:\s+(\d+)", is_float=False, is_int=True),
            "profit_factor": search_val(r"Profit Factor:\s+([-\d.,]+)"),
            "expectancy": search_val(r"Trade Expectancy:\s+(\$?[-\d.,]+)"),
            "max_drawdown_cash": search_val(r"Max Drawdown \(\$\):\s+(\$?[\d.,]+)"),
            "max_drawdown_pct": search_val(r"Max Drawdown \(%\):\s+([\d.,]+)%"),
            "sharpe": search_val(r"Sharpe Ratio \(Daily\):\s+([-\d.,]+)"),
            "sortino": search_val(r"Sortino Ratio \(Daily\):\s+([-\d.,]+)"),
            "calmar": search_val(r"Calmar Ratio:\s+([-\d.,]+)"),
            "average_win": search_val(r"Average Win:\s+(\$?[-\d.,]+)"),
            "average_loss": search_val(r"Average Loss:\s+(\$?[-\d.,]+)"),
            "win_loss_ratio": search_val(r"Win / Loss Ratio:\s+([-\d.,]+)"),
        }

        win_match = re.search(r"Winning Trades:\s+(\d+)\s+\(([\d.,]+)%\)", content)
        loss_match = re.search(r"Losing Trades:\s+(\d+)\s+\(([\d.,]+)%\)", content)
        metrics["winning_trades"] = int(win_match.group(1)) if win_match else 0
        metrics["winrate"] = parse_float(win_match.group(2)) if win_match else 0.0
        metrics["losing_trades"] = int(loss_match.group(1)) if loss_match else 0
        metrics["lossrate"] = parse_float(loss_match.group(2)) if loss_match else 0.0

        tp_match = re.search(r"TP\s+(\d+)\s+\(([\d.,]+)%\)", content)
        sl_match = re.search(r"SL\s+(\d+)\s+\(([\d.,]+)%\)", content)
        hold_match = re.search(r"Average Hold Time:\s+([\d.,]+)\s+([A-Z0-9]+)\s+bars", content)
        metrics["exit_tp_count"] = int(tp_match.group(1)) if tp_match else 0
        metrics["exit_tp_pct"] = parse_float(tp_match.group(2)) if tp_match else 0.0
        metrics["exit_sl_count"] = int(sl_match.group(1)) if sl_match else 0
        metrics["exit_sl_pct"] = parse_float(sl_match.group(2)) if sl_match else 0.0
        metrics["avg_hold_time"] = parse_float(hold_match.group(1)) if hold_match else 0.0
        metrics["hold_time_unit"] = hold_match.group(2) if hold_match else DEFAULT_TIMEFRAME
        metrics["source"] = file_meta(report_path)
        metrics["available"] = True
        return metrics
    except Exception as exc:
        return {"available": False, "error": f"Error parsing report: {exc}"}


def parse_metric_list(line):
    try:
        return [float(v) for v in ast.literal_eval(line.split(":", 1)[1].strip())]
    except Exception:
        return [0.0, 0.0, 0.0]


def parse_model_comparison(report_path):
    if not report_path.exists():
        return None

    models = []
    recommended = "Unknown"
    current = None
    model_data = {}

    try:
        for raw_line in report_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if "Recommended Model for Deployment:" in line:
                recommended = line.split(":", 1)[1].strip()
            elif line.startswith("Model:"):
                if current and model_data:
                    models.append(model_data)
                current = line.split(":", 1)[1].strip()
                model_data = {"name": current}
            elif current:
                if "Accuracy:" in line:
                    model_data["accuracy"] = parse_float(line.split(":", 1)[1])
                elif "Trade Signal Precision:" in line:
                    model_data["trade_precision"] = parse_float(line.split(":", 1)[1])
                elif "Precision per Class" in line:
                    vals = parse_metric_list(line)
                    model_data.update({
                        "precision_notrade": vals[0],
                        "precision_buy": vals[1],
                        "precision_sell": vals[2],
                    })
                elif "Recall per Class" in line:
                    vals = parse_metric_list(line)
                    model_data.update({
                        "recall_notrade": vals[0],
                        "recall_buy": vals[1],
                        "recall_sell": vals[2],
                    })
                elif "F1-Score per Class" in line:
                    vals = parse_metric_list(line)
                    model_data.update({
                        "f1_notrade": vals[0],
                        "f1_buy": vals[1],
                        "f1_sell": vals[2],
                    })

        if current and model_data:
            models.append(model_data)

        return {
            "available": bool(models),
            "recommended": recommended,
            "models": models,
            "source": file_meta(report_path),
        }
    except Exception as exc:
        return {"available": False, "recommended": "Unknown", "models": [], "error": str(exc)}


def parse_timeframe_model_comparison(report_path):
    if not report_path.exists():
        return None

    models = []
    recommended = "Unknown"

    try:
        for raw_line in report_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if line.startswith("Winner:") and line.upper() != "Winner: NONE":
                winner_parts = line.split(":", 1)[1].strip().split()
                if len(winner_parts) >= 3:
                    recommended = f"{winner_parts[1]} {winner_parts[2]}"
            elif re.match(r"^[A-Z0-9_.-]+\s+[A-Z0-9]+\s+\S+\s+(True|False)\s+\d+\s+", line):
                match = re.match(
                    r"^(?P<symbol>\S+)\s+"
                    r"(?P<timeframe>\S+)\s+"
                    r"(?P<model>\S+)\s+"
                    r"(?P<valid>True|False)\s+"
                    r"(?P<trades>\d+)\s+"
                    r"(?P<profit_factor>[-\d.]+)\s+"
                    r"(?P<drawdown_pct>[-\d.]+)\s+"
                    r"(?P<net_profit>[-\d.]+)\s+"
                    r"(?P<winrate>[-\d.]+)\s+"
                    r"(?P<trade_precision>[-\d.]+)\s+"
                    r"(?P<f1_buy>[-\d.]+)\s+"
                    r"(?P<f1_sell>[-\d.]+)\s+"
                    r"(?P<reason>.*)$",
                    line,
                )
                if not match:
                    continue
                item = match.groupdict()
                models.append({
                    "name": f"{item['timeframe']} {item['model']}",
                    "symbol": item["symbol"],
                    "timeframe": item["timeframe"],
                    "model": item["model"],
                    "valid": item["valid"] == "True",
                    "trades": int(item["trades"]),
                    "profit_factor": parse_float(item["profit_factor"]),
                    "drawdown_pct": parse_float(item["drawdown_pct"]),
                    "net_profit": parse_float(item["net_profit"]),
                    "winrate": parse_float(item["winrate"]) / 100.0,
                    "trade_precision": parse_float(item["trade_precision"]),
                    "f1_buy": parse_float(item["f1_buy"]),
                    "f1_sell": parse_float(item["f1_sell"]),
                    "reason": item["reason"],
                    "report_type": "timeframe_model_comparison",
                })

        ranked = sorted(
            models,
            key=lambda item: (
                item["valid"],
                item["net_profit"],
                item["profit_factor"],
                -item["drawdown_pct"],
                item["winrate"],
                item["trades"],
            ),
            reverse=True,
        )
        if ranked:
            recommended = ranked[0]["name"]
            models = ranked

        return {
            "available": bool(models),
            "recommended": recommended,
            "models": models,
            "source": file_meta(report_path),
            "report_type": "timeframe_model_comparison",
        }
    except Exception as exc:
        return {"available": False, "recommended": "Unknown", "models": [], "error": str(exc)}


def latest_log_file(log_dir, prefix):
    if not log_dir.exists():
        return None
    matches = sorted(
        [path for path in log_dir.iterdir() if path.is_file() and path.name.startswith(prefix) and path.suffix == ".log"],
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    return matches[0] if matches else None


def read_log_tail(file_path, num_lines=150):
    if not file_path or not file_path.exists():
        return ["Log file not found for selected context."]
    try:
        return [line.rstrip() for line in file_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-num_lines:]]
    except Exception as exc:
        return [f"Error reading log file: {exc}"]


def discover_contexts():
    contexts = {}

    def ensure(symbol, timeframe):
        key = (symbol, timeframe)
        contexts.setdefault(key, {
            "symbol": symbol,
            "timeframe": timeframe,
            "label": f"{symbol} / {timeframe}",
            "has_backtest": False,
            "has_logs": False,
            "has_models": False,
            "last_modified": None,
        })
        return contexts[key]

    def touch(item, path, attr):
        item[attr] = True
        newest = max((p.stat().st_mtime for p in path.iterdir() if p.is_file()), default=None)
        if newest and (item["last_modified"] is None or newest > item["last_modified"]):
            item["last_modified"] = newest

    for root, attr in ((BACKTEST_ROOT, "has_backtest"), (LOGS_ROOT, "has_logs"), (SAVED_MODELS_ROOT, "has_models")):
        if not root.exists():
            continue
        for symbol_dir in [p for p in root.iterdir() if p.is_dir() and not p.name.startswith("__")]:
            direct_files = [p for p in symbol_dir.iterdir() if p.is_file() and p.name != ".gitkeep"]
            if direct_files:
                touch(ensure(symbol_dir.name.upper(), DEFAULT_TIMEFRAME), symbol_dir, attr)
            for tf_dir in [p for p in symbol_dir.iterdir() if p.is_dir()]:
                try:
                    timeframe = normalize_timeframe(tf_dir.name)
                except ValueError:
                    continue
                if _has_files(tf_dir):
                    touch(ensure(symbol_dir.name.upper(), timeframe), tf_dir, attr)

    if not contexts:
        ensure(DEFAULT_SYMBOL, DEFAULT_TIMEFRAME)

    result = []
    for item in contexts.values():
        if item["last_modified"]:
            item["last_modified"] = datetime.fromtimestamp(item["last_modified"]).strftime("%Y-%m-%d %H:%M:%S")
        result.append(item)
    return sorted(result, key=lambda x: (x["symbol"], SUPPORTED_TIMEFRAMES.index(x["timeframe"]) if x["timeframe"] in SUPPORTED_TIMEFRAMES else 99))


def resolve_model_path(model_value, ctx):
    if not model_value:
        return None
    raw = Path(model_value)
    if raw.is_absolute():
        candidate = raw.resolve()
    else:
        candidate = (SAVED_MODELS_ROOT / raw).resolve()
    if not _inside_root(candidate, SAVED_MODELS_ROOT) or not candidate.exists():
        candidate = (context_artifact_dir(SAVED_MODELS_ROOT, ctx) / raw.name).resolve()
    if not _inside_root(candidate, SAVED_MODELS_ROOT):
        return None
    expected_dir = context_artifact_dir(SAVED_MODELS_ROOT, ctx).resolve()
    try:
        candidate.relative_to(expected_dir)
    except ValueError:
        raise ValueError(
            f"Selected model must be inside saved_models/{ctx['symbol']}/{ctx['timeframe']} "
            f"for the selected timeframe."
        )
    return candidate


def model_artifact_family(filename):
    name = filename.lower()
    if "xgboost" in name or "xgb" in name:
        return "xgboost"
    if "lightgbm" in name or "lgb" in name:
        return "lightgbm"
    if "catboost" in name or "cat" in name:
        return "catboost"
    return None


def model_artifact_variant(filename):
    name = filename.lower()
    if name.startswith("selected_"):
        return "selected"
    if name.startswith("candidate_"):
        return "candidate"
    return "model"


def latest_path(current, candidate):
    if current is None:
        return candidate
    if candidate.stat().st_mtime >= current.stat().st_mtime:
        return candidate
    return current


def choose_model_artifact(variants):
    selected = variants.get("selected")
    if selected:
        return selected
    candidate = variants.get("candidate")
    if candidate:
        return candidate
    return max(variants.values(), key=lambda path: path.stat().st_mtime, default=None)


def model_artifact_payload(path, family):
    stat = path.stat()
    variant = model_artifact_variant(path.name)
    label_family = MODEL_LABELS.get(family, family.title())
    label = f"{label_family} {variant} ({stat.st_size / (1024 * 1024):.2f} MB)"
    try:
        relative = str(path.resolve().relative_to(SAVED_MODELS_ROOT))
    except ValueError:
        relative = path.name
    return {
        "name": path.name,
        "label": label,
        "value": relative.replace("\\", "/"),
        "size": f"{stat.st_size / (1024 * 1024):.2f} MB",
        "created": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "modified_ts": stat.st_mtime,
        "variant": variant,
        "model_family": family,
    }


def comparison_rank_by_net_profit(ctx):
    report_path = BACKTEST_ROOT / ctx["symbol"] / "timeframe_model_comparison_report.txt"
    parsed = parse_timeframe_model_comparison(report_path)
    if not parsed or not parsed.get("models"):
        return {}

    ranked = {}
    for index, item in enumerate(parsed["models"]):
        if item.get("symbol") != ctx["symbol"] or item.get("timeframe") != ctx["timeframe"]:
            continue
        family = model_artifact_family(str(item.get("model", "")))
        if not family:
            continue
        ranked[family] = {
            "rank": index,
            "net_profit": item.get("net_profit", 0),
            "valid": item.get("valid", False),
        }
    return ranked


def get_dashboard_python():
    python_exe = BASE_DIR / ".venv" / "Scripts" / "python.exe"
    if not python_exe.exists():
        raise FileNotFoundError(
            f"Required Python environment not found: {python_exe}. Create and install dependencies in .venv."
        )
    return str(python_exe)


def parse_bool(value):
    return value is True or str(value).lower() in {"1", "true", "yes", "on"}


def parse_optional_float(value):
    if value in (None, ""):
        return None
    return float(value)


def selected_model_keys(data):
    raw_models = data.get("models") or []
    if isinstance(raw_models, str):
        raw_models = [raw_models]
    models = [str(model).strip().lower() for model in raw_models]
    models = [model for model in models if model in SUPPORTED_MODEL_KEYS]
    return models or list(SUPPORTED_MODEL_KEYS)


def selected_strategy_mode(data):
    mode = str(data.get("strategy_mode") or "hybrid").strip().lower()
    if mode not in SUPPORTED_STRATEGY_MODES:
        raise ValueError("Invalid strategy mode.")
    return mode


def resolve_credentials_file(value, symbol):
    credential = str(value or "auto").strip()
    if credential not in SUPPORTED_CREDENTIAL_FILES:
        raise ValueError("Invalid credentials selection.")
    if credential != "auto":
        return credential
    symbol_lower = symbol.lower()
    if "ustec" in symbol_lower:
        return "credentials_ustec.yaml"
    if "xauusd" in symbol_lower:
        return "credentials_xauusd.yaml"
    return "credentials.yaml"


def append_training_options(cmd, data, *, compare_mode=False):
    models = selected_model_keys(data)
    cmd.extend(["--models", *models])
    if parse_bool(data.get("tune")):
        cmd.append("--tune")
        tune_trials = data.get("tune_trials")
        if tune_trials not in (None, ""):
            cmd.extend(["--tune-trials", str(int(tune_trials))])
    if parse_bool(data.get("walk_forward")):
        cmd.append("--walk-forward")
    if parse_bool(data.get("class_balance")):
        cmd.append("--class-balance" if compare_mode else "--balance")
    if parse_bool(data.get("gpu")):
        cmd.append("--gpu")


def build_control_command(data):
    action = data.get("action")
    months = int(data.get("months", 6))
    ctx = resolve_context(data.get("symbol"), data.get("timeframe", DEFAULT_TIMEFRAME))
    python_bin = get_dashboard_python()
    strategy_mode = selected_strategy_mode(data)

    if action == "train":
        cmd = [
            python_bin, "main.py", "train",
            "--months", str(months),
            "--symbol", ctx["symbol"],
            "--timeframe", ctx["timeframe"],
        ]
        append_training_options(cmd, data)
    elif action == "compare":
        cmd = [
            python_bin, "main.py", "compare",
            "--months", str(months),
            "--symbol", ctx["symbol"],
            "--timeframes", ctx["timeframe"],
        ]
        append_training_options(cmd, data, compare_mode=True)
        balance = parse_optional_float(data.get("balance"))
        max_spread = parse_optional_float(data.get("max_spread"))
        if balance is not None:
            cmd.extend(["--balance", str(balance)])
        if max_spread is not None:
            cmd.extend(["--max-spread", str(max_spread)])
        cmd.extend(["--strategy-mode", strategy_mode])
    elif action == "backtest":
        model_path = resolve_model_path(data.get("model"), ctx)
        if not model_path:
            raise ValueError("Model path must be specified and exist.")
        cmd = [
            python_bin, "main.py", "backtest",
            "--model", str(model_path),
            "--months", str(months),
            "--symbol", ctx["symbol"],
            "--timeframe", ctx["timeframe"],
        ]
        balance = parse_optional_float(data.get("balance"))
        max_spread = parse_optional_float(data.get("max_spread"))
        if balance is not None:
            cmd.extend(["--balance", str(balance)])
        if max_spread is not None:
            cmd.extend(["--max-spread", str(max_spread)])
        cmd.extend(["--strategy-mode", strategy_mode])
    elif action in {"paper", "live"}:
        model_path = resolve_model_path(data.get("model"), ctx)
        if not model_path:
            raise ValueError("Model path must be specified and exist.")
        cred_file = resolve_credentials_file(data.get("credentials"), ctx["symbol"])
        cmd = [
            python_bin, "main.py", action,
            "--model", str(model_path),
            "--config", cred_file,
            "--symbol", ctx["symbol"],
            "--timeframe", ctx["timeframe"],
            "--strategy-mode", strategy_mode,
        ]
        if action == "live":
            cmd.append("--live")
    else:
        raise ValueError("Invalid control action.")

    return cmd, ctx


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/contexts")
def get_contexts():
    ctx = resolve_context()
    return jsonify({"default": ctx, "contexts": discover_contexts()})


@app.route("/api/summary")
def get_summary():
    ctx = resolve_context()
    backtest_dir = artifact_dir(BACKTEST_ROOT, ctx, ["performance_report.txt"], strict_context=True)
    report_path = backtest_dir / "performance_report.txt"
    stats = parse_performance_report(report_path)
    if not stats:
        stats = {"available": False, "message": "No performance report found for selected context."}
    stats["context"] = ctx
    return jsonify(stats)


@app.route("/api/models")
def get_models():
    ctx = resolve_context()
    backtest_dir = artifact_dir(BACKTEST_ROOT, ctx, ["model_comparison_report.txt"], strict_context=True)
    model_report_path = backtest_dir / "model_comparison_report.txt"
    compare_report_path = BACKTEST_ROOT / ctx["symbol"] / "timeframe_model_comparison_report.txt"

    report_path = model_report_path
    parser = parse_model_comparison
    if compare_report_path.exists():
        if not model_report_path.exists() or compare_report_path.stat().st_mtime >= model_report_path.stat().st_mtime:
            report_path = compare_report_path
            parser = parse_timeframe_model_comparison

    data = parser(report_path)
    if not data:
        data = {"available": False, "recommended": "Unknown", "models": [], "message": "No model comparison report found."}
    elif data.get("report_type") == "timeframe_model_comparison":
        models = [
            item for item in data.get("models", [])
            if item.get("symbol") == ctx["symbol"] and item.get("timeframe") == ctx["timeframe"]
        ]
        data["models"] = models
        data["available"] = bool(models)
        data["recommended"] = models[0]["name"] if models else "Unknown"
        if not models:
            data["message"] = f"No model comparison rows for {ctx['symbol']}/{ctx['timeframe']}."
    data["context"] = ctx
    return jsonify(data)


@app.route("/api/config")
def get_config():
    ctx = resolve_context()
    cfg = {
        "context": ctx,
        "symbol": {
            "symbol": ctx["symbol"],
            "timeframe": ctx["timeframe"],
            "configured_symbol": config.symbol.symbol,
            "configured_timeframe": config.symbol.timeframe,
            "point": config.symbol.point,
            "contract_size": config.symbol.contract_size,
            "min_lot": config.symbol.min_lot,
            "max_lot": config.symbol.max_lot,
            "lot_step": config.symbol.lot_step,
        },
        "data": {
            "training_months": config.data.training_months,
            "min_bars": config.data.min_bars,
            "train_ratio": config.data.train_ratio,
            "val_ratio": config.data.val_ratio,
            "test_ratio": config.data.test_ratio,
            "reward_risk_ratio": config.data.reward_risk_ratio,
            "atr_sl_multiplier": config.data.atr_sl_multiplier,
        },
        "features": {
            "ema_fast": config.features.ema_fast,
            "ema_medium": config.features.ema_medium,
            "ema_slow": config.features.ema_slow,
            "rsi_period": config.features.rsi_period,
            "atr_period": config.features.atr_period,
            "macd_fast": config.features.macd_fast,
            "macd_slow": config.features.macd_slow,
            "bb_period": config.features.bb_period,
            "sr_lookback": config.features.sr_lookback,
        },
        "risk": {
            "max_risk_per_trade": f"{config.risk.max_risk_per_trade * 100:.1f}%",
            "max_daily_drawdown": f"{config.risk.max_daily_drawdown * 100:.1f}%",
            "max_consecutive_losses": config.risk.max_consecutive_losses,
            "atr_sl_multiplier": config.risk.atr_sl_multiplier,
            "atr_tp_multiplier": config.risk.atr_tp_multiplier,
            "trailing_stop_activation": f"{config.risk.trailing_stop_activation * 100:.0f}%",
            "trailing_stop_atr": config.risk.trailing_stop_atr,
            "max_spread_points": config.risk.max_spread,
            "max_open_positions": config.risk.max_open_positions,
        },
        "filters": {
            "spread_filter": "Enabled" if config.filters.spread_filter_enabled else "Disabled",
            "volatility_filter": "Enabled" if config.filters.volatility_filter_enabled else "Disabled",
            "news_filter": "Enabled" if config.filters.news_filter_enabled else "Disabled",
            "session_filter": "Enabled" if config.filters.session_filter_enabled else "Disabled",
            "ranging_filter": "Enabled" if config.filters.ranging_filter_enabled else "Disabled",
            "allowed_sessions": ", ".join(config.filters.allowed_sessions),
        },
        "mt5": {
            "login": mt5_config.login,
            "server": mt5_config.server,
            "paper_trading": "True (Simulated)" if mt5_config.paper_trading else "False (Real Account)",
            "trading_enabled": "True" if mt5_config.trading_enabled else "False",
        },
        "artifact_dirs": {
            "backtest": str(artifact_dir(BACKTEST_ROOT, ctx, strict_context=True)),
            "logs": str(context_artifact_dir(LOGS_ROOT, ctx)),
            "saved_models": str(context_artifact_dir(SAVED_MODELS_ROOT, ctx)),
        },
    }
    return jsonify(cfg)


@app.route("/api/logs/<log_type>")
def get_logs(log_type):
    if log_type == "console":
        with process_lock:
            return jsonify({"lines": process_log[-150:]})

    ctx = resolve_context()
    prefix_map = {"main": "main_", "signals": "signals_", "trades": "trades_", "errors": "errors_"}
    prefix = prefix_map.get(log_type)
    if not prefix:
        return jsonify({"lines": ["Invalid log type request."]}), 400

    log_dir = context_artifact_dir(LOGS_ROOT, ctx)
    latest_file = latest_log_file(log_dir, prefix)
    return jsonify({
        "context": ctx,
        "lines": read_log_tail(latest_file),
        "source": file_meta(latest_file) if latest_file else None,
        "source_dir": str(log_dir),
    })


@app.route("/api/saved-models")
def get_saved_models():
    ctx = resolve_context()
    models_dir = context_artifact_dir(SAVED_MODELS_ROOT, ctx)
    models = []
    if models_dir.exists():
        grouped = {family: {} for family in MODEL_LABELS}
        for path in models_dir.iterdir():
            if path.name == ".gitkeep" or not path.is_file():
                continue
            family = model_artifact_family(path.name)
            if family not in grouped:
                continue
            variant = model_artifact_variant(path.name)
            grouped[family][variant] = latest_path(grouped[family].get(variant), path)

        for family in sorted(grouped, key=lambda key: MODEL_LABELS[key]):
            path = choose_model_artifact(grouped[family])
            if path:
                models.append(model_artifact_payload(path, family))

        comparison_rank = comparison_rank_by_net_profit(ctx)
        for model in models:
            rank_meta = comparison_rank.get(model["model_family"])
            if rank_meta:
                model["comparison_rank"] = rank_meta["rank"] + 1
                model["net_profit"] = rank_meta["net_profit"]
                model["valid_backtest"] = rank_meta["valid"]
        if comparison_rank:
            models.sort(
                key=lambda item: (
                    0 if item.get("variant") == "selected" else 1,
                    -item.get("modified_ts", 0),
                    comparison_rank.get(item["model_family"], {}).get("rank", 999),
                    item["label"],
                )
            )
        else:
            models.sort(key=lambda item: (0 if item.get("variant") == "selected" else 1, -item.get("modified_ts", 0), item["label"]))
    message = None if models else f"No saved models for {ctx['symbol']}/{ctx['timeframe']}."
    return jsonify({"context": ctx, "models": models, "source_dir": str(models_dir), "message": message})


@app.route("/api/status")
def get_status():
    global running_process, process_type
    with process_lock:
        is_active = running_process is not None and running_process.poll() is None
    return jsonify({
        "status": "Running Background Task" if is_active else "Idle",
        "active_task": process_type if is_active else None,
        "symbol": config.symbol.symbol,
        "timeframe": config.symbol.timeframe,
        "mt5_status": "Connected" if mt5_config.login > 0 else "Disconnected",
        "paper_trading": mt5_config.paper_trading,
        "live_trading": mt5_config.trading_enabled and not mt5_config.paper_trading,
    })


def run_process_thread(cmd, p_type):
    global running_process, process_log, process_type
    process_type = p_type
    process_log.clear()
    process_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Launching: {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(BASE_DIR),
        )
        with process_lock:
            running_process = proc
        for line in proc.stdout:
            with process_lock:
                if len(process_log) > 1000:
                    process_log.pop(0)
                process_log.append(line.rstrip())
        proc.wait()
        process_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Process finished with exit code {proc.returncode}")
    except Exception as exc:
        process_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Failed to run command: {exc}")
    finally:
        with process_lock:
            running_process = None


@app.route("/api/control/start", methods=["POST"])
def start_bot_action():
    global running_process, process_type
    data = request.json or {}

    with process_lock:
        if running_process and running_process.poll() is None:
            return jsonify({"status": "error", "message": f"Another background task ({process_type}) is already running."}), 400

    try:
        cmd, _ctx = build_control_command(data)
    except FileNotFoundError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500
    except (TypeError, ValueError) as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400

    action = data.get("action")
    thread = threading.Thread(target=run_process_thread, args=(cmd, action), daemon=True)
    thread.start()
    return jsonify({"status": "success", "message": f"Background process '{action}' started.", "command": cmd})


@app.route("/api/control/preview", methods=["POST"])
def preview_bot_action():
    data = request.json or {}
    try:
        cmd, ctx = build_control_command(data)
    except FileNotFoundError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500
    except (TypeError, ValueError) as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    return jsonify({"status": "success", "command": cmd, "context": ctx})


@app.route("/api/control/stop", methods=["POST"])
def stop_bot_action():
    global running_process, process_type
    with process_lock:
        if not running_process or running_process.poll() is not None:
            return jsonify({"status": "error", "message": "No active background tasks to stop."}), 400
        try:
            if sys.platform == "win32":
                subprocess.Popen(["taskkill", "/F", "/T", "/PID", str(running_process.pid)])
            else:
                running_process.terminate()
            process_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Task halted manually by user request.")
            return jsonify({"status": "success", "message": f"Successfully stopped active '{process_type}' process."})
        except Exception as exc:
            return jsonify({"status": "error", "message": f"Failed to stop process: {exc}"}), 500


@app.route("/charts/<symbol>/<timeframe>/<filename>")
def get_context_chart(symbol, timeframe, filename):
    try:
        ctx = resolve_context(symbol, timeframe)
    except ValueError:
        return "Invalid context", 400
    safe_name = os.path.basename(filename)
    if safe_name != filename or not safe_name.lower().endswith(".png"):
        return "Invalid file", 400
    chart_dir = artifact_dir(BACKTEST_ROOT, ctx, [safe_name], strict_context=True)
    chart_path = chart_dir / safe_name
    if not chart_path.exists():
        return "Chart not found", 404
    return send_from_directory(chart_dir, safe_name)


@app.route("/charts/<filename>")
def get_legacy_chart_image(filename):
    ctx = {"symbol": DEFAULT_SYMBOL, "timeframe": DEFAULT_TIMEFRAME}
    return get_context_chart(ctx["symbol"], ctx["timeframe"], filename)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  ML TRADING ENGINE - WEB DASHBOARD")
    print("  Access local panel at: http://127.0.0.1:5000/")
    print("=" * 70 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
