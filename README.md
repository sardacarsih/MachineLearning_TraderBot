# MachineLearning TraderBot

MachineLearning TraderBot is a Python trading research and execution project
for MetaTrader 5. It combines feature engineering, tree-based machine learning
models, backtesting, paper trading, live execution controls, and a Flask
dashboard for monitoring model and trading state.

The repository is prepared for GitHub with source code, tests, templates, and
documentation only. Local credentials, datasets, logs, trained models,
backtest output, reports, and databases are intentionally ignored.

## Features

- Historical OHLCV loading from MT5 or CSV.
- Technical feature engineering and triple-class trade labeling.
- Model training with XGBoost, LightGBM, CatBoost, and comparative selection.
- Event-driven historical backtesting with performance reports and charts.
- Paper trading loop for real-time simulation without broker orders.
- MT5 live trading integration with explicit safety gates.
- Flask dashboard for context, config, logs, saved models, and control actions.
- Tests for dashboard behavior, feature pipeline, currency risk, MT5 config,
  order safety, and live trade analysis.

## Repository Layout

```text
config/       Runtime settings and MT5 credential loading
data/         Data loading, feature engineering, and labeling code
models/       Model wrappers, training, and comparison utilities
strategy/     Signal, risk, rules, and market filters
mt5/          MT5 connector, account manager, order execution, paper DB
backtest/     Backtester and performance code
scripts/      CLI entry points for train, backtest, paper, live, analysis
templates/    Dashboard UI template
tests/        Automated test suite
utils/        Logging, dashboard helpers, banner, common utilities
```

Generated folders such as `logs/`, `reports/`, `saved_models/`, and
`catboost_info/` are local artifacts and are not committed. Market data files
under `data/` and output files under `backtest/` are ignored while the Python
source modules in those packages remain tracked.

## Requirements

- Windows for MetaTrader 5 Python integration.
- Python 3.10 or 3.11.
- MetaTrader 5 desktop terminal.
- Broker demo or live account for MT5 connectivity.

## Setup

```powershell
git clone https://github.com/sardacarsih/MachineLearning_TraderBot.git
cd MachineLearning_TraderBot
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Credentials

Copy the safe template and fill it locally:

```powershell
Copy-Item credentials.example.yaml credentials.yaml
```

Keep live credentials out of Git. The `.gitignore` excludes
`credentials*.yaml`, while allowing `credentials.example.yaml`.

Safe default:

```yaml
mt5:
  login: 0
  password: ""
  server: ""
  terminal_path: ""
  paper_trading: true
  trading_enabled: false
```

For live trading, set the real MT5 values only in your local credential file.
Use live mode only after paper testing and account verification.

You can also use environment variables supported by `config/mt5_config.py`:

```text
MT5_LOGIN
MT5_PASSWORD
MT5_SERVER
MT5_PATH
```

## Usage

Run commands through the unified entry point:

```powershell
.\.venv\Scripts\python.exe main.py [train | backtest | compare | live | paper | analyze-live] --help
```

### Train Models

```powershell
.\.venv\Scripts\python.exe main.py train --symbol XAUUSD --timeframe M5 --months 12 --walk-forward --tune
```

Train from a local CSV:

```powershell
.\.venv\Scripts\python.exe main.py train --symbol XAUUSD --timeframe M5 --csv data\XAUUSD\M5\xauusd_m5_12m.csv --models xgboost lightgbm catboost
```

### Compare Timeframes

```powershell
.\.venv\Scripts\python.exe main.py compare --symbol XAUUSD --timeframes M1 M5 M15 --months 12 --strategy-mode hybrid
```

### Backtest

```powershell
.\.venv\Scripts\python.exe main.py backtest --symbol XAUUSD --timeframe M5 --model saved_models\XAUUSD\M5\selected_catboost_model --months 6 --balance 10000
```

### Paper Trading

```powershell
.\.venv\Scripts\python.exe main.py paper --symbol XAUUSD --timeframe M5 --model saved_models\XAUUSD\M5\selected_catboost_model --config credentials.yaml
```

### Live Trading

```powershell
.\.venv\Scripts\python.exe main.py live --symbol XAUUSD --timeframe M5 --model saved_models\XAUUSD\M5\selected_catboost_model --config credentials.yaml --live
```

Live trading can place real orders when the local credential file enables it:

```yaml
paper_trading: false
trading_enabled: true
```

Review the account, symbol mapping, lot size, and risk settings before using
real funds.

### Analyze Trading Logs

```powershell
.\.venv\Scripts\python.exe main.py analyze-live --symbol XAUUSD --timeframe M5 --start-date 20260604 --end-date 20260604
```

Include MT5 deal history when credentials are configured:

```powershell
.\.venv\Scripts\python.exe main.py analyze-live --symbol XAUUSD --timeframe M5 --mt5-history --history-days 1 --config credentials.yaml
```

### Dashboard

```powershell
.\.venv\Scripts\python.exe dashboard.py
```

Then open the local Flask URL printed by the process.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

If the virtual environment is not active:

```powershell
python -m pytest
```

## Artifact Policy

The following are intentionally local-only:

- Real broker credentials and `.env` files.
- Market data snapshots and CSV caches.
- Trained model binaries and serialized model files.
- Logs, reports, paper trading databases, and backtest output.
- CatBoost training metadata and Python caches.

This keeps the GitHub repository small, reproducible, and safer to share.
Regenerate artifacts by running the training, backtest, paper, live, or
analysis commands in your local environment.

## Risk Disclaimer

This project is for educational and research purposes only. Algorithmic
trading involves high leverage and a significant risk of capital loss. Past
performance does not guarantee future results. Do not trade money you cannot
afford to lose. The authors are not responsible for losses incurred by using
this software.

## License

MIT License. See [LICENSE](LICENSE).
