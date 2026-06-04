# MachineLearning TraderBot

MachineLearning TraderBot adalah proyek riset dan eksekusi trading berbasis
Python untuk MetaTrader 5. Proyek ini menggabungkan feature engineering,
model machine learning berbasis tree, backtesting, paper trading, kontrol live
trading, dan dashboard Flask untuk memantau status model serta aktivitas
trading.

Repository ini disiapkan untuk GitHub dengan hanya menyertakan source code,
tests, template, dan dokumentasi yang aman. Credential lokal, dataset, logs,
model hasil training, output backtest, reports, dan database tidak ikut
di-commit.

## Fitur Utama

- Load data historis OHLCV dari MT5 atau file CSV lokal.
- Feature engineering indikator teknikal dan labeling sinyal tiga kelas.
- Training model dengan XGBoost, LightGBM, CatBoost, dan proses perbandingan.
- Backtesting historis event-driven dengan laporan performa dan chart.
- Paper trading real-time tanpa mengirim order broker sungguhan.
- Integrasi live trading MT5 dengan safety gate eksplisit.
- Dashboard Flask untuk konteks, konfigurasi, logs, model tersimpan, dan kontrol proses.
- Test suite untuk dashboard, feature pipeline, currency risk, config MT5,
  order safety, dan analisis live trade.

## Struktur Repository

```text
config/       Pengaturan runtime dan loader credential MT5
data/         Data loader, feature engineering, dan labeling
models/       Wrapper model, training, dan utility perbandingan model
strategy/     Signal, risk manager, trading rules, dan market filters
mt5/          Connector MT5, account manager, order executor, paper DB
backtest/     Backtester dan perhitungan performa
scripts/      Entry point CLI untuk train, backtest, paper, live, analysis
templates/    Template UI dashboard
tests/        Automated test suite
utils/        Logging, helper dashboard, banner, dan utility umum
```

Folder hasil generate seperti `logs/`, `reports/`, `saved_models/`, dan
`catboost_info/` bersifat lokal dan tidak di-commit. File data market di
`data/` serta output backtest di `backtest/` juga di-ignore, sementara source
Python di package tersebut tetap dilacak Git.

## Kebutuhan Sistem

- Windows, karena integrasi Python MetaTrader 5 membutuhkan MT5 desktop.
- Python 3.10 atau 3.11.
- MetaTrader 5 Desktop Terminal.
- Akun broker demo atau live untuk koneksi MT5.

## Instalasi

```powershell
git clone https://github.com/sardacarsih/MachineLearning_TraderBot.git
cd MachineLearning_TraderBot
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Konfigurasi Credential

Salin template aman terlebih dahulu:

```powershell
Copy-Item credentials.example.yaml credentials.yaml
```

Jangan commit credential asli. `.gitignore` sudah mengecualikan
`credentials*.yaml`, tetapi tetap mengizinkan `credentials.example.yaml`.

Default aman:

```yaml
mt5:
  login: 0
  password: ""
  server: ""
  terminal_path: ""
  paper_trading: true
  trading_enabled: false
```

Untuk live trading, isi nilai MT5 asli hanya di file credential lokal Anda.
Gunakan live mode hanya setelah paper testing dan verifikasi akun.

Environment variable yang juga didukung oleh `config/mt5_config.py`:

```text
MT5_LOGIN
MT5_PASSWORD
MT5_SERVER
MT5_PATH
```

## Cara Menjalankan

Gunakan `main.py` sebagai entry point utama:

```powershell
.\.venv\Scripts\python.exe main.py [train | backtest | compare | live | paper | analyze-live] --help
```

### Training Model

```powershell
.\.venv\Scripts\python.exe main.py train --symbol XAUUSD --timeframe M5 --months 12 --walk-forward --tune
```

Training dari CSV lokal:

```powershell
.\.venv\Scripts\python.exe main.py train --symbol XAUUSD --timeframe M5 --csv data\XAUUSD\M5\xauusd_m5_12m.csv --models xgboost lightgbm catboost
```

### Membandingkan Timeframe

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

Live trading dapat mengirim order sungguhan jika file credential lokal
mengaktifkan konfigurasi berikut:

```yaml
paper_trading: false
trading_enabled: true
```

Sebelum memakai dana real, pastikan akun, mapping symbol, lot size, dan risk
setting sudah diverifikasi.

### Analisis Log Trading

```powershell
.\.venv\Scripts\python.exe main.py analyze-live --symbol XAUUSD --timeframe M5 --start-date 20260604 --end-date 20260604
```

Gunakan deal history MT5 jika credential sudah dikonfigurasi:

```powershell
.\.venv\Scripts\python.exe main.py analyze-live --symbol XAUUSD --timeframe M5 --mt5-history --history-days 1 --config credentials.yaml
```

### Dashboard

```powershell
.\.venv\Scripts\python.exe dashboard.py
```

Setelah proses berjalan, buka URL Flask lokal yang muncul di terminal.

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Jika virtual environment tidak aktif:

```powershell
python -m pytest
```

## Kebijakan Artifact

File dan folder berikut sengaja dibuat lokal saja:

- Credential broker asli dan file `.env`.
- Snapshot data market dan cache CSV.
- Binary model hasil training dan file model serialized.
- Logs, reports, database paper trading, dan output backtest.
- Metadata training CatBoost dan cache Python.

Kebijakan ini menjaga repository tetap ringan, reproducible, dan lebih aman
untuk dibagikan. Artifact dapat dibuat ulang dengan menjalankan command
training, backtest, paper, live, atau analysis di environment lokal.

## Disclaimer Risiko

Proyek ini dibuat untuk tujuan edukasi dan riset. Algorithmic trading memiliki
risiko kerugian modal yang signifikan, terutama saat menggunakan leverage.
Performa masa lalu tidak menjamin hasil di masa depan. Jangan trading dengan
dana yang tidak siap Anda tanggung kerugiannya. Pengembang tidak bertanggung
jawab atas kerugian finansial akibat penggunaan software ini.

## Lisensi

MIT License. Lihat [LICENSE](LICENSE).
