# Dokumentasi End-to-End Pipeline MT5 ke Live Trading

Dokumen ini menjelaskan alur developer dari pengambilan data MetaTrader 5 (MT5) sampai model siap dipakai untuk paper/live trading. Fokusnya adalah proses, file kode utama, command yang dapat dijalankan, artefak output, dan cara membaca hasil.

> Catatan risiko: mode paper adalah default untuk launcher rekomendasi. Order broker real hanya dikirim jika command live diberi flag `-Live` atau proses Python menerima argumen `--live`.

## 1. Overview Pipeline

Alur lengkap sistem:

```text
MT5 historical bars
  -> cache CSV
  -> feature engineering
  -> label generation
  -> chronological train/validation/test split
  -> model training
  -> optional walk-forward validation
  -> model comparison report
  -> saved model candidates
  -> historical backtest simulation
  -> timeframe/model ranking
  -> paper/live launcher
  -> logs and live trade analysis
```

File entrypoint utama:

- `main.py`: router command `train`, `backtest`, `compare`, `paper`, `live`, dan `analyze-live`.
- `scripts/train.py`: pipeline training dan evaluasi model.
- `models/model_trainer.py`: split data, training, tuning, walk-forward, overfitting check, dan save model.
- `scripts/backtest_run.py`: load model, generate signal, dan menjalankan backtest.
- `backtest/backtester.py`: simulasi order historis, SL/TP, trailing stop, commission, slippage, dan metrik trading.
- `backtest/performance.py`: export `performance_report.txt` dan chart performa.
- `scripts/train_compare_timeframes.py`: training/backtest multi-timeframe dan ranking kandidat.
- `scripts/run_live_priority_recommendations.ps1`: launcher paper/live berdasarkan prioritas final.

## 2. Prasyarat Environment

Pastikan komponen berikut siap sebelum menjalankan pipeline:

- Virtualenv `.venv` sudah dibuat dan dependency dari `requirements.txt` sudah terpasang.
- Terminal MT5 aktif, login ke akun yang benar, dan symbol broker tersedia di Market Watch.
- File credentials YAML tersedia, misalnya `credentials_xauusd_m1.yaml`, `credentials_xagusd_m5.yaml`, atau `credentials_ustec_m5.yaml`.
- Symbol memakai nama broker yang benar. Contoh yang dipakai repo: `XAUUSD`, `XAGUSD`, dan `USTEC_x100`.
- Folder output akan dibuat otomatis oleh konfigurasi:
  - `data/<SYMBOL>/<TIMEFRAME>/`
  - `saved_models/<SYMBOL>/<TIMEFRAME>/`
  - `backtest/<SYMBOL>/<TIMEFRAME>/`
  - `logs/<SYMBOL>/<TIMEFRAME>/`
  - `reports/`

Command dasar memakai Python virtualenv:

```powershell
.\.venv\Scripts\python.exe main.py [train | compare | backtest | paper | live | analyze-live] --help
```

## 3. Pengambilan Data Dari MT5

Data historis diambil saat menjalankan `train` atau `backtest` tanpa argumen `--csv`.

Di `scripts/train.py`, prosesnya:

1. `config.set_symbol(...)` dan `config.set_timeframe(...)` mengatur konteks symbol/timeframe dan folder output.
2. Jika MT5 tersedia, script mencoba membaca spesifikasi symbol:
   - `point`
   - `digits`
   - `contract_size`
3. `DataLoader.load_from_mt5(months=...)` mengambil bar historis.
4. Data disimpan sebagai cache CSV supaya run berikutnya bisa fallback tanpa MT5.

Jika pengambilan MT5 gagal, script mencari cache lewat `find_cached_csv(symbol, timeframe, months)`.

Contoh lokasi cache:

```text
data/XAUUSD/M5/xauusd_m5_12m.csv
data/XAGUSD/M15/xagusd_m15_12m.csv
data/USTEC_X100/M5/ustec_x100_m5_12m.csv
```

Jika ingin mengunci dataset agar hasil lebih reproducible, jalankan training/backtest dengan `--csv` ke file cache tertentu.

## 4. Training Model

Command training standar:

```powershell
.\.venv\Scripts\python.exe main.py train --symbol XAUUSD --timeframe M5 --months 12 --walk-forward
```

Tahapan di `scripts/train.py`:

1. Load data dari MT5 atau CSV.
2. `FeatureEngineer.add_all_features(...)` membuat feature teknikal.
3. `LabelGenerator.generate_labels(...)` membuat label target:
   - `NO_TRADE`
   - `BUY`
   - `SELL`
4. `ModelTrainer.prepare_data(...)` membagi data secara kronologis menjadi train/validation/test.
5. Optional `--tune` menjalankan Optuna untuk hyperparameter tuning.
6. `ModelTrainer.train_all_models(...)` melatih model tree yang dipilih:
   - `xgboost`
   - `lightgbm`
   - `catboost`
7. `ModelTrainer.evaluate_all_models(...)` mengevaluasi model di test set.
8. `ModelComparison.generate_report(...)` membuat laporan model.
9. `save_all_models()` menyimpan semua kandidat.
10. `save_best_model()` menyimpan selected model.

Output utama:

```text
saved_models/<SYMBOL>/<TIMEFRAME>/candidate_xgboost_model
saved_models/<SYMBOL>/<TIMEFRAME>/candidate_lightgbm_model
saved_models/<SYMBOL>/<TIMEFRAME>/candidate_catboost_model
saved_models/<SYMBOL>/<TIMEFRAME>/selected_<model>_model
backtest/<SYMBOL>/<TIMEFRAME>/model_comparison_report.txt
backtest/<SYMBOL>/<TIMEFRAME>/confusion_matrices.png
backtest/<SYMBOL>/<TIMEFRAME>/roc_curves.png
backtest/<SYMBOL>/<TIMEFRAME>/precision_recall_curves.png
backtest/<SYMBOL>/<TIMEFRAME>/feature_importance.png
```

### Chronological Split Dan Lookahead Trimming

`ModelTrainer.prepare_data(...)` memakai split kronologis, bukan random split. Default proporsinya:

```text
Train:      70%
Validation: 15%
Test:       15%
```

Karena label dibuat dari beberapa bar ke depan, boundary split dipotong sebesar `config.data.label_lookahead_max`. Tujuannya mencegah label di akhir train/validation memakai candle dari periode berikutnya.

### Cara Model Dipilih Saat Training

Model terbaik saat training dipilih dari `trade_signal_precision`, yaitu precision gabungan untuk sinyal trade `BUY` dan `SELL`. Metrik ini dipakai karena false entry lebih mahal daripada sekadar salah klasifikasi `NO_TRADE`.

## 5. Walk-Forward Validation

Walk-forward aktif jika command training memakai `--walk-forward`.

```powershell
.\.venv\Scripts\python.exe main.py train --symbol XAUUSD --timeframe M5 --months 12 --walk-forward
```

Implementasinya ada di `ModelTrainer.walk_forward_analysis(...)`.

Proses per fold:

1. Data tetap urut waktu.
2. Dataset dibagi menjadi `k_splits + 1` segmen, default `k_splits = config.model.walk_forward_splits`.
3. Training window bersifat expanding:
   - fold berikutnya memakai data train yang lebih panjang.
4. Test window bergerak maju:
   - setiap fold menguji periode waktu setelah training window.
5. Akhir training window dipotong sebesar `label_lookahead_max` agar label train tidak mengintip test window.
6. Dari training window fold, 85% dipakai train dan 15% dipakai validation.
7. Model dibuat ulang dari nol untuk setiap fold.

Metrik yang dicatat per fold:

- `accuracy`
- `precision_macro`
- `recall_macro`
- `f1_macro`
- `trade_signal_precision`

Di log training, script menampilkan rata-rata `trade_signal_precision` per model.

Penting: walk-forward memvalidasi stabilitas kualitas sinyal ML across time. Walk-forward bukan simulasi profit trading, sehingga tidak menghasilkan `Net Profit`, `Profit Factor`, atau `Max Drawdown`. Metrik profit/risk dihitung di tahap backtest.

## 6. Historical Backtest

Command backtest:

```powershell
.\.venv\Scripts\python.exe main.py backtest --symbol XAUUSD --timeframe M5 --model saved_models\XAUUSD\M5\candidate_xgboost_model --months 6 --balance 10000 --strategy-mode ml
```

Tahapan di `scripts/backtest_run.py`:

1. Load model tersimpan dengan `load_saved_model(...)`.
2. Load data historis dari MT5 atau fallback CSV.
3. Jalankan `FeatureEngineer` dengan schema yang sesuai model.
4. Validasi feature schema agar urutan, nama, dan jumlah feature cocok dengan model.
5. `model.predict_proba(...)` menghasilkan probabilitas kelas.
6. Probabilitas dikonversi menjadi sinyal:
   - `BUY` jika probabilitas BUY melewati confidence threshold dan lebih besar dari SELL.
   - `SELL` jika probabilitas SELL melewati confidence threshold dan lebih besar dari BUY.
   - selain itu `NO_TRADE`.
7. `Backtester.run(...)` menjalankan simulasi trading historis.
8. `PerformanceAnalyzer.export_report(...)` menulis laporan dan chart.

Confidence threshold diambil dari `config.resolve_confidence().signal_threshold`, termasuk override per symbol/timeframe jika tersedia di credentials/config.

### Mode Strategi

Backtest mendukung dua mode:

- `--strategy-mode ml`: menerima sinyal ML setelah safety/no-trade filters.
- `--strategy-mode hybrid`: sinyal ML masih harus lolos validasi technical trading rules.

Untuk rekomendasi live final di repo ini, launcher prioritas memakai `--strategy-mode ml`.

### Simulasi Trade

`backtest/backtester.py` mensimulasikan:

- batas `max_open_positions`
- spread/no-trade filter
- entry berdasarkan close price bar
- SL dari `RiskManager.calculate_sl(...)`
- TP dari `RiskManager.calculate_tp(...)`
- trailing stop
- slippage entry/exit
- commission per lot
- lot sizing berdasarkan risk
- optional confidence lot multiplier
- forced close untuk posisi tersisa di akhir data

Output backtest:

```text
backtest/<SYMBOL>/<TIMEFRAME>/<strategy-mode>/performance_report.txt
backtest/<SYMBOL>/<TIMEFRAME>/<strategy-mode>/equity_curve.png
backtest/<SYMBOL>/<TIMEFRAME>/<strategy-mode>/drawdown.png
backtest/<SYMBOL>/<TIMEFRAME>/<strategy-mode>/monthly_returns.png
backtest/<SYMBOL>/<TIMEFRAME>/<strategy-mode>/trade_distribution.png
backtest/<SYMBOL>/<TIMEFRAME>/<strategy-mode>/cumulative_pnl.png
```

## 7. Cara Membaca Metrik Backtest

`performance_report.txt` berisi:

- `Initial Balance`: modal awal simulasi.
- `Net Profit`: total profit/loss bersih dari semua trade.
- `Ending Balance`: initial balance + net profit.
- `Total Trades`: jumlah trade tertutup.
- `Winning Trades`: jumlah dan persentase trade profit.
- `Losing Trades`: jumlah dan persentase trade loss.
- `Profit Factor`: gross profit / gross loss.
- `Trade Expectancy`: rata-rata ekspektasi profit per trade.
- `Max Drawdown`: penurunan terbesar dari equity peak.
- `Sharpe Ratio`: risk-adjusted return berbasis daily return.
- `Sortino Ratio`: seperti Sharpe, tetapi fokus downside volatility.
- `Calmar Ratio`: annualized return dibanding max drawdown.
- `Average Win` dan `Average Loss`: rata-rata profit/loss per trade.
- `Close Reason Breakdown`: jumlah close karena TP, SL, atau force close.

Profit factor tinggi tidak otomatis aman jika jumlah trade terlalu kecil. Drawdown rendah juga bisa misleading jika model hanya membuka sedikit trade. Karena itu ranking memakai kombinasi validitas, profit, PF, drawdown, winrate, dan trade count.

## 8. Compare Dan Ranking Rekomendasi

Command compare multi-timeframe:

```powershell
.\.venv\Scripts\python.exe main.py compare --symbol XAUUSD --timeframes M1 M5 M15 --months 12 --strategy-mode ml
```

`scripts/train_compare_timeframes.py` menjalankan training/backtest untuk timeframe yang dipilih, lalu membaca:

- `model_comparison_report.txt`
- `<strategy-mode>/performance_report.txt`

Aturan validitas kandidat:

- `total_trades >= config.backtest.min_trades`
- `profit_factor >= config.backtest.min_profit_factor`
- `max_drawdown_pct <= config.backtest.max_drawdown_pct`
- `winrate >= config.backtest.min_winrate`

Ranking kandidat:

```text
valid backtest
  -> net profit lebih tinggi
  -> profit factor lebih tinggi
  -> drawdown lebih rendah
  -> winrate lebih tinggi
  -> trade count lebih tinggi
```

Output ranking:

```text
backtest/<SYMBOL>/timeframe_model_comparison_report.txt
```

Jika tidak ada kandidat valid, report menampilkan `Winner: NONE` dan `Best Candidate` sebagai kandidat terbaik menurut ranking meski gagal aturan validitas.

## 9. Paper Dan Live Operation

Launcher prioritas final:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_live_priority_recommendations.ps1 -ValidateOnly
```

`-ValidateOnly` memeriksa:

- Python virtualenv ada.
- Model path ada.
- Credentials path ada.
- Password credentials bukan placeholder.

Menjalankan top-N prioritas dalam paper mode:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_live_priority_recommendations.ps1 -TopN 3
```

Menjalankan real live mode:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_live_priority_recommendations.ps1 -Live
```

Default script adalah paper mode, sehingga tidak mengirim order real. Flag `-Live` wajib untuk broker order real.

Setiap job memakai:

- symbol/timeframe tertentu
- model path dari `saved_models/...`
- credentials YAML per symbol/timeframe
- strategy mode `ml`
- log folder `logs/<SYMBOL>/<TIMEFRAME>/`

Contoh command Python live yang disusun launcher:

```powershell
.\.venv\Scripts\python.exe main.py live --config credentials_xauusd_m1.yaml --symbol XAUUSD --timeframe M1 --model saved_models\XAUUSD\M1\candidate_lightgbm_model --strategy-mode ml
```

Untuk real live, launcher menambahkan `--live`.

## 10. Logs Dan Analisis Live

Output runtime live/paper berada di:

```text
logs/<SYMBOL>/<TIMEFRAME>/main_<YYYYMMDD>.log
logs/<SYMBOL>/<TIMEFRAME>/signals_<YYYYMMDD>.log
logs/<SYMBOL>/<TIMEFRAME>/trades_<YYYYMMDD>.log
```

Analisis live bisa dijalankan dengan:

```powershell
.\.venv\Scripts\python.exe main.py analyze-live --symbol XAUUSD --timeframe M5 --start-date 20260604 --end-date 20260604
```

Atau memakai MT5 history:

```powershell
.\.venv\Scripts\python.exe main.py analyze-live --symbol XAUUSD --timeframe M5 --mt5-history --history-days 1 --config credentials.yaml
```

Output analysis tersimpan di `reports/`.

## 11. Troubleshooting

### MT5 gagal connect

Cek:

- MT5 terminal sedang terbuka.
- Akun sudah login.
- Server credentials benar.
- Python bisa import `MetaTrader5`.
- Terminal MT5 sesuai instalasi yang digunakan package `MetaTrader5`.

Jika MT5 gagal tetapi cache CSV ada, script akan mencoba fallback CSV.

### Symbol broker tidak cocok

Gejala:

- log menampilkan symbol tidak ditemukan di Market Watch.
- data MT5 gagal di-load.
- order live ditolak karena symbol invalid.

Solusi:

- cek nama persis symbol broker, misalnya `USTEC_x100` vs `USTEC_X100`.
- pastikan symbol dipilih di Market Watch.
- gunakan nama yang sama pada command, credentials, dan folder model.

### Cached CSV tidak ditemukan

Gejala:

- MT5 gagal load data.
- fallback `find_cached_csv(...)` tidak menemukan file.

Solusi:

- jalankan ulang dengan MT5 aktif agar cache dibuat.
- atau berikan file eksplisit:

```powershell
.\.venv\Scripts\python.exe main.py train --symbol XAUUSD --timeframe M5 --csv data\XAUUSD\M5\xauusd_m5_12m.csv
```

### Feature schema mismatch

Gejala:

- backtest/live menolak model karena feature schema tidak cocok.

Penyebab umum:

- model dilatih dengan versi feature lama.
- model timeframe berbeda dari data live/backtest.
- higher timeframe features berbeda antara training dan inference.

Solusi:

- pakai model dari folder symbol/timeframe yang sama.
- retrain model setelah perubahan feature engineering.
- jangan memindahkan model antar timeframe.

### Model path salah

Gejala:

- `Failed to load model`
- `Rank model not found`

Solusi:

- cek folder `saved_models/<SYMBOL>/<TIMEFRAME>/`.
- gunakan `candidate_<model>_model` atau `selected_<model>_model` yang benar.

### Report tertimpa run terbaru

Beberapa output memakai nama tetap seperti `performance_report.txt`. Jika backtest dijalankan ulang untuk symbol/timeframe/strategy yang sama, report lama bisa tertimpa.

Untuk audit hasil final:

- simpan salinan report penting dengan nama timestamp.
- catat command dan model path yang menghasilkan report.
- gunakan `reports/*.json` atau dokumen snapshot jika tersedia.

## 12. Checklist Sebelum Live

Sebelum memakai `-Live`, lakukan checklist ini:

- Jalankan `-ValidateOnly` pada launcher.
- Pastikan credentials bukan template/placeholder.
- Pastikan model path sesuai symbol/timeframe.
- Review `model_comparison_report.txt`.
- Review `performance_report.txt`.
- Pastikan kandidat lolos minimum trade, PF, drawdown, dan winrate.
- Review hasil walk-forward, terutama stabilitas `trade_signal_precision`.
- Jalankan paper mode dulu.
- Review `logs/<SYMBOL>/<TIMEFRAME>/signals_*.log`.
- Review `logs/<SYMBOL>/<TIMEFRAME>/trades_*.log`.
- Cek daily drawdown dan consecutive loss behavior.
- Pastikan account balance, currency, contract size, min lot, max lot, dan spread sesuai broker.

## 13. Command Ringkas

Training dengan walk-forward:

```powershell
.\.venv\Scripts\python.exe main.py train --symbol XAUUSD --timeframe M5 --months 12 --walk-forward
```

Compare multi-timeframe:

```powershell
.\.venv\Scripts\python.exe main.py compare --symbol XAUUSD --timeframes M1 M5 M15 --months 12 --strategy-mode ml
```

Backtest model tertentu:

```powershell
.\.venv\Scripts\python.exe main.py backtest --symbol XAUUSD --timeframe M5 --model saved_models\XAUUSD\M5\candidate_xgboost_model --months 6 --balance 10000 --strategy-mode ml
```

Validasi launcher rekomendasi:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_live_priority_recommendations.ps1 -ValidateOnly
```

Jalankan top 3 rekomendasi dalam paper mode:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_live_priority_recommendations.ps1 -TopN 3
```

Jalankan rekomendasi dalam real live mode:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_live_priority_recommendations.ps1 -Live
```
