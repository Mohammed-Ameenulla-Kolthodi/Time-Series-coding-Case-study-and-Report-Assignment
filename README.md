# Appliance Energy Forecasting

A single-script, reproducible time-series forecasting pipeline comparing
benchmark models, SARIMAX, a feature-based ML model (XGBoost), and a
time-series foundation model (Chronos) on the UCI **Appliances Energy
Prediction** dataset.

## Project aim

Forecast short-term household appliance energy use and evaluate whether
increasingly complex models improve on simple benchmark methods.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -r requirements.txt

python run_pipeline.py
```

This downloads the dataset (cached after the first run), runs every
model, and writes forecasts, metrics, and figures to `outputs/`.

### Faster / offline runs

```bash
# Skip the expensive SARIMA grid search (uses a fixed order instead)
python run_pipeline.py --no-sarima-search

# Shrink the grid-search window (default: last 90 days of training data)
python run_pipeline.py --sarima-search-window 720

# Skip the foundation model (e.g. no internet access to Hugging Face)
python run_pipeline.py --no-foundation-model
```

## Repository structure

```text
appliance-energy-forecasting/
├── README.md
├── requirements.txt
├── .gitignore
├── run_pipeline.py       <- everything: data, models, evaluation, plots
├── Appliance_Energy_Forecasting_Report.docx   (or .pdf)
├── data/
│   ├── raw/               (downloaded CSV, gitignored)
│   └── processed/         (cached hourly CSV, gitignored)
├── outputs/
│   ├── figures/
│   ├── forecasts/all_forecasts.csv
│   └── metrics/model_comparison.csv, sarima_grid_search.csv, ...
└── tests/
    └── test_pipeline.py
```

`run_pipeline.py` is organised into clearly commented, numbered sections,
each built from small functions (not one unbroken block):

1. Config
2. Data loading & preparation (download, clean, resample to hourly)
3. EDA / stationarity (ADF, KPSS, ACF/PACF, differencing)
4. Benchmark models (mean, naive, seasonal naive x2, drift)
5. SARIMAX (grid search by AIC, residual diagnostics, CI forecast)
6. Feature engineering (lag/rolling/time features, no leakage)
7. Feature-based ML model (XGBoost)
8. Foundation model (Chronos, zero-shot, with a documented fallback)
9. Evaluation metrics (MAE, RMSE, MASE, Bias)
10. Plotting
11. Main pipeline (orchestrates everything) + CLI

## Notable design decisions (for the report)

**SARIMA grid search.** The brief asks for a full search over
p ∈ [0,6], d ∈ [0,2], q ∈ [0,6] (147 combinations) by AIC. Fitting 147 full
seasonal SARIMAX models on the entire (~8,700-hour) training set would take
well over an hour, so the search is run on the most recent
`--sarima-search-window` hours (default 90 days) — local ARIMA dynamics are
unlikely to differ much across a stable recent window. The **final** model
is refit on the full training set once the order is chosen. The seasonal
order is fixed at `(1,1,1,24)` (from the stationarity analysis) rather than
additionally grid-searched, since a combined non-seasonal × seasonal search
(147 × 8 = 1,176 fits) isn't feasible for a course assignment. State both
trade-offs explicitly in your report.

**Foundation model.** Chronos needs to download pretrained weights from
Hugging Face on first use. If the machine has no route to
`huggingface.co` (e.g. a locked-down sandbox), the code automatically
falls back to Holt-Winters exponential smoothing and prints a
`RuntimeWarning`. **This fallback is not a foundation model** — if it
triggers, say so explicitly in your report, and re-run somewhere with
normal internet access (e.g. Google Colab) for a genuine Chronos result.
Install with `pip install chronos-forecasting torch`.

**Data leakage.** Lag/rolling target features are always built with
`.shift(1)` first, so no feature ever sees the current/future target.
Time-of-day/day-of-week features are legitimately known in advance and are
not shifted. The outdoor weather exogenous variables used by SARIMAX and
the feature model are the **realised test-period values** — this is a
**conditional forecast**, not a true operational forecast (a real
deployment would only have a weather *forecast*, not realised values).
This is exactly what assignment Question 5 asks you to discuss.

## Outputs produced

- `outputs/forecasts/all_forecasts.csv` — actual values + every model's forecast
- `outputs/metrics/model_comparison.csv` — MAE/RMSE/MASE/Bias per model, plus
  % RMSE improvement over the strongest benchmark
- `outputs/metrics/stationarity_tests.csv` — ADF/KPSS results at each differencing stage
- `outputs/metrics/sarima_grid_search.csv` — every (p,d,q) tried and its AIC
- `outputs/metrics/sarimax_residual_diagnostics.csv` — Ljung-Box, residual moments
- `outputs/metrics/feature_importance.csv` — XGBoost feature importances
- `outputs/metrics/foundation_model_source.csv` — records whether the foundation-model
  result came from a real Chronos run (`"chronos"`) or the documented Holt-Winters
  fallback (`"holtwinters_fallback"`) — check this before citing the foundation-model
  result in the report
- `outputs/figures/` — overview plot, seasonal decomposition, ACF/PACF (level,
  differenced, seasonal-differenced, and SARIMAX residuals), forecast
  comparison with CI band, error diagnostics, feature importance,
  model-comparison bar chart

## Tests

```bash
pytest
```

Covers: lag features don't leak future values, rolling features exclude the
current observation, the ML table has no missing values, forecast lengths
match the requested horizon, naive/seasonal-naive/mean forecasts are
computed correctly, and MASE is exactly zero for a perfect forecast.

## Status

The pipeline has been run to completion with the full 147-combination SARIMA
grid search and a genuine zero-shot Chronos forecast (confirmed via
`outputs/metrics/foundation_model_source.csv` = `"chronos"`, not the
fallback). All outputs in `outputs/` reflect this final run. The
accompanying report (`Appliance_Energy_Forecasting_Report.docx` /
`.pdf`) answers the six assignment discussion questions using these results.
