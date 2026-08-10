"""
run_pipeline.py
================

Appliance Energy Forecasting -- full case study in one script.

Forecasts household appliance energy use (target: `Appliances`) 24 hours
ahead using five model families of increasing complexity, evaluates them
all on the same held-out test period, and saves every plot/metric/forecast
needed for the report.

Pipeline stages (see assignment brief Parts 1-8):
    1. Load & prepare data (download, clean, resample to hourly)
    2. Define the forecasting problem (train/test split, metrics)
    3. Benchmark models (mean, naive, seasonal naive x2, drift)
    4. SARIMAX (grid search by AIC, residual diagnostics, CI forecast)
    5/6. Covariates + feature-based ML model (XGBoost)
    7. Foundation model (Chronos, zero-shot)
    8. Evaluation (MAE, RMSE, MASE, Bias; plots; comparison to benchmark)

Usage
-----
    python run_pipeline.py
    python run_pipeline.py --no-sarima-search        # skip AIC grid search
    python run_pipeline.py --sarima-search-window 720
    python run_pipeline.py --no-foundation-model      # skip Chronos

Outputs are written to outputs/{figures,forecasts,metrics}/.
"""

from __future__ import annotations

import argparse
import itertools
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings("ignore")


# ============================================================
# 0. Configuration
# ============================================================

RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FORECAST_DIR = OUTPUT_DIR / "forecasts"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURE_DIR = OUTPUT_DIR / "figures"

for _p in [RAW_DIR, PROCESSED_DIR, FORECAST_DIR, METRICS_DIR, FIGURE_DIR]:
    _p.mkdir(parents=True, exist_ok=True)

TARGET = "Appliances"

# Hourly data: 24 obs = 1 day, 168 obs = 1 week
DAILY_PERIOD = 24
WEEKLY_PERIOD = 168

# Test period: final 14 days of hourly data
TEST_STEPS = 14 * 24

# UCI hosting is not always reachable from every network; fall back to a
# GitHub mirror of the identical file (hosted by the dataset's authors).
DATA_URL_PRIMARY = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/"
    "energydata_complete.csv"
)
DATA_URL_MIRROR = (
    "https://raw.githubusercontent.com/LuisM78/"
    "Appliances-energy-prediction-data/master/energydata_complete.csv"
)
RAW_CSV_PATH = RAW_DIR / "energydata_complete.csv"
HOURLY_CSV_PATH = PROCESSED_DIR / "appliance_hourly.csv"

# Outdoor weather columns used as SARIMAX/feature-model exogenous variables.
# (Indoor T1..T9 / RH_1..RH_9 sensor columns are used as ML features only,
# not as SARIMAX exog, to keep the SARIMAX model small and interpretable.)
OUTDOOR_WEATHER_COLS = [
    "T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint",
]
INDOOR_SENSOR_COLS = [
    "T1", "RH_1", "T2", "RH_2", "T3", "RH_3", "T4", "RH_4", "T5", "RH_5",
    "T6", "RH_6", "T7", "RH_7", "T8", "RH_8", "T9", "RH_9",
]
# Random noise columns included by the original authors as a feature-
# selection sanity check; they carry no real information and are dropped.
NOISE_COLS = ["rv1", "rv2"]

# SARIMA grid-search ranges (as specified in the assignment brief)
SARIMA_P_RANGE = range(0, 7)   # 0..6
SARIMA_D_RANGE = range(0, 3)   # 0..2
SARIMA_Q_RANGE = range(0, 7)   # 0..6
SARIMA_SEASONAL_ORDER = (1, 1, 1, DAILY_PERIOD)

FIGSIZE = (14, 7)


# ============================================================
# 1. Data loading and preparation
# ============================================================

def download_raw_csv(dest: Path) -> Path:
    """Download the raw CSV from UCI, falling back to a GitHub mirror."""
    for url in (DATA_URL_PRIMARY, DATA_URL_MIRROR):
        try:
            print(f"Downloading data from {url} ...")
            urllib.request.urlretrieve(url, dest)
            return dest
        except Exception as exc:  # noqa: BLE001
            print(f"  failed ({exc}); trying next source if available.")
    raise RuntimeError(
        f"Could not download energydata_complete.csv. Please download it "
        f"manually and place it at {dest}"
    )


def load_raw_data(force_download: bool = False) -> pd.DataFrame:
    """Load the raw 10-minute dataset, downloading it if not already cached."""
    if force_download or not RAW_CSV_PATH.exists():
        download_raw_csv(RAW_CSV_PATH)

    df = pd.read_csv(RAW_CSV_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Some columns are stored as strings with leading whitespace (e.g. "  60")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop(columns=[c for c in NOISE_COLS if c in df.columns])
    return df


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Print and return a summary of missing values per column."""
    report = pd.DataFrame({
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(3),
    })
    report = report[report["n_missing"] > 0].sort_values("n_missing", ascending=False)
    print(f"Rows: {len(df)}")
    print("No missing values in the raw data." if report.empty else report)
    return report


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 10-minute data to hourly means. Small gaps introduced by
    resampling are filled by time interpolation (reasonable for a smooth
    physical quantity like energy demand); any remaining gaps are dropped.
    """
    hourly = df.resample("h").mean()
    n_before = hourly.isna().sum().sum()
    hourly = hourly.interpolate("time", limit_direction="both")
    n_after = hourly.isna().sum().sum()
    print(f"Hourly resample: filled {n_before - n_after} missing cells "
          f"via interpolation ({n_after} remain and are dropped).")
    return hourly.dropna()


def load_hourly_data(force_download: bool = False, force_resample: bool = False) -> pd.DataFrame:
    """Full data-loading entry point: raw -> clean -> hourly, with caching."""
    if not force_resample and HOURLY_CSV_PATH.exists():
        print(f"Loading cached hourly data from {HOURLY_CSV_PATH}")
        return pd.read_csv(HOURLY_CSV_PATH, index_col=0, parse_dates=True)

    raw = load_raw_data(force_download=force_download)
    missing_value_report(raw)

    hourly = resample_hourly(raw)
    hourly.to_csv(HOURLY_CSV_PATH)

    print(f"Raw shape: {raw.shape}  ->  Hourly shape: {hourly.shape}")
    print(f"Hourly range: {hourly.index.min()} -> {hourly.index.max()}")
    return hourly


def train_test_split_series(series: pd.Series, test_steps: int):
    """Chronological split: the last `test_steps` observations become the test set."""
    return series.iloc[:-test_steps], series.iloc[-test_steps:]


# ============================================================
# 2. EDA / stationarity plots
# ============================================================

def plot_series_overview(series: pd.Series, title: str, save_path=None):
    """Full-length line plot plus a one-week zoom-in."""
    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE)
    series.plot(ax=axes[0], linewidth=0.6)
    axes[0].set_title(f"{title} -- full period")
    axes[0].set_ylabel("Wh")

    series.iloc[: 24 * 7].plot(ax=axes[1], linewidth=1.2)
    axes[1].set_title(f"{title} -- first week (zoom)")
    axes[1].set_ylabel("Wh")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_seasonal_decomposition(series: pd.Series, period: int = 24, save_path=None):
    result = seasonal_decompose(series, model="additive", period=period)
    fig = result.plot()
    fig.set_size_inches(12, 8)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return result


def adf_test(series: pd.Series, name: str = "series") -> dict:
    """
    Augmented Dickey-Fuller test.
    H0: the series has a unit root (non-stationary). p < 0.05 => reject H0.
    """
    stat, pvalue, used_lag, nobs, crit_values, _ = adfuller(series.dropna(), autolag="AIC")
    is_stationary = pvalue < 0.05

    print(f"\nADF test on {name}: statistic={stat:.4f}, p-value={pvalue:.4g} "
          f"=> {'STATIONARY' if is_stationary else 'NON-STATIONARY'} (5%)")

    return {"test": "ADF", "series": name, "statistic": stat, "p_value": pvalue,
            "used_lag": used_lag, "n_obs": nobs, "is_stationary_5pct": is_stationary}


def kpss_test(series: pd.Series, name: str = "series") -> dict:
    """
    KPSS test, used alongside ADF because it has the OPPOSITE null
    hypothesis (H0: series IS stationary). Agreement between the two tests
    gives more confidence than either alone.
    """
    stat, pvalue, _, _ = kpss(series.dropna(), regression="c", nlags="auto")
    is_stationary = pvalue > 0.05

    print(f"KPSS test on {name}: statistic={stat:.4f}, p-value={pvalue:.4g} "
          f"=> {'STATIONARY' if is_stationary else 'NON-STATIONARY'} (5%)")

    return {"test": "KPSS", "series": name, "statistic": stat, "p_value": pvalue,
             "is_stationary_5pct": is_stationary}


def plot_acf_pacf(series: pd.Series, lags: int = 72, title_prefix: str = "", save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    axes[0].set_title(f"{title_prefix} ACF")
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], method="ywm")
    axes[1].set_title(f"{title_prefix} PACF")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_stationarity_analysis(series: pd.Series) -> pd.DataFrame:
    """
    Full stationarity workflow for the report:
    1. Test the raw (level) series (ADF + KPSS), plot ACF/PACF.
    2. First-order difference, re-test, re-plot.
    3. Seasonal (24h) difference on top, re-test, re-plot.
    """
    records = []

    records.append(adf_test(series, "Appliances (level)"))
    records.append(kpss_test(series, "Appliances (level)"))
    plot_acf_pacf(series, lags=168, title_prefix="Level -",
                  save_path=FIGURE_DIR / "acf_pacf_level.png")

    diff1 = series.diff(1).dropna()
    records.append(adf_test(diff1, "Appliances (1st difference)"))
    records.append(kpss_test(diff1, "Appliances (1st difference)"))
    plot_acf_pacf(diff1, lags=168, title_prefix="1st difference -",
                  save_path=FIGURE_DIR / "acf_pacf_diff1.png")

    diff_seasonal = diff1.diff(DAILY_PERIOD).dropna()
    records.append(adf_test(diff_seasonal, "Appliances (1st diff + seasonal diff-24)"))
    records.append(kpss_test(diff_seasonal, "Appliances (1st diff + seasonal diff-24)"))
    plot_acf_pacf(diff_seasonal, lags=168, title_prefix="1st diff + seasonal diff(24) -",
                  save_path=FIGURE_DIR / "acf_pacf_diff_seasonal.png")

    return pd.DataFrame(records)


# ============================================================
# 3. Benchmark models
# ============================================================

def mean_forecast(y_train, horizon, index):
    """Forecast = the historical mean, repeated."""
    return pd.Series(y_train.mean(), index=index, name="mean")


def naive_forecast(y_train, horizon, index):
    """Forecast = the last observed value, repeated (random-walk forecast)."""
    return pd.Series(y_train.iloc[-1], index=index, name="naive")


def seasonal_naive_forecast(y_train, horizon, index, seasonality):
    """
    Recursive seasonal-naive forecast: y_hat(t) = y(t - seasonality).
    Recursive because once the horizon exceeds `seasonality`, the forecast
    must reuse its own earlier forecast values.
    """
    history = list(y_train.values)
    values = []
    for _ in range(horizon):
        values.append(history[-seasonality])
        history.append(values[-1])
    name = "seasonal_naive_daily" if seasonality == 24 else "seasonal_naive_weekly"
    return pd.Series(values, index=index, name=name)


def drift_forecast(y_train, horizon, index):
    """Naive forecast extended with the average per-step change seen in training."""
    slope = (y_train.iloc[-1] - y_train.iloc[0]) / (len(y_train) - 1)
    values = [y_train.iloc[-1] + slope * step for step in range(1, horizon + 1)]
    return pd.Series(values, index=index, name="drift")


def all_benchmark_forecasts(y_train, horizon, index) -> dict:
    return {
        "mean": mean_forecast(y_train, horizon, index),
        "naive": naive_forecast(y_train, horizon, index),
        "seasonal_naive_daily": seasonal_naive_forecast(y_train, horizon, index, DAILY_PERIOD),
        "seasonal_naive_weekly": seasonal_naive_forecast(y_train, horizon, index, WEEKLY_PERIOD),
        "drift": drift_forecast(y_train, horizon, index),
    }


# ============================================================
# 4. SARIMAX: grid search, fit, forecast, diagnostics
# ============================================================
#
# Grid-search design note
# ------------------------
# The brief asks for a full grid search over p in [0,6], d in [0,2],
# q in [0,6] -- 147 candidate orders. Fitting 147 full seasonal SARIMAX
# models on ~8,700 hourly training observations each is very expensive
# (potentially well over an hour). To keep the search tractable while
# still honestly implementing "loop over all combinations, select by
# AIC", the search is run on a *recent subset* of the training data
# (`search_window` hours, default 90 days) -- local ARIMA dynamics are
# unlikely to differ much across a stable recent window. Once the best
# (p,d,q) is chosen, the FINAL model is refit on the FULL training set.
# The seasonal order is fixed at (1,1,1,24), based on the stationarity
# analysis (daily seasonality), rather than additionally grid-searched,
# since a combined non-seasonal x seasonal search (147 x 8 = 1,176 fits)
# is not feasible for a course assignment. Both trade-offs should be
# stated explicitly in the report.

def grid_search_sarima_order(y: pd.Series, seasonal_order=SARIMA_SEASONAL_ORDER,
                              search_window: int = 24 * 90,
                              p_range=None, d_range=None, q_range=None,
                              verbose: bool = True) -> pd.DataFrame:
    """Loop over (p,d,q) combinations, fit SARIMAX, record AIC for each."""
    p_range = p_range if p_range is not None else SARIMA_P_RANGE
    d_range = d_range if d_range is not None else SARIMA_D_RANGE
    q_range = q_range if q_range is not None else SARIMA_Q_RANGE

    y_search = y.iloc[-search_window:] if search_window else y
    combos = list(itertools.product(p_range, d_range, q_range))

    if verbose:
        print(f"Grid-searching {len(combos)} (p,d,q) combinations on the most "
              f"recent {len(y_search)} training hours (seasonal_order fixed at "
              f"{seasonal_order}) ...")

    results = []
    for i, (p, d, q) in enumerate(combos):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    y_search, order=(p, d, q), seasonal_order=seasonal_order,
                    trend="c" if d == 0 else None,
                    enforce_stationarity=False, enforce_invertibility=False,
                )
                fit = model.fit(disp=False, maxiter=50)
            aic = fit.aic
        except Exception:  # noqa: BLE001
            aic = np.nan
        results.append({"p": p, "d": d, "q": q, "AIC": aic})

        if verbose and (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(combos)} combinations tried")

    results_df = pd.DataFrame(results).sort_values("AIC").reset_index(drop=True)
    if verbose:
        print("\nTop 5 (p,d,q) combinations by AIC:")
        print(results_df.head(5).to_string(index=False))
    return results_df


def fit_sarimax(y_train, order, seasonal_order=SARIMA_SEASONAL_ORDER, X_train=None):
    """Fit the final SARIMAX model on the FULL training set with the chosen order."""
    model = SARIMAX(
        y_train, exog=X_train, order=order, seasonal_order=seasonal_order,
        trend="c" if order[1] == 0 else None,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    return model.fit(disp=False)


def forecast_sarimax(fit, horizon, index, X_test=None, alpha: float = 0.05):
    """Forecast `horizon` steps ahead with (1-alpha) confidence intervals."""
    fc = fit.get_forecast(steps=horizon, exog=X_test)

    mean = fc.predicted_mean
    mean.index = index
    mean.name = "sarimax"

    ci = fc.conf_int(alpha=alpha)
    ci.index = index
    ci.columns = ["sarimax_lower", "sarimax_upper"]

    return mean, ci


def sarimax_residual_diagnostics(fit, lags: int = 48) -> dict:
    """Ljung-Box test for residual autocorrelation + residual distribution stats."""
    residuals = fit.resid.dropna()
    lb = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    return {
        "residuals": residuals,
        "ljung_box_stat": float(lb["lb_stat"].iloc[0]),
        "ljung_box_pvalue": float(lb["lb_pvalue"].iloc[0]),
        "resid_mean": float(residuals.mean()),
        "resid_std": float(residuals.std()),
        "resid_skew": float(residuals.skew()),
        "resid_kurtosis": float(residuals.kurtosis()),
    }


def plot_residual_histogram(residuals: pd.Series, title="SARIMAX residuals", save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(residuals, bins=40, color="tab:blue", alpha=0.8)
    axes[0].set_title(f"{title}: histogram")
    axes[0].set_xlabel("Residual")

    scipy_stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title(f"{title}: Q-Q plot")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 6. Feature engineering (for the feature-based ML model)
# ============================================================
#
# Data-leakage safeguard: every lag/rolling feature of the TARGET is built
# with `.shift(1)` before any rolling window, so no feature can ever see
# the current or a future value of the target. Time-of-day / day-of-week
# features are exempt from this rule because they are genuinely known in
# advance for any future timestamp.

DEFAULT_LAGS = [1, 2, 3, 6, 12, 24, 48, 168]
DEFAULT_ROLLING_WINDOWS = [3, 6, 12, 24, 168]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out.index.hour
    out["dayofweek"] = out.index.dayofweek
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)
    return out


def add_lag_features(df: pd.DataFrame, target=TARGET, lags=None) -> pd.DataFrame:
    out = df.copy()
    for lag in (lags or DEFAULT_LAGS):
        out[f"lag_{lag}"] = out[target].shift(lag)
    return out


def add_rolling_features(df: pd.DataFrame, target=TARGET, windows=None) -> pd.DataFrame:
    """Rolling mean/std computed strictly on PAST values (shift(1) before rolling)."""
    out = df.copy()
    shifted = out[target].shift(1)
    for window in (windows or DEFAULT_ROLLING_WINDOWS):
        out[f"roll_mean_{window}"] = shifted.rolling(window).mean()
        out[f"roll_std_{window}"] = shifted.rolling(window).std()
    return out


def make_ml_table(df: pd.DataFrame, target=TARGET) -> pd.DataFrame:
    """Build the full supervised-learning table: sensors + time + lag/rolling features."""
    out = add_time_features(df)
    out = add_lag_features(out, target=target)
    out = add_rolling_features(out, target=target)
    return out.dropna()


def feature_groups(columns) -> dict:
    """Categorise feature columns for the ablation discussion (assignment Q3)."""
    groups = {"lag": [], "rolling": [], "time": [], "indoor_sensor": [],
              "outdoor_weather": [], "other": []}
    for c in columns:
        if c.startswith("lag_"):
            groups["lag"].append(c)
        elif c.startswith("roll_"):
            groups["rolling"].append(c)
        elif c in ("hour", "dayofweek", "is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos"):
            groups["time"].append(c)
        elif c in INDOOR_SENSOR_COLS:
            groups["indoor_sensor"].append(c)
        elif c in OUTDOOR_WEATHER_COLS:
            groups["outdoor_weather"].append(c)
        else:
            groups["other"].append(c)
    return {k: v for k, v in groups.items() if v}


# ============================================================
# 7. Feature-based ML model (XGBoost, with a dependency-free fallback)
# ============================================================

try:
    from xgboost import XGBRegressor
    _HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    _HAS_XGBOOST = False


def fit_feature_model(X_train, y_train, random_state=RANDOM_STATE):
    """XGBoost is preferred; HistGradientBoostingRegressor is a dependency-free fallback."""
    if _HAS_XGBOOST:
        model = XGBRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=random_state, n_jobs=-1,
        )
    else:
        model = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.03, max_leaf_nodes=31,
            random_state=random_state,
        )
    model.fit(X_train, y_train)
    return model


def forecast_feature_model(model, X_test, index) -> pd.Series:
    return pd.Series(model.predict(X_test), index=index, name="feature_model")


def get_feature_importance(model, feature_names) -> pd.Series:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return pd.Series(dtype=float)
    return pd.Series(importances, index=feature_names).sort_values(ascending=False)


def plot_feature_importance(importance: pd.Series, top_n=20, save_path=None):
    fig, ax = plt.subplots(figsize=(9, 7))
    importance.head(top_n).sort_values().plot(kind="barh", ax=ax, color="tab:green")
    ax.set_title(f"Top {top_n} feature importances (feature-based model)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 8. Foundation model (Chronos, zero-shot) with documented fallback
# ============================================================
#
# Environment note: Chronos downloads pretrained weights from Hugging Face
# on first use. If there is no internet route to huggingface.co (e.g. a
# locked-down machine), this automatically falls back to Holt-Winters
# exponential smoothing and prints a warning. THE FALLBACK IS NOT A
# FOUNDATION MODEL -- if it triggers, say so explicitly in the report, and
# re-run somewhere with normal internet access (e.g. Colab) for a genuine
# result. Install with: pip install chronos-forecasting torch

def _foundation_fallback(y_train: pd.Series, horizon: int, index, reason: str) -> pd.Series:
    # Printed with a highly visible banner using print() rather than
    # warnings.warn(), because this script sets
    # warnings.filterwarnings("ignore") globally to silence noisy
    # statsmodels convergence warnings -- which would ALSO silence this
    # message if it were a plain warning. print() cannot be silently
    # filtered, so this notice is guaranteed to appear in the console.
    print("\n" + "!" * 70)
    print("FOUNDATION MODEL FALLBACK TRIGGERED")
    print(f"Reason: {reason}")
    print("Chronos was NOT used. Falling back to Holt-Winters exponential")
    print("smoothing instead. THIS IS NOT A FOUNDATION-MODEL FORECAST.")
    print("The 'foundation_model' row in your results is Holt-Winters, not")
    print("Chronos. State this explicitly in your report, and re-run with")
    print("chronos-forecasting + torch installed, on a machine with internet")
    print("access to huggingface.co, to get a genuine Chronos result.")
    print("!" * 70 + "\n")

    model = ExponentialSmoothing(
        y_train, trend="add", damped_trend=True, seasonal="add",
        seasonal_periods=DAILY_PERIOD,
    ).fit()
    pred = model.forecast(horizon)
    pred.index = index
    return pred.rename("foundation_model_fallback_holtwinters")


def forecast_foundation_model(y_train: pd.Series, horizon: int, index,
                               model_name="amazon/chronos-t5-small",
                               num_samples: int = 20) -> pd.Series:
    """Zero-shot forecast with Chronos (target-only, no covariates); median of samples."""
    try:
        import torch
        from chronos import ChronosPipeline
    except ImportError as exc:
        return _foundation_fallback(
            y_train, horizon, index,
            reason=f"chronos-forecasting/torch not installed ({exc}). "
                   f"Install with `pip install chronos-forecasting torch`.",
        )

    try:
        print("Loading Chronos pretrained weights "
              f"({model_name}) from Hugging Face Hub...")
        pipeline = ChronosPipeline.from_pretrained(
            model_name, device_map="cpu", dtype=torch.float32,
        )
        print("Chronos loaded successfully. Running zero-shot forecast...")
        context_tensor = torch.tensor(y_train.values, dtype=torch.float32)
        samples = pipeline.predict(
            context_tensor, prediction_length=horizon, num_samples=num_samples,
        )
        median = np.median(samples[0].numpy(), axis=0)
        print("Chronos forecast complete (this IS a real foundation-model result).")
        return pd.Series(median, index=index, name="foundation_model")
    except Exception as exc:  # noqa: BLE001
        return _foundation_fallback(
            y_train, horizon, index,
            reason=f"Chronos failed to load/run ({exc}); usually a network "
                   f"issue (no route to huggingface.co).",
        )


# ============================================================
# 9. Evaluation metrics
# ============================================================

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def bias(y_true, y_pred) -> float:
    """Mean signed error: positive => model over-forecasts on average."""
    return float(np.mean(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)))


def mase(y_true, y_pred, y_train, seasonality: int = 24) -> float:
    """
    Mean Absolute Scaled Error (Hyndman & Koehler, 2006): MAE scaled by the
    in-sample MAE of a seasonal-naive forecast. MASE < 1 => beats seasonal
    naive in-sample. Scale-free, so comparable across models/series.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = pd.Series(y_train).astype(float)

    seasonal_errors = np.abs(y_train.iloc[seasonality:].values - y_train.iloc[:-seasonality].values)
    scale = seasonal_errors.mean()
    if scale == 0 or np.isnan(scale):
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def evaluate_forecast(name: str, y_true: pd.Series, y_pred: pd.Series,
                       y_train: pd.Series, seasonality: int = 24) -> dict:
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred).reindex(y_true.index).astype(float)
    valid = y_true.notna() & y_pred.notna()
    y_true_v, y_pred_v = y_true.loc[valid], y_pred.loc[valid]

    return {
        "model": name, "n_points": int(valid.sum()),
        "MAE": mae(y_true_v, y_pred_v), "RMSE": rmse(y_true_v, y_pred_v),
        "MASE": mase(y_true_v, y_pred_v, y_train, seasonality=seasonality),
        "Bias": bias(y_true_v, y_pred_v),
    }


def evaluate_all(forecasts: dict, test: pd.Series, train: pd.Series,
                  seasonality: int = 24) -> pd.DataFrame:
    rows = [evaluate_forecast(name, test, pred, train, seasonality=seasonality)
            for name, pred in forecasts.items()]
    return pd.DataFrame(rows).sort_values("MASE").reset_index(drop=True)


def compare_to_best_benchmark(results_df: pd.DataFrame, benchmark_names: list) -> pd.DataFrame:
    """Add each model's % RMSE improvement over the strongest benchmark (Part 8)."""
    bench = results_df[results_df["model"].isin(benchmark_names)]
    if bench.empty:
        return results_df
    best_bench_rmse = bench["RMSE"].min()
    best_bench_name = bench.loc[bench["RMSE"].idxmin(), "model"]

    out = results_df.copy()
    out["best_benchmark"] = best_bench_name
    out["pct_improvement_vs_best_benchmark"] = (
        (best_bench_rmse - out["RMSE"]) / best_bench_rmse * 100
    ).round(2)
    return out


# ============================================================
# 10. Forecast / error plots
# ============================================================

def plot_forecasts(train, test, forecast_df, sarimax_ci=None, context_days=14, save_path=None):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    train.tail(context_days * 24).plot(ax=ax, label="Training data", linewidth=1.2, color="tab:gray")
    test.plot(ax=ax, label="Actual (test)", linewidth=2.2, color="black")

    for col in forecast_df.columns:
        if col != "actual":
            forecast_df[col].plot(ax=ax, label=col, alpha=0.85, linewidth=1.3)

    if sarimax_ci is not None:
        ax.fill_between(sarimax_ci.index, sarimax_ci.iloc[:, 0], sarimax_ci.iloc[:, 1],
                         alpha=0.15, color="tab:orange", label="SARIMAX 95% CI")

    ax.set_title("Appliance energy: 24h-ahead forecasts vs. actual (test period)")
    ax.set_ylabel("Appliance energy use (Wh)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_error_diagnostics(forecast_df, model_cols, save_path=None):
    """Boxplot of forecast errors by model + error-over-time line plot."""
    errors = pd.DataFrame({col: forecast_df[col] - forecast_df["actual"] for col in model_cols})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    errors.plot(kind="box", ax=axes[0])
    axes[0].axhline(0, color="red", linewidth=0.8, linestyle="--")
    axes[0].set_title("Forecast error distribution by model")
    axes[0].set_ylabel("Error (forecast - actual)")
    axes[0].tick_params(axis="x", rotation=45)

    errors.plot(ax=axes[1], alpha=0.7, linewidth=1.0)
    axes[1].axhline(0, color="red", linewidth=0.8, linestyle="--")
    axes[1].set_title("Forecast error over the test period")
    axes[1].set_ylabel("Error (forecast - actual)")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison_bar(results_df, metric="RMSE", save_path=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = results_df.sort_values(metric)
    ax.bar(ordered["model"], ordered[metric], color="tab:purple", alpha=0.8)
    ax.set_title(f"Model comparison: {metric}")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 11. Main pipeline
# ============================================================

def run_pipeline(run_sarima_grid_search: bool = True,
                  sarima_search_window: int = 24 * 90,
                  sarima_p_range=None, sarima_d_range=None, sarima_q_range=None,
                  run_foundation_model: bool = True) -> dict:
    """Run the complete pipeline end to end and return key artefacts."""
    artefacts = {}

    # ---- Part 1: data loading, EDA, stationarity ----
    print("=" * 70); print("PART 1: Data loading and preparation"); print("=" * 70)

    hourly = load_hourly_data()
    y = hourly[TARGET]

    plot_series_overview(y, "Appliance energy use", FIGURE_DIR / "appliances_overview.png")
    plot_seasonal_decomposition(y, period=DAILY_PERIOD, save_path=FIGURE_DIR / "seasonal_decomposition.png")

    print("\n" + "=" * 70); print("Stationarity analysis"); print("=" * 70)
    stationarity_results = run_stationarity_analysis(y)
    stationarity_results.to_csv(METRICS_DIR / "stationarity_tests.csv", index=False)
    artefacts["stationarity_results"] = stationarity_results

    # ---- Part 2: train/test split ----
    train, test = train_test_split_series(y, TEST_STEPS)
    horizon = len(test)
    print(f"\nTrain: {train.index.min()} -> {train.index.max()} ({len(train)} obs)")
    print(f"Test : {test.index.min()} -> {test.index.max()} ({len(test)} obs)")

    forecasts = {}

    # ---- Part 3: benchmarks ----
    print("\n" + "=" * 70); print("PART 3: Benchmark models"); print("=" * 70)
    forecasts.update(all_benchmark_forecasts(train, horizon, test.index))

    # ---- Part 4: SARIMAX ----
    print("\n" + "=" * 70); print("PART 4: SARIMAX"); print("=" * 70)

    exog_cols = [c for c in OUTDOOR_WEATHER_COLS if c in hourly.columns]
    X = hourly[exog_cols] if exog_cols else None
    X_train = X.iloc[:-TEST_STEPS] if X is not None else None
    X_test = X.iloc[-TEST_STEPS:] if X is not None else None

    if run_sarima_grid_search:
        grid_results = grid_search_sarima_order(
            train, search_window=sarima_search_window,
            p_range=sarima_p_range, d_range=sarima_d_range, q_range=sarima_q_range,
        )
        grid_results.to_csv(METRICS_DIR / "sarima_grid_search.csv", index=False)
        best = grid_results.dropna(subset=["AIC"]).iloc[0]
        best_order = (int(best["p"]), int(best["d"]), int(best["q"]))
    else:
        best_order = (1, 0, 1)

    print(f"\nSelected SARIMA order: {best_order}, seasonal order: {SARIMA_SEASONAL_ORDER}")

    sarimax_fit = fit_sarimax(train, order=best_order, X_train=X_train)
    print(sarimax_fit.summary().tables[0])

    sarimax_pred, sarimax_ci = forecast_sarimax(sarimax_fit, horizon, test.index, X_test=X_test)
    forecasts["sarimax"] = sarimax_pred
    artefacts["sarimax_ci"] = sarimax_ci

    diag = sarimax_residual_diagnostics(sarimax_fit)
    print(f"\nLjung-Box p-value (lag 48): {diag['ljung_box_pvalue']:.4g} "
          f"({'significant residual autocorrelation' if diag['ljung_box_pvalue'] < 0.05 else 'no significant residual autocorrelation'})")

    plot_acf_pacf(diag["residuals"], lags=48, title_prefix="SARIMAX residuals -",
                  save_path=FIGURE_DIR / "sarimax_residual_acf_pacf.png")
    plot_residual_histogram(diag["residuals"], save_path=FIGURE_DIR / "sarimax_residual_hist.png")
    pd.DataFrame([diag]).drop(columns=["residuals"]).to_csv(
        METRICS_DIR / "sarimax_residual_diagnostics.csv", index=False)

    # ---- Parts 5 & 6: covariates + feature-based model ----
    print("\n" + "=" * 70); print("PARTS 5 & 6: Feature engineering + feature-based model"); print("=" * 70)

    ml_data = make_ml_table(hourly, target=TARGET)
    ml_train = ml_data.iloc[:-TEST_STEPS]
    ml_test = ml_data.iloc[-TEST_STEPS:]
    feature_cols = [c for c in ml_data.columns if c != TARGET]

    feature_model = fit_feature_model(ml_train[feature_cols], ml_train[TARGET])
    feature_pred = forecast_feature_model(feature_model, ml_test[feature_cols], ml_test.index)
    forecasts["feature_model"] = feature_pred.reindex(test.index)

    importance = get_feature_importance(feature_model, feature_cols)
    importance.to_csv(METRICS_DIR / "feature_importance.csv")
    plot_feature_importance(importance, save_path=FIGURE_DIR / "feature_importance.png")
    artefacts["feature_groups"] = feature_groups(feature_cols)

    # ---- Part 7: foundation model ----
    print("\n" + "=" * 70); print("PART 7: Foundation model (Chronos, zero-shot)"); print("=" * 70)

    if run_foundation_model:
        foundation_pred = forecast_foundation_model(train, horizon, test.index)
    else:
        foundation_pred = pd.Series(np.nan, index=test.index, name="foundation_model")

    foundation_model_source = (
        "chronos" if foundation_pred.name == "foundation_model" else
        "holtwinters_fallback" if "fallback" in str(foundation_pred.name) else
        "skipped"
    )
    print(f"\nfoundation_model source used: {foundation_model_source}")
    pd.DataFrame([{"foundation_model_source": foundation_model_source}]).to_csv(
        METRICS_DIR / "foundation_model_source.csv", index=False
    )
    artefacts["foundation_model_source"] = foundation_model_source

    forecasts["foundation_model"] = foundation_pred.reindex(test.index)

    # ---- Part 8: evaluation ----
    print("\n" + "=" * 70); print("PART 8: Evaluation"); print("=" * 70)

    results_df = evaluate_all(forecasts, test, train, seasonality=DAILY_PERIOD)
    benchmark_names = ["mean", "naive", "seasonal_naive_daily", "seasonal_naive_weekly", "drift"]
    results_df = compare_to_best_benchmark(results_df, benchmark_names)

    print("\nModel comparison (sorted by MASE):")
    print(results_df.round(3).to_string(index=False))

    forecast_df = pd.DataFrame({"actual": test})
    for name, pred in forecasts.items():
        forecast_df[name] = pred.reindex(test.index)

    forecast_df.to_csv(FORECAST_DIR / "all_forecasts.csv")
    results_df.to_csv(METRICS_DIR / "model_comparison.csv", index=False)

    plot_forecasts(train, test, forecast_df, sarimax_ci=sarimax_ci,
                   save_path=FIGURE_DIR / "forecast_comparison.png")
    plot_error_diagnostics(forecast_df, model_cols=list(forecasts.keys()),
                           save_path=FIGURE_DIR / "error_diagnostics.png")
    plot_model_comparison_bar(results_df, metric="RMSE",
                              save_path=FIGURE_DIR / "model_comparison_rmse.png")

    print("\nSaved outputs to:", OUTPUT_DIR)

    artefacts.update({
        "hourly": hourly, "train": train, "test": test, "forecasts": forecasts,
        "forecast_df": forecast_df, "results_df": results_df,
        "sarimax_fit": sarimax_fit, "feature_model": feature_model,
        "best_sarima_order": best_order, "seasonal_order": SARIMA_SEASONAL_ORDER,
    })
    return artefacts


def main():
    parser = argparse.ArgumentParser(description="Run the appliance-energy forecasting pipeline.")
    parser.add_argument("--no-sarima-search", action="store_true",
                        help="Skip the (p,d,q) grid search; use a fixed order (1,0,1) instead.")
    parser.add_argument("--sarima-search-window", type=int, default=24 * 90,
                        help="Most-recent training hours used for the SARIMA grid search "
                             "(default: 90 days). Use a smaller value for a faster run.")
    parser.add_argument("--no-foundation-model", action="store_true",
                        help="Skip the Chronos foundation-model forecast (useful offline).")
    args = parser.parse_args()

    run_pipeline(
        run_sarima_grid_search=not args.no_sarima_search,
        sarima_search_window=args.sarima_search_window,
        run_foundation_model=not args.no_foundation_model,
    )


if __name__ == "__main__":
    main()
