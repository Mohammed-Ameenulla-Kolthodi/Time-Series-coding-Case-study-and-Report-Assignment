"""
Tests for run_pipeline.py functions.

Covers the specific checks the assignment brief calls out:
    - forecast lengths match the test period
    - MASE is zero for a perfect forecast
    - lag/rolling features do not use future target values
    - the processed feature table has no missing target values
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_pipeline as rp  # noqa: E402


def _toy_series(n=500):
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.Series(np.arange(n, dtype=float), index=idx)


def _toy_df(n=300):
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({"Appliances": np.arange(n, dtype=float)}, index=idx)


# ---------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------

def test_forecast_lengths_match_horizon():
    y_train = _toy_series()
    horizon = 24
    index = pd.date_range("2024-02-01", periods=horizon, freq="h")

    for fc in rp.all_benchmark_forecasts(y_train, horizon, index).values():
        assert len(fc) == horizon
        assert list(fc.index) == list(index)


def test_naive_forecast_repeats_last_value():
    y_train = _toy_series()
    horizon = 10
    index = pd.date_range("2024-02-01", periods=horizon, freq="h")
    fc = rp.naive_forecast(y_train, horizon, index)
    assert (fc.values == y_train.iloc[-1]).all()


def test_seasonal_naive_matches_correct_lag():
    y_train = _toy_series()
    horizon = 5
    index = pd.date_range("2024-02-01", periods=horizon, freq="h")
    seasonality = 24
    fc = rp.seasonal_naive_forecast(y_train, horizon, index, seasonality)
    assert fc.iloc[0] == y_train.iloc[-seasonality]


def test_mean_forecast_is_constant_training_mean():
    y_train = _toy_series()
    horizon = 6
    index = pd.date_range("2024-02-01", periods=horizon, freq="h")
    fc = rp.mean_forecast(y_train, horizon, index)
    assert np.allclose(fc.values, y_train.mean())


# ---------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------

def test_mase_is_zero_for_perfect_forecast():
    idx = pd.date_range("2024-01-01", periods=48, freq="h")
    y_train = pd.Series(np.sin(np.arange(200)) * 10 + 50)
    y_true = pd.Series(np.arange(48), index=idx, dtype=float)
    y_pred = y_true.copy()
    assert np.isclose(rp.mase(y_true, y_pred, y_train, seasonality=24), 0.0)


def test_rmse_mae_bias_basic_properties():
    y_true = pd.Series([10, 20, 30])
    y_pred_perfect = pd.Series([10, 20, 30])
    y_pred_over = pd.Series([12, 22, 32])  # constant +2 over-forecast

    assert np.isclose(rp.rmse(y_true, y_pred_perfect), 0.0)
    assert np.isclose(rp.mae(y_true, y_pred_perfect), 0.0)
    assert np.isclose(rp.bias(y_true, y_pred_over), 2.0)


def test_evaluate_forecast_returns_expected_keys():
    idx = pd.date_range("2024-01-01", periods=24, freq="h")
    y_train = pd.Series(np.random.rand(200) * 100)
    y_true = pd.Series(np.random.rand(24) * 100, index=idx)
    y_pred = pd.Series(np.random.rand(24) * 100, index=idx)
    result = rp.evaluate_forecast("dummy_model", y_true, y_pred, y_train)
    assert set(["model", "n_points", "MAE", "RMSE", "MASE", "Bias"]).issubset(result.keys())
    assert result["n_points"] == 24


# ---------------------------------------------------------------
# Feature engineering / leakage checks
# ---------------------------------------------------------------

def test_lag_features_do_not_leak_future_values():
    df = _toy_df()
    out = rp.add_lag_features(df, target="Appliances", lags=[1, 24])
    expected = df["Appliances"].shift(1)
    aligned = out.dropna(subset=["lag_1"])
    assert (aligned["lag_1"] == expected.loc[aligned.index]).all()


def test_rolling_features_exclude_current_observation():
    df = _toy_df()
    out = rp.add_rolling_features(df, target="Appliances", windows=[3])
    t = out.index[10]
    actual = out.loc[t, "roll_mean_3"]
    expected = df["Appliances"].shift(1).rolling(3).mean().loc[t]
    assert np.isclose(actual, expected)


def test_make_ml_table_has_no_missing_values():
    df = _toy_df()
    table = rp.make_ml_table(df, target="Appliances")
    assert table.isna().sum().sum() == 0
    assert len(table) > 0


def test_time_features_are_known_in_advance():
    df = _toy_df()
    out = rp.add_time_features(df)
    assert {"hour", "dayofweek", "is_weekend", "hour_sin", "hour_cos",
            "dow_sin", "dow_cos"}.issubset(out.columns)
    assert out["hour"].between(0, 23).all()
