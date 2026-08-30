"""
Unit tests. Run with: python -m pytest tests/ -v
(or, if pytest isn't installed, run this file directly: python tests/test_pipeline.py)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from feature_engineering import (
    add_time_features, add_derived_features, add_targets, build_feature_pipeline
)
from backfill import pm25_to_aqi
from inference import aqi_category


def make_sample_df(n_hours=100):
    timestamps = pd.date_range("2026-01-01", periods=n_hours, freq="h")
    return pd.DataFrame({
        "timestamp": timestamps,
        "city": "TestCity",
        "aqi": np.linspace(50, 150, n_hours),
        "pm25": np.linspace(40, 120, n_hours),
        "pm10": np.linspace(30, 90, n_hours),
        "o3": np.random.uniform(5, 30, n_hours),
        "no2": np.random.uniform(5, 20, n_hours),
        "so2": np.random.uniform(1, 10, n_hours),
        "co": np.random.uniform(1, 8, n_hours),
        "temp": np.random.uniform(20, 35, n_hours),
        "humidity": np.random.uniform(30, 70, n_hours),
        "pressure": np.random.uniform(1000, 1015, n_hours),
        "wind_speed": np.random.uniform(0, 5, n_hours),
    })


def test_time_features_no_nans():
    df = add_time_features(make_sample_df())
    assert df["hour"].between(0, 23).all()
    assert df["hour_sin"].between(-1, 1).all()
    assert not df["hour"].isna().any()


def test_lag_features_align_correctly():
    df = add_derived_features(make_sample_df())
    # row i's lag_1h should equal row i-1's aqi
    assert df["aqi_lag_1h"].iloc[5] == df["aqi"].iloc[4]
    assert df["aqi_lag_24h"].iloc[30] == df["aqi"].iloc[6]
    assert pd.isna(df["aqi_lag_1h"].iloc[0])  # no prior data for first row


def test_no_future_leakage_in_lags():
    df = add_derived_features(make_sample_df())
    # change_rate at row i must only depend on rows <= i
    manual_diff = df["aqi"].iloc[10] - df["aqi"].iloc[9]
    assert abs(df["aqi_change_rate"].iloc[10] - manual_diff) < 1e-9


def test_targets_shift_forward_correctly():
    df = add_targets(make_sample_df(), horizons=(24, 48, 72))
    assert df["target_24h"].iloc[0] == df["aqi"].iloc[24]
    assert pd.isna(df["target_72h"].iloc[-1])  # last rows can't have future targets


def test_full_pipeline_shape_consistent():
    raw = make_sample_df(200)
    out = build_feature_pipeline(raw, add_target_cols=True, horizons=(24, 48, 72))
    assert len(out) == len(raw)
    assert "target_24h" in out.columns
    assert "aqi_rolling_mean_6h" in out.columns


def test_pm25_to_aqi_matches_epa_breakpoints():
    assert pm25_to_aqi(0) == 0
    assert pm25_to_aqi(12.0) == 50
    assert pm25_to_aqi(35.4) == 100
    assert pm25_to_aqi(None) is None


def test_aqi_category_boundaries():
    assert aqi_category(25)[0] == "Good"
    assert aqi_category(75)[0] == "Moderate"
    assert aqi_category(125)[0] == "Unhealthy for Sensitive Groups"
    assert aqi_category(175)[0] == "Unhealthy"
    assert aqi_category(250)[0] == "Very Unhealthy"
    assert aqi_category(350)[0] == "Hazardous"


def test_no_random_shuffle_split_used():
    """Regression guard: time_based_split must preserve chronological order."""
    from training_pipeline import time_based_split
    df = make_sample_df(100)
    train, test = time_based_split(df, test_frac=0.2)
    assert train["timestamp"].max() <= test["timestamp"].min()
    assert len(test) == 20


ALL_TESTS = [
    test_time_features_no_nans,
    test_lag_features_align_correctly,
    test_no_future_leakage_in_lags,
    test_targets_shift_forward_correctly,
    test_full_pipeline_shape_consistent,
    test_pm25_to_aqi_matches_epa_breakpoints,
    test_aqi_category_boundaries,
    test_no_random_shuffle_split_used,
]

if __name__ == "__main__":
    passed, failed = 0, 0
    for test_fn in ALL_TESTS:
        try:
            test_fn()
            print(f"✅ {test_fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"❌ {test_fn.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
