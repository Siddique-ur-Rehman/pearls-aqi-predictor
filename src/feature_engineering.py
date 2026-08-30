
import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Adds hour/day/month/day-of-week + cyclical encodings of hour."""
    df = df.copy()
    ts = pd.to_datetime(df[ts_col])
    df["hour"] = ts.dt.hour
    df["day"] = ts.dt.day
    df["month"] = ts.dt.month
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_derived_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """
    Adds lag features, rolling stats, and change rates.
    IMPORTANT: df must be sorted by timestamp ascending before calling this,
    and should ideally contain one city only (call per-city if multi-city).
    """
    df = df.sort_values(ts_col).reset_index(drop=True).copy()

    # Lag features (past AQI values)
    df["aqi_lag_1h"] = df["aqi"].shift(1)
    df["aqi_lag_3h"] = df["aqi"].shift(3)
    df["aqi_lag_6h"] = df["aqi"].shift(6)
    df["aqi_lag_24h"] = df["aqi"].shift(24)

    # Change rate
    df["aqi_change_rate"] = df["aqi"].diff()
    df["aqi_change_rate_3h"] = df["aqi"].diff(3)

    # Rolling stats (trailing windows only — no future leakage)
    df["aqi_rolling_mean_6h"] = df["aqi"].rolling(window=6, min_periods=1).mean()
    df["aqi_rolling_std_6h"] = df["aqi"].rolling(window=6, min_periods=1).std()
    df["aqi_rolling_mean_24h"] = df["aqi"].rolling(window=24, min_periods=1).mean()

    # Pollutant lags (pm25 is usually the dominant driver of AQI)
    if "pm25" in df.columns:
        df["pm25_lag_1h"] = df["pm25"].shift(1)
        df["pm25_rolling_mean_6h"] = df["pm25"].rolling(window=6, min_periods=1).mean()

    return df


def add_weather_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Weather features known to correlate with pollutant buildup/dispersion."""
    df = df.copy()
    if "wind_speed" in df.columns:
        # Calm air (< 1.5 m/s) traps pollutants near the surface
        df["is_calm_wind"] = (df["wind_speed"] < 1.5).astype(int)
    if "temp" in df.columns and "humidity" in df.columns:
        df["temp_humidity_interaction"] = df["temp"] * df["humidity"]
    return df


def add_targets(df: pd.DataFrame, horizons=(24, 48, 72)) -> pd.DataFrame:
    """
    Adds forward-shifted target columns for each forecast horizon.
    Rows near the end of the dataset will have NaN targets (correctly dropped later —
    we simply don't have the future yet for those timestamps).
    """
    df = df.copy()
    for h in horizons:
        df[f"target_{h}h"] = df["aqi"].shift(-h)
    return df


def build_feature_pipeline(raw_df: pd.DataFrame, add_target_cols: bool = False, horizons=(24, 48, 72)) -> pd.DataFrame:
    """
    Full feature engineering pipeline, single entry point used by both
    the hourly feature pipeline and (minus targets) the inference path.
    """
    df = add_time_features(raw_df)
    df = add_derived_features(df)
    df = add_weather_interaction_features(df)
    if add_target_cols:
        df = add_targets(df, horizons=horizons)
    return df


FEATURE_COLUMNS = [
    "hour_sin", "hour_cos", "month_sin", "month_cos", "day_of_week", "is_weekend",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_24h",
    "aqi_change_rate", "aqi_change_rate_3h",
    "aqi_rolling_mean_6h", "aqi_rolling_std_6h", "aqi_rolling_mean_24h",
    "pm25", "pm25_lag_1h", "pm25_rolling_mean_6h",
    "pm10", "o3", "no2", "so2", "co",
    "temp", "humidity", "pressure", "wind_speed",
    "is_calm_wind", "temp_humidity_interaction",
]
