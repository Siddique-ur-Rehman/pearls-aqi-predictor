
import os
import pandas as pd

from config import (
    HOPSWORKS_API_KEY, HOPSWORKS_PROJECT_NAME,
    FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION,
    FEATURE_VIEW_NAME, FEATURE_VIEW_VERSION,
    FEATURES_DATA_PATH,
)

DEFAULT_BACKEND = os.getenv("FEATURE_STORE_BACKEND", "local")


def _local_write(df: pd.DataFrame, path: str = FEATURES_DATA_PATH):
    if os.path.exists(path):
        existing = pd.read_csv(path, parse_dates=["timestamp"])
        combined = pd.concat([existing, df], ignore_index=True)
        # de-dupe on (city, timestamp) in case the pipeline re-runs on overlapping hours
        combined = combined.drop_duplicates(subset=["city", "timestamp"], keep="last")
        combined = combined.sort_values("timestamp")
        combined.to_csv(path, index=False)
    else:
        df.sort_values("timestamp").to_csv(path, index=False)


def _local_read(path: str = FEATURES_DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["timestamp"])

def _get_hopsworks_project():
    import hopsworks
    host = os.getenv("HOPSWORKS_HOST")
    kwargs = {
        "api_key_value": HOPSWORKS_API_KEY,
        "project": HOPSWORKS_PROJECT_NAME or None,
    }
    if host:
        kwargs["host"] = host
    return hopsworks.login(**kwargs)

def _fix_null_dtype_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Forces all-NaN columns to float64 instead of untyped null, which Hopsworks rejects."""
    df = df.copy()
    skip_cols = {"city", "timestamp"}
    for col in df.columns:
        if col in skip_cols:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


def _hopsworks_write(df: pd.DataFrame):
    project = _get_hopsworks_project()
    fs = project.get_feature_store()
    df = _fix_null_dtype_columns(df)
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        description="Hourly AQI + weather features for AQI forecasting",
        online_enabled=True,
        time_travel_format="HUDI",
    )
    fg.insert(df, write_options={"wait_for_job": True})
    return fg


def _hopsworks_read() -> pd.DataFrame:
    project = _get_hopsworks_project()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    return fg.read()


def _hopsworks_get_or_create_feature_view(fs, fg):
    """
    Feature views snapshot a query against a feature group so training data
    generation is reproducible/versioned. Selects all columns.
    """
    try:
        return fs.get_feature_view(FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION)
    except Exception:
        query = fg.select_all()
        return fs.create_feature_view(
            name=FEATURE_VIEW_NAME,
            version=FEATURE_VIEW_VERSION,
            query=query,
        )




def write_features(df: pd.DataFrame, backend: str = DEFAULT_BACKEND):
    if backend == "local":
        _local_write(df)
    elif backend == "hopsworks":
        _hopsworks_write(df)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def read_features(backend: str = DEFAULT_BACKEND) -> pd.DataFrame:
    if backend == "local":
        return _local_read()
    elif backend == "hopsworks":
        return _hopsworks_read()
    else:
        raise ValueError(f"Unknown backend: {backend}")


def get_latest_feature_row(city: str, backend: str = DEFAULT_BACKEND) -> pd.Series:
    """Used at inference time to get the most recent feature vector for a city."""
    df = read_features(backend=backend)
    if df.empty:
        raise ValueError("No features available yet — run the feature pipeline first.")
    city_df = df[df["city"] == city].sort_values("timestamp")
    if city_df.empty:
        raise ValueError(f"No features found for city='{city}'")
    return city_df.iloc[-1]
