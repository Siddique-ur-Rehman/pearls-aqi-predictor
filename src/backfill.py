"""
Historical Backfill — run once (or periodically) to build up training data.

Uses OpenWeather's Air Pollution History API since AQICN doesn't offer
free historical access. Converts OpenWeather's raw pollutant concentrations
into an approximate US AQI so it's consistent with the AQICN 'aqi' field
used everywhere else in the pipeline.
"""
import sys
import time
import argparse
import pandas as pd
from datetime import datetime, timedelta, timezone

from config import CITY_NAME, CITY_LAT, CITY_LON, OPENWEATHER_API_KEY
from fetch_data import fetch_openweather_history
from feature_engineering import build_feature_pipeline
from feature_store import write_features, DEFAULT_BACKEND


# --- US EPA AQI breakpoints for PM2.5 (24hr avg, µg/m³) — simplified table ---
# Used to convert OpenWeather's raw PM2.5 concentration into a comparable AQI value.
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]


def pm25_to_aqi(pm25: float) -> float:
    """Converts a raw PM2.5 concentration (µg/m³) to a US AQI value via linear interpolation."""
    if pm25 is None or pd.isna(pm25):
        return None
    pm25 = max(0.0, float(pm25))
    for c_lo, c_hi, i_lo, i_hi in PM25_BREAKPOINTS:
        if c_lo <= pm25 <= c_hi:
            return round((i_hi - i_lo) / (c_hi - c_lo) * (pm25 - c_lo) + i_lo, 1)
    return 500.0  # cap at hazardous max for anything beyond the table


def openweather_to_standard_format(ow_df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Maps OpenWeather's column names/units onto our standard schema."""
    df = ow_df.copy()
    df["city"] = city
    df["aqi"] = df["pm2_5"].apply(pm25_to_aqi)
    df["pm25"] = df["pm2_5"]
    # OpenWeather doesn't provide temp/humidity/wind in this endpoint —
    # left as NaN here; the feature pipeline's weather-interaction features
    # will simply be NaN for these historical-only rows, which is fine since
    # they're only used as auxiliary signal, not the core AQI lag features.
    for col in ["temp", "humidity", "pressure", "wind_speed"]:
        if col not in df.columns:
            df[col] = None
    df = df.rename(columns={"nh3": "nh3_raw"})  # not in our standard schema, kept as extra
    keep_cols = ["timestamp", "city", "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
                 "temp", "humidity", "pressure", "wind_speed"]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = None
    return df[keep_cols]


def backfill(days: int = 60, city: str = CITY_NAME, lat: float = CITY_LAT, lon: float = CITY_LON,
             backend: str = DEFAULT_BACKEND, chunk_days: int = 7):
    """
    Backfills `days` of history in chunks (OpenWeather's history endpoint
    can be unreliable on very large single ranges, so we chunk it).
    """
    end = datetime.now(timezone.utc)
    start_overall = end - timedelta(days=days)

    all_chunks = []
    chunk_start = start_overall
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=chunk_days), end)
        print(f"[backfill] Fetching {chunk_start.date()} to {chunk_end.date()}...")
        try:
            chunk_df = fetch_openweather_history(
                lat=lat, lon=lon,
                start_ts=int(chunk_start.timestamp()),
                end_ts=int(chunk_end.timestamp()),
                token=OPENWEATHER_API_KEY,
            )
            if not chunk_df.empty:
                all_chunks.append(chunk_df)
            time.sleep(1)  # be polite to the free-tier API
        except Exception as e:
            print(f"[backfill] WARNING: chunk failed ({e}), skipping.")
        chunk_start = chunk_end

    if not all_chunks:
        raise RuntimeError("No data fetched — check OPENWEATHER_API_KEY and city coordinates.")

    raw = pd.concat(all_chunks, ignore_index=True).drop_duplicates(subset=["timestamp"])
    standardized = openweather_to_standard_format(raw, city=city)
    print(f"[backfill] Fetched {len(standardized)} raw hourly records.")

    featured = build_feature_pipeline(standardized, add_target_cols=False)
    write_features(featured, backend=backend)
    print(f"[backfill] Wrote {len(featured)} feature rows to '{backend}' feature store.")
    return featured


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--city", type=str, default=CITY_NAME)
    args = parser.parse_args()

    if not OPENWEATHER_API_KEY:
        print("ERROR: OPENWEATHER_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    backfill(days=args.days, city=args.city)
