
import sys
import pandas as pd

from config import CITY_NAME, AQICN_API_KEY
from fetch_data import fetch_and_parse_current
from feature_engineering import build_feature_pipeline
from feature_store import write_features, read_features, DEFAULT_BACKEND


def run(city: str = CITY_NAME, backend: str = DEFAULT_BACKEND):
    print(f"[feature_pipeline] Fetching current AQI for {city}...")
    new_row = fetch_and_parse_current(city=city)

    # Pull recent history so lag/rolling features can be computed correctly —
    # a single row alone can't have aqi_lag_24h etc.
    print(f"[feature_pipeline] Reading recent history from '{backend}' feature store...")
    try:
        history = read_features(backend=backend)
        history = history[history["city"] == city] if not history.empty else history
    except Exception as e:
        print(f"[feature_pipeline] No prior history found ({e}); starting fresh.")
        history = pd.DataFrame()

    combined = pd.concat([history, new_row], ignore_index=True)
    combined = combined.drop_duplicates(subset=["city", "timestamp"], keep="last")
    combined = combined.sort_values("timestamp")

    print(f"[feature_pipeline] Engineering features on {len(combined)} rows of history...")
    featured = build_feature_pipeline(combined, add_target_cols=False)

    # Only write the NEW row(s) back — history rows are already stored.
    # We recompute lag features for the new row using the freshly-merged history,
    # so just take the last row (the one we just fetched).
    new_featured_row = featured.tail(1)

       print(f"[feature_pipeline] Writing latest feature row to '{backend}' feature store...")
    write_features(new_featured_row, backend=backend)

    print(f"[feature_pipeline] Done. Latest AQI={new_featured_row['aqi'].values[0]} "
          f"at {new_featured_row['timestamp'].values[0]}")

    try:
        verify_df = read_features(backend=backend)
        verify_df = verify_df[verify_df["city"] == city] if not verify_df.empty else verify_df
        if verify_df.empty:
            print("[feature_pipeline] ⚠️  VERIFY FAILED: read-back returned 0 rows after write.")
        else:
            print(f"[feature_pipeline] ✅ VERIFY: {len(verify_df)} total rows now in store, "
                  f"latest timestamp = {verify_df['timestamp'].max()}")
    except Exception as e:
        print(f"[feature_pipeline] ⚠️  VERIFY FAILED: could not read back after write ({e})")

    return new_featured_row

if __name__ == "__main__":
    if not AQICN_API_KEY:
        print("ERROR: AQICN_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)
    run()
