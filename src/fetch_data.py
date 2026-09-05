
import requests
import pandas as pd
from datetime import datetime, timezone

from config import AQICN_API_KEY, OPENWEATHER_API_KEY, CITY_NAME, CITY_LAT, CITY_LON

def fetch_aqicn_current(
    city: str = CITY_NAME,
    token: str = AQICN_API_KEY,
    lat: float = CITY_LAT,
    lon: float = CITY_LON,
    use_geo: bool = True,
) -> dict:
    """
    Fetch the latest AQI reading. Defaults to AQICN's geo-based endpoint,
    which finds the nearest ACTIVE station to the given coordinates — the
    plain city-name endpoint can silently resolve to a dead/stale station.
    """
    if use_geo:
        url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={token}"
    else:
        url = f"https://api.waqi.info/feed/{city}/?token={token}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != "ok":
        raise ValueError(f"AQICN API error for city='{city}': {payload}")
    return payload["data"]


def parse_aqicn_response(data: dict) -> dict:
    """
    Flattens AQICN's nested response into a single-row record ready for a DataFrame.
    AQICN returns overall AQI in `aqi`, and per-pollutant sub-indices in `iaqi`
    (each pollutant is itself an index value, not a raw concentration).
    """
    iaqi = data.get("iaqi", {})

    def get_val(key):
        entry = iaqi.get(key)
        return entry.get("v") if entry else None

    time_info = data.get("time", {})
    ts_str = time_info.get("s")
    tz_offset = time_info.get("tz")

    if ts_str and tz_offset:
        timestamp = pd.Timestamp(f"{ts_str}{tz_offset}").tz_convert("UTC")
    elif ts_str:
        timestamp = pd.Timestamp(ts_str, tz="UTC")
    else:
        timestamp = pd.Timestamp.now(tz="UTC")

    record = {
        "timestamp": timestamp,
                "city": CITY_NAME,
        "aqi": data.get("aqi"),
        "pm25": get_val("pm25"),
        "pm10": get_val("pm10"),
        "o3": get_val("o3"),
        "no2": get_val("no2"),
        "so2": get_val("so2"),
        "co": get_val("co"),
        "temp": get_val("t"),
        "humidity": get_val("h"),
        "pressure": get_val("p"),
        "wind_speed": get_val("w"),
    }
    return record


def fetch_and_parse_current(city: str = CITY_NAME, token: str = AQICN_API_KEY) -> pd.DataFrame:
    """Convenience wrapper: fetch + parse in one call, returns a single-row DataFrame."""
    raw = fetch_aqicn_current(city, token)
    record = parse_aqicn_response(raw)
    return pd.DataFrame([record])


def fetch_openweather_history(
    lat: float = CITY_LAT,
    lon: float = CITY_LON,
    start_ts: int = None,
    end_ts: int = None,
    token: str = OPENWEATHER_API_KEY,
) -> pd.DataFrame:
    """
    Fetch historical air pollution data from OpenWeather (Unix timestamps, UTC).
    Used for backfill since AQICN doesn't offer free historical access.
    """
    url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
    params = {"lat": lat, "lon": lon, "start": start_ts, "end": end_ts, "appid": token}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    payload = resp.json()

    records = []
    for item in payload.get("list", []):
        row = {
            "timestamp": pd.to_datetime(item["dt"], unit="s", utc=True),
            "aqi_ow_scale": item["main"]["aqi"],  # OpenWeather uses a 1-5 scale, NOT US AQI
            **item["components"],  # co, no, no2, o3, so2, pm2_5, pm10, nh3 (raw µg/m³)
        }
        records.append(row)
    return pd.DataFrame(records)
