
import os
from dotenv import load_dotenv

load_dotenv()

# --- API keys ---
AQICN_API_KEY = os.getenv("AQICN_API_KEY", "")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")

# --- City config ---
# lat/lon needed for OpenWeather; AQICN uses a station/city name string
CITY_NAME = os.getenv("CITY_NAME", "Peshawar")
CITY_LAT = float(os.getenv("CITY_LAT", "34.0151"))
CITY_LON = float(os.getenv("CITY_LON", "71.5249"))

# --- Feature store config ---
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "")
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = "aqi_fv"
FEATURE_VIEW_VERSION = 1

# --- Forecast horizons (in hours) ---
HORIZONS = [24, 48, 72]

# --- Local paths (used before/instead of feature store, for local dev) ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw_aqi_data.csv")
FEATURES_DATA_PATH = os.path.join(DATA_DIR, "features.csv")

os.makedirs(DATA_DIR, exist_ok=True)
