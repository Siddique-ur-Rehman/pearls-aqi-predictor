
import os
import json
import glob
import joblib
import pandas as pd

from config import HORIZONS, CITY_NAME
from feature_store import get_latest_feature_row, DEFAULT_BACKEND

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_best_model_for_horizon(horizon: int, backend: str = DEFAULT_BACKEND):
    """
    Finds the trained model for a given horizon.
    - "local": looks for {name}_h{horizon}.joblib in the local models/ folder.
    - "hopsworks": downloads the model from the Hopsworks Model Registry.
    """
    if backend == "local":
        pattern = os.path.join(MODEL_DIR, f"*_h{horizon}.joblib")
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(
                f"No trained model found for horizon={horizon}h. Run training_pipeline.py first."
            )
        latest_path = max(matches, key=os.path.getmtime)
        bundle = joblib.load(latest_path)
        return bundle["model"], bundle["feature_cols"], latest_path

    elif backend == "hopsworks":
        from feature_store import _get_hopsworks_project

        project = _get_hopsworks_project()
        mr = project.get_model_registry()

        candidate_names = [f"aqi_ridge_h{horizon}", f"aqi_random_forest_h{horizon}"]
        last_err = None
        for model_name in candidate_names:
            try:
                hw_model = mr.get_model(model_name)
                model_dir = hw_model.download()
                matches = glob.glob(os.path.join(model_dir, f"*_h{horizon}.joblib"))
                if not matches:
                    continue
                bundle = joblib.load(matches[0])
                return bundle["model"], bundle["feature_cols"], matches[0]
            except Exception as e:
                last_err = e
                continue
        raise FileNotFoundError(
            f"No model found in Hopsworks registry for horizon={horizon}h. "
            f"Tried: {candidate_names}. Last error: {last_err}"
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def predict_next_3_days(city: str = CITY_NAME, backend: str = DEFAULT_BACKEND) -> dict:
    """
    Returns: {
        "city": ..., "current_aqi": ..., "current_timestamp": ...,
        "forecast": {"24h": {"aqi": ..., "model": ...}, "48h": {...}, "72h": {...}}
    }
    """
    latest_row = get_latest_feature_row(city, backend=backend)

    forecast = {}
    for horizon in HORIZONS:
        model, feature_cols, model_path = load_best_model_for_horizon(horizon, backend=backend)
        X = latest_row[feature_cols].to_frame().T
        # fillna defensively — a single missing pollutant shouldn't crash inference
        X = X.fillna(X.median(numeric_only=True)).fillna(0)
        pred = model.predict(X)[0]
        forecast[f"{horizon}h"] = {
            "aqi": round(float(pred), 1),
            "model_used": os.path.basename(model_path).split("_h")[0],
        }

    return {
        "city": city,
        "current_aqi": float(latest_row["aqi"]),
        "current_timestamp": str(latest_row["timestamp"]),
        "forecast": forecast,
    }


def aqi_category(aqi: float) -> tuple:
    """Returns (category_label, severity_level 1-6) per US EPA AQI bands."""
    if aqi <= 50:
        return "Good", 1
    elif aqi <= 100:
        return "Moderate", 2
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", 3
    elif aqi <= 200:
        return "Unhealthy", 4
    elif aqi <= 300:
        return "Very Unhealthy", 5
    else:
        return "Hazardous", 6


if __name__ == "__main__":
    result = predict_next_3_days()
    print(json.dumps(result, indent=2))
    for horizon, data in result["forecast"].items():
        cat, _ = aqi_category(data["aqi"])
        print(f"  {horizon}: AQI={data['aqi']} ({cat}) via {data['model_used']}")
