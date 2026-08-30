"""
Flask API — thin layer exposing AQI predictions over HTTP.
Satisfies the project's Flask requirement and lets other clients (mobile
apps, other dashboards) consume forecasts without importing Python code directly.

Run locally with: python app/flask_api.py
Then: curl http://localhost:5000/predict?city=Peshawar
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from flask import Flask, jsonify, request

from config import CITY_NAME
from feature_store import read_features, DEFAULT_BACKEND
from inference import predict_next_3_days, aqi_category

app = Flask(__name__)


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "pearls-aqi-predictor-api"})


@app.route("/predict", methods=["GET"])
def predict():
    city = request.args.get("city", CITY_NAME)
    try:
        result = predict_next_3_days(city=city, backend=DEFAULT_BACKEND)
        for horizon, data in result["forecast"].items():
            cat, severity = aqi_category(data["aqi"])
            data["category"] = cat
            data["severity"] = severity
        current_cat, current_severity = aqi_category(result["current_aqi"])
        result["current_category"] = current_cat
        result["current_severity"] = current_severity
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/history", methods=["GET"])
def history():
    city = request.args.get("city", CITY_NAME)
    limit = int(request.args.get("limit", 168))  # default: last 7 days of hourly data
    df = read_features(backend=DEFAULT_BACKEND)
    if df.empty:
        return jsonify({"error": "No feature data available yet"}), 404
    city_df = df[df["city"] == city].sort_values("timestamp").tail(limit)
    records = city_df[["timestamp", "aqi", "pm25", "pm10", "temp", "humidity"]].copy()
    records["timestamp"] = records["timestamp"].astype(str)
    return jsonify(records.to_dict(orient="records"))


@app.route("/explain", methods=["GET"])
def explain():
    city = request.args.get("city", CITY_NAME)
    horizon = int(request.args.get("horizon", 24))
    try:
        from explain import explain_forecast, top_n_drivers
        hist = read_features(backend=DEFAULT_BACKEND)
        hist = hist[hist["city"] == city]
        explanation = explain_forecast(horizon=horizon, city=city, background_df=hist)
        top = top_n_drivers(explanation, n=8)
        return jsonify({
            "horizon": horizon,
            "model": explanation["model_name"],
            "top_drivers": top.to_dict(orient="records"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
