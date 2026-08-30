"""
Pearls AQI Predictor — Streamlit Dashboard.

Run locally with:  streamlit run app/streamlit_app.py
Deploy on Streamlit Community Cloud by connecting this repo.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config import CITY_NAME
from feature_store import read_features, DEFAULT_BACKEND
from inference import predict_next_3_days, aqi_category

st.set_page_config(page_title="Pearls AQI Predictor", page_icon="🌫️", layout="wide")


AQI_COLORS = {
    "Good": "#00e400",
    "Moderate": "#ffff00",
    "Unhealthy for Sensitive Groups": "#ff7e00",
    "Unhealthy": "#ff0000",
    "Very Unhealthy": "#8f3f97",
    "Hazardous": "#7e0023",
}


@st.cache_data(ttl=3600)  # re-fetch at most once per hour — matches pipeline cadence
def get_forecast(city: str):
    return predict_next_3_days(city=city, backend=DEFAULT_BACKEND)


@st.cache_data(ttl=3600)
def get_history(city: str):
    df = read_features(backend=DEFAULT_BACKEND)
    if df.empty:
        return df
    return df[df["city"] == city].sort_values("timestamp")


def render_header():
    st.title("🌫️ Pearls AQI Predictor")
    st.caption("3-day Air Quality Index forecast — 100% serverless ML pipeline")


def render_current_and_forecast(city: str):
    try:
        result = get_forecast(city)
    except FileNotFoundError:
        st.warning("No trained models found yet. Run `python src/training_pipeline.py` first.")
        return None
    except ValueError as e:
        st.warning(f"No feature data yet: {e}")
        return None

    current_aqi = result["current_aqi"]
    current_cat, _ = aqi_category(current_aqi)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current AQI", f"{current_aqi:.0f}", help=current_cat)

    horizons = ["24h", "48h", "72h"]
    cols = [col2, col3, col4]
    for h, col in zip(horizons, cols):
        val = result["forecast"][h]["aqi"]
        cat, _ = aqi_category(val)
        delta = val - current_aqi
        col.metric(f"+{h} Forecast", f"{val:.0f}", delta=f"{delta:+.0f}", help=cat)

    st.markdown(f"*Last updated: {result['current_timestamp']}*")
    return result


def render_forecast_chart(result: dict):
    labels = ["Now", "+24h", "+48h", "+72h"]
    values = [result["current_aqi"], result["forecast"]["24h"]["aqi"],
              result["forecast"]["48h"]["aqi"], result["forecast"]["72h"]["aqi"]]
    colors = [AQI_COLORS[aqi_category(v)[0]] for v in values]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=values, mode="lines+markers+text",
        text=[f"{v:.0f}" for v in values], textposition="top center",
        line=dict(color="#555", width=2),
        marker=dict(size=16, color=colors, line=dict(width=2, color="white")),
    ))
    fig.update_layout(
        title="3-Day AQI Forecast", yaxis_title="AQI",
        height=400, showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_alert(result: dict):
    all_vals = [result["current_aqi"]] + [v["aqi"] for v in result["forecast"].values()]
    worst = max(all_vals)
    cat, severity = aqi_category(worst)

    if severity >= 4:
        st.error(f"⚠️ **{cat}** air quality expected (AQI up to {worst:.0f}) — "
                  f"limit outdoor exposure, especially for sensitive groups.")
    elif severity == 3:
        st.warning(f"**{cat}** air quality expected (AQI up to {worst:.0f}) — "
                    f"sensitive groups should reduce prolonged outdoor exertion.")
    else:
        st.success(f"Air quality expected to stay **{cat.lower()}** over the next 3 days.")


def render_history_chart(city: str):
    hist = get_history(city)
    if hist.empty:
        st.info("No historical data yet — run the backfill script to populate this chart.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["timestamp"], y=hist["aqi"], mode="lines", name="AQI",
                              line=dict(color="#e74c3c")))
    fig.update_layout(title=f"Historical AQI — {city}", yaxis_title="AQI", height=350)
    st.plotly_chart(fig, use_container_width=True)


def render_explanation_panel(city: str):
    with st.expander("🔍 Why this prediction? (SHAP feature importance)"):
        try:
            from explain import explain_forecast, top_n_drivers
            hist = get_history(city)
            explanation = explain_forecast(horizon=24, city=city, background_df=hist)
            top = top_n_drivers(explanation, n=8)

            fig = go.Figure(go.Bar(
                x=top["shap_value"], y=top["feature"], orientation="h",
                marker_color=["#e74c3c" if v > 0 else "#3498db" for v in top["shap_value"]],
            ))
            fig.update_layout(
                title="Top features driving the +24h forecast",
                xaxis_title="Impact on predicted AQI (SHAP value)",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Red bars push the prediction higher; blue bars pull it lower.")
        except ImportError:
            st.info("Install `shap` (see requirements.txt) to enable this panel.")
        except Exception as e:
            st.info(f"Explanation unavailable yet: {e}")


def main():
    with st.sidebar:
        st.header("Settings")
        city = st.selectbox("City", [CITY_NAME, "Lahore", "Karachi", "Islamabad", "Quetta"], index=0)
        st.markdown("---")
        st.caption("Data refreshes hourly via the automated feature pipeline. "
                    "Models retrain daily.")

    render_header()
    result = render_current_and_forecast(city)

    if result:
        render_alert(result)
        render_forecast_chart(result)

    st.markdown("---")
    render_history_chart(city)

    if result:
        render_explanation_panel(city)


if __name__ == "__main__":
    main()
