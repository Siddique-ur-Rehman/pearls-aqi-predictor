<div align="center">

# 🌫️ Pearls AQI Predictor

**3-day Air Quality Index forecasting on a 100% serverless ML pipeline**

[![Live App](https://img.shields.io/badge/🚀_Live_App-Streamlit-FF4B4B?style=for-the-badge)](https://pearls-aqi-predictor-l7pati8mpttgstvz3mfsu6.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](#)

[![Scikit--learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Hopsworks](https://img.shields.io/badge/Hopsworks-1EB182?style=flat-square)](https://www.hopsworks.ai/)
[![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![SHAP](https://img.shields.io/badge/SHAP-8A2BE2?style=flat-square)](https://shap.readthedocs.io/)

</div>

---

Predicts Air Quality Index (AQI) 3 days ahead for Peshawar using a fully serverless
ML pipeline — live data ingestion, automated training, and a public dashboard,
all running without a dedicated server.

<div align="center">

### 🔗 Quick Links

[![Open Live Dashboard](https://img.shields.io/badge/Open_Live_Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://pearls-aqi-predictor-l7pati8mpttgstvz3mfsu6.streamlit.app/)
[![AQICN API](https://img.shields.io/badge/AQICN_API-1E90FF?style=for-the-badge)](https://aqicn.org/api/)
[![OpenWeather API](https://img.shields.io/badge/OpenWeather_API-EB6E4B?style=for-the-badge)](https://openweathermap.org/api)
[![Hopsworks](https://img.shields.io/badge/Hopsworks_Console-1EB182?style=for-the-badge)](https://app.hopsworks.ai)

</div>

---

## Architecture

```
AQICN API ──hourly──▶ Feature Pipeline ──▶ Hopsworks Feature Store
                                                     │
                                          daily training pipeline
                                                     │
                                                     ▼
                                         Hopsworks Model Registry
                                                     │
                                    ┌────────────────┴────────────────┐
                                    ▼                                 ▼
                          Streamlit Dashboard                  Flask API
                       (forecast, alerts, SHAP)          (/predict /history /explain)
```

Everything runs on free-tier infrastructure with no server of your own:

| Layer | Service |
|---|---|
| Automation (hourly + daily jobs) | [GitHub Actions](https://github.com/features/actions) |
| Feature Store & Model Registry | [Hopsworks](https://www.hopsworks.ai/) |
| Dashboard hosting | [Streamlit Community Cloud](https://streamlit.io/cloud) |
| Live weather/pollutant data | [AQICN](https://aqicn.org/api/) |
| Historical backfill data | [OpenWeather](https://openweathermap.org/api) |

---

## 1. Setup

### 1.1 Clone & install
```bash
git clone <your-repo-url>
cd pearls-aqi-predictor
pip install -r requirements.txt
```

### 1.2 Get API keys (all free tier)

| Service | Get a key | Used for |
|---|---|---|
| AQICN | [aqicn.org/data-platform/token](https://aqicn.org/data-platform/token/) | live hourly readings |
| OpenWeather | [openweathermap.org/api](https://openweathermap.org/api) | historical backfill |
| Hopsworks | [app.hopsworks.ai](https://app.hopsworks.ai) | feature store + model registry |

### 1.3 Configure environment
```bash
cp .env.example .env
# then edit .env and paste in your keys
```

> **Note on Hopsworks host:** Hopsworks clusters live on region-specific hostnames
> (e.g. `eu-west.cloud.hopsworks.ai`), not a single fixed URL. Check your project's
> URL bar in the browser and set `HOPSWORKS_HOST` accordingly — the client can
> otherwise fall back to a broken default host and fail with a DNS error.

---

## 2. Local development workflow (no Hopsworks needed yet)

The feature store layer (`src/feature_store.py`) has a **local CSV backend** so you
can build and test the entire pipeline before setting up Hopsworks.

```bash
cd src

# 1. Backfill historical data (needed before you can train anything)
python backfill.py --days 60          # writes to data/features.csv locally

# 2. Explore the data
jupyter notebook ../notebooks/01_eda.ipynb

# 3. Train models (Ridge + Random Forest per horizon, picks the best)
python training_pipeline.py

# 4. (Optional) Train the LSTM deep-learning models too
python train_lstm.py

# 5. Run one manual feature pipeline update (simulates the hourly job)
python feature_pipeline.py

# 6. Get a forecast
python inference.py

# 7. Launch the dashboard
cd ..
streamlit run app/streamlit_app.py

# 8. Or launch the API
python app/flask_api.py
# curl http://localhost:5000/predict?city=Peshawar
```

Run the test suite any time with:
```bash
python -m pytest tests/ -v
# or, if pytest isn't installed:
python tests/test_pipeline.py
```

**On Windows:** installing the `hopsworks` package locally can fail with a
`twofish` build error requiring Microsoft C++ Build Tools. You don't need
`hopsworks` installed locally at all — keep `FEATURE_STORE_BACKEND=local`
for development, and let GitHub Actions (Linux) install it for automation.

---

## 3. Moving to production (Hopsworks + GitHub Actions)

### 3.1 Switch the backend
```bash
export FEATURE_STORE_BACKEND=hopsworks
```
The same code now reads/writes real Hopsworks Feature Groups and pushes trained
models to the Hopsworks Model Registry instead of local files.

### 3.2 Automate with GitHub Actions
1. Push this repo to GitHub.
2. In repo **Settings → Secrets and variables → Actions**, add:
   - `AQICN_API_KEY`
   - `OPENWEATHER_API_KEY`
   - `HOPSWORKS_API_KEY`
   - `HOPSWORKS_PROJECT_NAME`
   - `HOPSWORKS_HOST`
3. Three workflows in `.github/workflows/` handle automation:
   - `backfill.yml` — manual trigger, loads historical data into Hopsworks
   - `feature_pipeline.yml` — every hour
   - `training_pipeline.yml` — daily at 02:00 UTC
4. Trigger any of them manually from the **Actions** tab (`workflow_dispatch`) —
   run `backfill` once first, then `feature_pipeline`, then `training_pipeline`.

### 3.3 Deploy the dashboard
1. Push to GitHub (if not already).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo, set the
   main file to `app/streamlit_app.py`.
3. In the app's **Secrets** panel (TOML format), add:
   ```toml
   AQICN_API_KEY = "..."
   HOPSWORKS_API_KEY = "..."
   HOPSWORKS_PROJECT_NAME = "..."
   HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"
   FEATURE_STORE_BACKEND = "hopsworks"
   ```
4. Deploy — it redeploys automatically on every push to `main`.

> **Note:** Streamlit Cloud's Secrets panel populates `st.secrets`, not `os.environ`.
> `app/streamlit_app.py` bridges every secret into the environment at startup so
> the rest of the codebase (which reads credentials via `os.getenv()`) works
> unchanged on both local and cloud deployments.

---

## 4. Project layout

```
pearls-aqi-predictor/
├── .github/workflows/
│   ├── backfill.yml               # manual: load historical data
│   ├── feature_pipeline.yml       # hourly automation
│   └── training_pipeline.yml      # daily automation
├── src/
│   ├── config.py                  # env vars, city, paths
│   ├── fetch_data.py               # AQICN + OpenWeather API calls
│   ├── feature_engineering.py      # time/lag/rolling/derived features
│   ├── feature_store.py            # Hopsworks + local CSV dual backend
│   ├── feature_pipeline.py         # hourly job entry point
│   ├── backfill.py                 # historical data + PM2.5→AQI conversion
│   ├── training_pipeline.py        # Ridge/RF training, eval, model registry
│   ├── train_lstm.py               # deep-learning branch (TensorFlow)
│   ├── inference.py                 # loads models (local or Hopsworks), forecasts
│   └── explain.py                   # SHAP feature importance
├── app/
│   ├── streamlit_app.py            # dashboard (deployed on Streamlit Cloud)
│   └── flask_api.py                # /predict /history /explain endpoints
├── notebooks/
│   └── 01_eda.ipynb                # exploratory data analysis
├── tests/
│   └── test_pipeline.py            # unit tests (leakage guards, EPA breakpoints, etc.)
├── requirements.txt
├── .env.example
└── README.md
```

---

## 5. Design notes & things worth knowing

- **Forecasting approach:** direct multi-horizon (separate model per 24h/48h/72h) rather
  than recursive — less error compounding, cleaner to evaluate.
- **Always compares against a naive persistence baseline** ("tomorrow = today").
  `training_pipeline.py` warns if a trained model fails to beat it.
- **Time-based train/test split only** — never shuffled, since random shuffling leaks
  future information into training for time series data.
- **Same feature engineering code path** is used by the hourly pipeline, the training
  pipeline, and inference (`feature_engineering.build_feature_pipeline`) — this avoids
  training/serving skew.
- **PM2.5→AQI conversion for backfilled data** uses the official [EPA breakpoint table](https://www.airnow.gov/aqi/aqi-basics/),
  verified against known reference points in `tests/test_pipeline.py`.
- **All-NaN feature columns are coerced to typed `float64` before writing to Hopsworks**
  — on a group's first insert, lag/rolling features with no history yet are entirely
  `NaN`, and Hopsworks rejects an untyped "null" column outright.
- **Feature group writes use `wait_for_job=False`** — Hopsworks' free-tier offline
  materialization job can hang during its own Spark session shutdown even after the
  actual data write succeeds; not blocking on that avoids a false failure.
- **Model loading is backend-aware** (`inference.py`) — reads local `.joblib` files
  during local development, or downloads from the Hopsworks Model Registry in
  production, since files saved on an ephemeral GitHub Actions runner don't persist.

## 6. Known limitations to address as you extend this

- OpenWeather's historical endpoint doesn't provide temp/humidity/wind — those
  columns are `NaN` for backfilled-only rows until live AQICN data fills them in
  going forward.
- The LSTM module and SHAP explanations require `tensorflow` and `shap` respectively.
- Multi-city support in the dashboard is present in the UI but each city needs its own
  backfilled history before its forecast will work.
- Early model accuracy is weak (near-baseline RMSE) until enough live weather data
  accumulates via the hourly pipeline — retrain periodically as history grows.

---

<div align="center">

**Built by [Siddique ur Rehman](#)** · Data Science Instructor & Practitioner, Peshawar

[![Live App](https://img.shields.io/badge/🚀_Try_the_Live_App-FF4B4B?style=for-the-badge)](https://pearls-aqi-predictor-l7pati8mpttgstvz3mfsu6.streamlit.app/)

</div>
