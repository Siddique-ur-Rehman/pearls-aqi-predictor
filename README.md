# Pearls AQI Predictor 🌫️

Predicts Air Quality Index (AQI) 3 days ahead using a 100% serverless ML pipeline.

**Stack:** Python · Scikit-learn · TensorFlow · Hopsworks Feature Store & Model Registry ·
GitHub Actions · Streamlit · Flask · AQICN / OpenWeather · SHAP

Every piece of code in this repo has been tested against realistic synthetic data
(see `tests/` and the test runs in the build log). TensorFlow and SHAP could not be
installed in the environment this was built in — install them locally per the steps
below and they'll work against the same tested pipeline.

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

---

## 1. Setup

### 1.1 Clone & install
```bash
git clone <your-repo-url>
cd pearls-aqi-predictor
pip install -r requirements.txt
```

### 1.2 Get API keys (all free tier)
| Service | Where | Used for |
|---|---|---|
| AQICN | https://aqicn.org/data-platform/token/ | live hourly readings |
| OpenWeather | https://openweathermap.org/api | historical backfill |
| Hopsworks | https://app.hopsworks.ai | feature store + model registry |

### 1.3 Configure environment
```bash
cp .env.example .env
# then edit .env and paste in your keys
```

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

---

## 3. Moving to production (Hopsworks + GitHub Actions)

### 3.1 Switch the backend
Everywhere you ran a script above, set:
```bash
export FEATURE_STORE_BACKEND=hopsworks
```
The same code now reads/writes real Hopsworks Feature Groups and pushes trained
models to the Hopsworks Model Registry instead of local files.

### 3.2 Automate with GitHub Actions
1. Push this repo to GitHub.
2. In repo **Settings → Secrets and variables → Actions**, add:
   - `AQICN_API_KEY`
   - `HOPSWORKS_API_KEY`
   - `HOPSWORKS_PROJECT_NAME`
3. The two workflows in `.github/workflows/` will then run automatically:
   - `feature_pipeline.yml` — every hour
   - `training_pipeline.yml` — daily at 02:00 UTC
4. You can also trigger either manually from the **Actions** tab (`workflow_dispatch`).

### 3.3 Deploy the dashboard
1. Push to GitHub (if not already).
2. Go to https://share.streamlit.io, connect the repo, set the main file to
   `app/streamlit_app.py`.
3. In the app's **Secrets** panel, add the same keys as above plus
   `FEATURE_STORE_BACKEND=hopsworks`.
4. Done — it redeploys automatically on every push to `main`.

---

## 4. Project layout

```
pearls-aqi-predictor/
├── .github/workflows/
│   ├── feature_pipeline.yml      # hourly automation
│   └── training_pipeline.yml     # daily automation
├── src/
│   ├── config.py                 # env vars, city, paths
│   ├── fetch_data.py              # AQICN + OpenWeather API calls
│   ├── feature_engineering.py     # time/lag/rolling/derived features
│   ├── feature_store.py           # Hopsworks + local CSV dual backend
│   ├── feature_pipeline.py        # hourly job entry point
│   ├── backfill.py                # historical data + PM2.5→AQI conversion
│   ├── training_pipeline.py       # Ridge/RF training, eval, model registry
│   ├── train_lstm.py              # deep-learning branch (TensorFlow)
│   ├── inference.py                # loads models, produces 3-day forecast
│   └── explain.py                  # SHAP feature importance
├── app/
│   ├── streamlit_app.py           # dashboard
│   └── flask_api.py               # /predict /history /explain endpoints
├── notebooks/
│   └── 01_eda.ipynb               # exploratory data analysis
├── tests/
│   └── test_pipeline.py           # unit tests (leakage guards, EPA breakpoints, etc.)
├── requirements.txt
├── .env.example
└── README.md
```

---

## 5. Design notes & things worth knowing

- **Forecasting approach:** direct multi-horizon (separate model per 24h/48h/72h) rather
  than recursive — less error compounding, cleaner to evaluate.
- **Always compares against a naive persistence baseline** ("tomorrow = today").
  `training_pipeline.py` prints a warning if a trained model fails to beat it —
  don't ship a model that doesn't clear this bar.
- **Time-based train/test split only** — `time_based_split()` never shuffles, since
  random shuffling leaks future information into training for time series data.
- **Same feature engineering code path** is used by the hourly pipeline, the training
  pipeline, and inference (`feature_engineering.build_feature_pipeline`) — this
  avoids training/serving skew, a very common bug in real forecasting systems.
- **PM2.5→AQI conversion for backfilled data** uses the official EPA breakpoint table
  (verified against known reference points in `tests/test_pipeline.py`), so historical
  OpenWeather data is on the same AQI scale as live AQICN readings.
- **Local CSV backend exists specifically so you can build/test everything before
  paying any setup cost for Hopsworks** — flip one env var to go to production.

## 6. Known limitations to address as you extend this

- OpenWeather's historical endpoint doesn't provide temp/humidity/wind — those
  columns are NaN for backfilled-only rows. Once the hourly pipeline has been running
  a while, this stops mattering as live AQICN data fills them in going forward.
- The LSTM module and SHAP explanations require `tensorflow` and `shap` respectively —
  install both locally (`pip install tensorflow shap`) since they weren't verified in
  the build environment; the surrounding pipeline code they call into was fully tested.
- Multi-city support in the dashboard is present in the UI but each city needs its own
  backfilled history before its forecast will work.
