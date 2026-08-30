"""
Training Pipeline — run daily.

Reads engineered features + targets from the feature store, trains multiple
models per forecast horizon (24h/48h/72h), evaluates them, and saves the
best model per horizon.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def rmse(y_true, y_pred):
    """RMSE helper — sklearn 1.4+ dropped mean_squared_error(squared=False)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

from config import HORIZONS, FEATURES_DATA_PATH
from feature_engineering import add_targets, FEATURE_COLUMNS
from feature_store import read_features, DEFAULT_BACKEND

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def naive_persistence_baseline(y_true_current: np.ndarray, y_true_future: np.ndarray) -> dict:
    """The 'tomorrow = today' baseline. Any real model must beat this to be worth deploying."""
    preds = y_true_current  # predict no change
    return {
        "rmse": rmse(y_true_future, preds),
        "mae": mean_absolute_error(y_true_future, preds),
        "r2": r2_score(y_true_future, preds),
    }


def time_based_split(df: pd.DataFrame, ts_col: str = "timestamp", test_frac: float = 0.2):
    """
    Time-series-safe split: train on the older portion, test on the most recent.
    NEVER use random shuffling for time series data — it leaks future info into training.
    """
    df = df.sort_values(ts_col).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def prepare_training_data(backend: str = DEFAULT_BACKEND) -> pd.DataFrame:
    df = read_features(backend=backend)
    if df.empty:
        raise ValueError("Feature store is empty — run feature_pipeline.py / backfill.py first.")
    df = add_targets(df, horizons=HORIZONS)
    return df


def train_models_for_horizon(train_df, test_df, horizon: int, feature_cols=FEATURE_COLUMNS):
    """Trains Ridge + RandomForest for one horizon, returns metrics + fitted models."""
    target_col = f"target_{horizon}h"

    candidate_cols = [c for c in feature_cols if c in train_df.columns]

    fully_missing = [c for c in candidate_cols if train_df[c].isna().all()]
    if fully_missing:
        print(f"  [horizon={horizon}h] Skipping fully-missing columns: {fully_missing}")
    
    # Safely instantiates whether fully_missing contains items or is empty
    available_cols = [c for c in candidate_cols if c not in fully_missing]

    train_clean = train_df.dropna(subset=[target_col, "aqi"])
    test_clean = test_df.dropna(subset=[target_col, "aqi"])

    if len(train_clean) < 20 or len(test_clean) < 5:
        raise ValueError(
            f"Not enough clean data for horizon={horizon}h "
            f"(train={len(train_clean)}, test={len(test_clean)}). Need more backfilled history."
        )

    train_medians = train_clean[available_cols].median()
    X_train = train_clean[available_cols].fillna(train_medians)
    y_train = train_clean[target_col]
    X_test = test_clean[available_cols].fillna(train_medians)
    y_test = test_clean[target_col]

    results = {}

    # --- Baseline ---
    baseline_metrics = naive_persistence_baseline(test_clean["aqi"].values, y_test.values)
    results["persistence_baseline"] = {"model": None, "metrics": baseline_metrics}

    # --- Ridge ---
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_preds = ridge.predict(X_test)
    results["ridge"] = {
        "model": ridge,
        "metrics": {
            "rmse": rmse(y_test, ridge_preds),
            "mae": mean_absolute_error(y_test, ridge_preds),
            "r2": r2_score(y_test, ridge_preds),
        },
    }

    # --- Random Forest ---
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    results["random_forest"] = {
        "model": rf,
        "metrics": {
            "rmse": rmse(y_test, rf_preds),
            "mae": mean_absolute_error(y_test, rf_preds),
            "r2": r2_score(y_test, rf_preds),
        },
    }

    return results, available_cols


def select_best_model(results: dict, exclude_baseline: bool = True):
    """Picks the model with lowest RMSE (baseline excluded from selection but kept for comparison)."""
    candidates = {k: v for k, v in results.items() if not (exclude_baseline and k == "persistence_baseline")}
    best_name = min(candidates, key=lambda k: candidates[k]["metrics"]["rmse"])
    return best_name, candidates[best_name]


def save_model(model, name: str, horizon: int, feature_cols: list, metrics: dict):
    """Always saves locally first (used by the dashboard/API when running locally)."""
    path = os.path.join(MODEL_DIR, f"{name}_h{horizon}.joblib")
    joblib.dump({"model": model, "feature_cols": feature_cols}, path)

    meta_path = os.path.join(MODEL_DIR, f"{name}_h{horizon}_meta.json")
    with open(meta_path, "w") as f:
        json.dump({"horizon": horizon, "model_name": name, "metrics": metrics}, f, indent=2, default=str)
    return path


def register_model_hopsworks(local_path: str, name: str, horizon: int, metrics: dict):
    """
    Pushes a trained model to the Hopsworks Model Registry. Required in CI —
    GitHub Actions runners are ephemeral, so the joblib file in save_model()
    would otherwise vanish when the job ends. Call this only when
    FEATURE_STORE_BACKEND=hopsworks (see run()).
    """
    import hopsworks
    from config import HOPSWORKS_API_KEY, HOPSWORKS_PROJECT_NAME

    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME or None)
    mr = project.get_model_registry()

    model_name = f"aqi_{name}_h{horizon}"
    hw_model = mr.python.create_model(
        name=model_name,
        metrics={k: float(v) for k, v in metrics.items()},
        description=f"AQI forecast model ({name}) for horizon={horizon}h",
    )
    # Hopsworks expects a directory to upload, not a single file
    upload_dir = os.path.dirname(local_path)
    hw_model.save(upload_dir)
    print(f"  Registered '{model_name}' in Hopsworks Model Registry")
    return hw_model


def run(backend: str = DEFAULT_BACKEND):
    print("[training_pipeline] Loading training data...")
    df = prepare_training_data(backend=backend)
    train_df, test_df = time_based_split(df, test_frac=0.2)
    print(f"[training_pipeline] Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    summary = {}
    for horizon in HORIZONS:
        print(f"\n[training_pipeline] === Horizon: {horizon}h ===")
        results, feature_cols = train_models_for_horizon(train_df, test_df, horizon)

        for name, res in results.items():
            m = res["metrics"]
            print(f"  {name:22s} RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  R2={m['r2']:.3f}")

        best_name, best_res = select_best_model(results)
        print(f"  --> Best model: {best_name} (RMSE={best_res['metrics']['rmse']:.2f})")

        baseline_rmse = results["persistence_baseline"]["metrics"]["rmse"]
        if best_res["metrics"]["rmse"] >= baseline_rmse:
            print(f"  ⚠️  WARNING: best model does not beat the naive persistence baseline!")

        save_path = save_model(best_res["model"], best_name, horizon, feature_cols, best_res["metrics"])
        print(f"  Saved to {save_path}")

        if backend == "hopsworks":
            try:
                register_model_hopsworks(save_path, best_name, horizon, best_res["metrics"])
            except Exception as e:
                print(f"  ⚠️  Hopsworks model registry push failed: {e}")

        summary[f"{horizon}h"] = {
            "best_model": best_name,
            "metrics": best_res["metrics"],
            "beats_baseline": best_res["metrics"]["rmse"] < baseline_rmse,
        }

    print("\n[training_pipeline] Training complete. Summary:")
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    run()
