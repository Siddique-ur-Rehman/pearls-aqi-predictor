
import os
import json
import numpy as np
import pandas as pd

from config import HORIZONS
from feature_engineering import FEATURE_COLUMNS
from training_pipeline import time_based_split, rmse

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def build_sequences(df: pd.DataFrame, feature_cols: list, target_col: str, window: int = 24):
    """
    Converts a flat feature table into (samples, timesteps, features) windows.
    Each sample uses the past `window` hours of features to predict one target value.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    clean = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    X, y = [], []
    for i in range(window, len(clean)):
        window_slice = clean.iloc[i - window: i][feature_cols].values
        target_val = clean.iloc[i][target_col]
        X.append(window_slice)
        y.append(target_val)
    return np.array(X), np.array(y)


def build_lstm_model(n_timesteps: int, n_features: int):
    """
    Lazy-imports TensorFlow so the rest of the codebase works even if
    TensorFlow isn't installed (e.g. this sandbox).
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(n_timesteps, n_features)),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def normalize_features(X_train, X_test):
    """Simple per-feature min-max scaling fit on train only (no leakage)."""
    train_min = X_train.min(axis=(0, 1), keepdims=True)
    train_max = X_train.max(axis=(0, 1), keepdims=True)
    range_ = np.where(train_max - train_min == 0, 1, train_max - train_min)
    X_train_scaled = (X_train - train_min) / range_
    X_test_scaled = (X_test - train_min) / range_
    return X_train_scaled, X_test_scaled, train_min, range_


def train_lstm_for_horizon(train_df, test_df, horizon: int, window: int = 24, epochs: int = 30):
    from tensorflow.keras.callbacks import EarlyStopping

    feature_cols = [c for c in FEATURE_COLUMNS if c in train_df.columns]
    target_col = f"target_{horizon}h"

    X_train, y_train = build_sequences(train_df, feature_cols, target_col, window)
    X_test, y_test = build_sequences(test_df, feature_cols, target_col, window)

    if len(X_train) < 50:
        raise ValueError(f"Not enough sequences to train LSTM for horizon={horizon}h "
                          f"(got {len(X_train)}, need 50+). Backfill more history.")

    X_train_s, X_test_s, train_min, train_range = normalize_features(X_train, X_test)

    model = build_lstm_model(n_timesteps=window, n_features=len(feature_cols))
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(
        X_train_s, y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

    preds = model.predict(X_test_s, verbose=0).flatten()
    metrics = {
        "rmse": rmse(y_test, preds),
        "mae": float(np.mean(np.abs(y_test - preds))),
    }

    model_path = os.path.join(MODEL_DIR, f"lstm_h{horizon}.keras")
    model.save(model_path)

    scaler_path = os.path.join(MODEL_DIR, f"lstm_h{horizon}_scaler.npz")
    np.savez(scaler_path, min=train_min, range=train_range)

    meta_path = os.path.join(MODEL_DIR, f"lstm_h{horizon}_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "horizon": horizon, "model_name": "lstm", "window": window,
            "feature_cols": feature_cols, "metrics": metrics,
        }, f, indent=2)

    print(f"[lstm] horizon={horizon}h  RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}  "
          f"saved to {model_path}")
    return model, metrics


def train_lstm_all_horizons(df: pd.DataFrame):
    train_df, test_df = time_based_split(df, test_frac=0.2)
    results = {}
    for horizon in HORIZONS:
        try:
            _, metrics = train_lstm_for_horizon(train_df, test_df, horizon)
            results[f"{horizon}h"] = metrics
        except Exception as e:
            print(f"[lstm] Skipped horizon={horizon}h: {e}")
    return results


if __name__ == "__main__":
    from feature_store import read_features, DEFAULT_BACKEND
    from feature_engineering import add_targets

    df = read_features(backend=DEFAULT_BACKEND)
    df = add_targets(df, horizons=HORIZONS)
    train_lstm_all_horizons(df)
