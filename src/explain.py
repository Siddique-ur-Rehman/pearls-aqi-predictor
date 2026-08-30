
import os
import numpy as np
import pandas as pd

from inference import load_best_model_for_horizon
from feature_store import get_latest_feature_row, DEFAULT_BACKEND
from config import CITY_NAME


def get_shap_explainer(model, model_name: str, background_data: pd.DataFrame = None):
    """
    Picks the right SHAP explainer type for the model family.
    Tree-based models (RandomForest) use TreeExplainer (fast, exact).
    Linear models (Ridge) use LinearExplainer (needs a background dataset).
    """
    import shap

    if model_name == "random_forest":
        return shap.TreeExplainer(model)
    elif model_name == "ridge":
        if background_data is None or len(background_data) < 10:
            raise ValueError("Ridge/linear models need background_data (a sample of training rows) for SHAP.")
        return shap.LinearExplainer(model, background_data)
    else:
        # Fallback: model-agnostic KernelExplainer (slower, works on anything)
        if background_data is None:
            raise ValueError("KernelExplainer requires background_data.")
        return shap.KernelExplainer(model.predict, shap.sample(background_data, 50))


def explain_forecast(horizon: int, city: str = CITY_NAME, backend: str = DEFAULT_BACKEND,
                      background_df: pd.DataFrame = None):
    """
    Returns SHAP values for the current prediction at a given horizon, plus
    the feature names/values, sorted by absolute contribution — ready to plot
    or render as a table in Streamlit.
    """
    model, feature_cols, model_path = load_best_model_for_horizon(horizon)
    model_name = os.path.basename(model_path).split("_h")[0]

    latest_row = get_latest_feature_row(city, backend=backend)
    X = latest_row[feature_cols].to_frame().T
    X = X.fillna(X.median(numeric_only=True)).fillna(0).astype(float)

    background = None
    if background_df is not None and not background_df.empty:
        background = background_df[feature_cols].fillna(background_df[feature_cols].median()).astype(float)

    explainer = get_shap_explainer(model, model_name, background_data=background)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):  # some explainers return a list per output
        shap_values = shap_values[0]

    contributions = pd.DataFrame({
        "feature": feature_cols,
        "value": X.iloc[0].values,
        "shap_value": shap_values[0] if shap_values.ndim > 1 else shap_values,
    })
    contributions["abs_impact"] = contributions["shap_value"].abs()
    contributions = contributions.sort_values("abs_impact", ascending=False).reset_index(drop=True)

    return {
        "model_name": model_name,
        "horizon": horizon,
        "base_value": float(explainer.expected_value) if not isinstance(explainer.expected_value, np.ndarray)
                      else float(explainer.expected_value[0]),
        "contributions": contributions,
    }


def top_n_drivers(explanation: dict, n: int = 5) -> pd.DataFrame:
    """Convenience: just the top N features driving this prediction, for a compact UI display."""
    return explanation["contributions"].head(n)[["feature", "value", "shap_value"]]


if __name__ == "__main__":
    from feature_store import read_features
    bg = read_features(backend=DEFAULT_BACKEND)
    result = explain_forecast(horizon=24, background_df=bg)
    print(f"Top drivers for {result['horizon']}h forecast (model: {result['model_name']}):")
    print(top_n_drivers(result))
