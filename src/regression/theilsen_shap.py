import argparse
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import TheilSenRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None

try:
    import category_encoders as ce
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "category_encoders is required for this script. Install it via `pip install category-encoders`."
    ) from exc

from load_data import load_regression_data

CYCLICAL_MAP: Dict[str, int] = {
    "dep_time": 1440,
    "month": 12,
    "day_of_week": 7,
    "day_of_month": 31,
}

CATEGORICAL_FEATURES = ["op_unique_carrier", "origin", "dest"]
NUMERIC_BASE = ["distance", "dep_delay"]
BEST_THEILSEN_PARAMS = {"max_iter": 221, "max_subpopulation": 11_396}


def hhmm_to_minutes(time_val):
    if pd.isna(time_val):
        return np.nan
    try:
        if isinstance(time_val, str) and ":" in time_val:
            hour, minute = time_val.split(":")
            hour = int(hour)
            minute = int(minute)
            if hour == 24:
                return 0
            return hour * 60 + minute
        s = str(int(float(time_val))).zfill(4)
        if s == "2400":
            return 0
        hour = int(s[:2])
        minute = int(s[2:])
        return hour * 60 + minute
    except (ValueError, TypeError):
        return np.nan


class TimeConverter(BaseEstimator, TransformerMixin):
    def __init__(self, variable: str):
        self.variable = variable

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        if self.variable in X.columns:
            X[self.variable] = X[self.variable].apply(hhmm_to_minutes)
        return X


class CyclicalFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, variables_map: Dict[str, int]):
        self.variables_map = variables_map

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for var, period in self.variables_map.items():
            if var in X.columns:
                numeric_vals = pd.to_numeric(X[var], errors="coerce")
                X[f"{var}_sin"] = np.sin(2 * np.pi * numeric_vals / period)
                X[f"{var}_cos"] = np.cos(2 * np.pi * numeric_vals / period)
                X = X.drop(columns=[var])
        return X


def build_feature_pipeline():
    cyclical_features_generated = [
        "dep_time_sin",
        "dep_time_cos",
        "month_sin",
        "month_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "day_of_month_sin",
        "day_of_month_cos",
    ]
    all_numeric = NUMERIC_BASE + cyclical_features_generated

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "target_encoder",
                ce.TargetEncoder(handle_unknown="value", handle_missing="value"),
            )
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, all_numeric),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("time_converter", TimeConverter(variable="dep_time")),
            ("cyclical_features", CyclicalFeatures(CYCLICAL_MAP)),
            ("preprocessor", preprocessor),
        ]
    )


def sample_rows(
    X: pd.DataFrame, y: pd.Series, size: Optional[int], rng: np.random.Generator
):
    if size is None or size <= 0 or size >= len(X):
        return X.reset_index(drop=True), y.reset_index(drop=True)
    indices = rng.choice(len(X), size=size, replace=False)
    return X.iloc[indices].reset_index(drop=True), y.iloc[indices].reset_index(
        drop=True
    )


def ensure_dataframe(matrix: np.ndarray, columns) -> pd.DataFrame:
    return pd.DataFrame(matrix, columns=columns)


def main():
    parser = argparse.ArgumentParser(
        description="SHAP-style importance analysis for tuned Theil-Sen Regressor"
    )
    parser.add_argument(
        "--data-path",
        default="../../data/preprocessed_flight_data.csv",
        help="Path to the preprocessed CSV",
    )
    parser.add_argument(
        "--n-rows", type=int, default=None, help="Optional row cap when reading the CSV"
    )
    parser.add_argument(
        "--train-sample-size",
        type=int,
        default=50_000,
        help="Number of rows used to fit the feature pipeline and model (max).",
    )
    parser.add_argument(
        "--eval-sample-size",
        type=int,
        default=50_000,
        help="Rows used for SHAP evaluation subset.",
    )
    parser.add_argument(
        "--output-dir",
        default="../../results/theilsen_shap",
        help="Destination folder for SHAP artifacts",
    )
    parser.add_argument(
        "--max-display", type=int, default=25, help="How many features to show in plots"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for sampling"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.random_state)

    X, y = load_regression_data(path=args.data_path, n_rows=args.n_rows)
    print(f"Loaded {len(X):,} rows with {X.shape[1]} features")

    X_train, y_train = sample_rows(X, y, args.train_sample_size, rng)
    print(f"Training on {len(X_train):,} rows for Theil-Sen (subset)")

    feature_pipeline = build_feature_pipeline()
    X_train_matrix = feature_pipeline.fit_transform(X_train, y_train)
    preprocessor = feature_pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()

    model = TheilSenRegressor(
        random_state=args.random_state,
        n_jobs=-1,
        max_iter=BEST_THEILSEN_PARAMS["max_iter"],
        max_subpopulation=BEST_THEILSEN_PARAMS["max_subpopulation"],
    )
    model.fit(X_train_matrix, y_train)

    X_eval, y_eval = sample_rows(X, y, args.eval_sample_size, rng)
    print(f"Evaluation subset for SHAP: {len(X_eval):,} rows")
    X_eval_matrix = feature_pipeline.transform(X_eval)

    y_pred = model.predict(X_eval_matrix)
    mae = mean_absolute_error(y_eval, y_pred)
    rmse = np.sqrt(mean_squared_error(y_eval, y_pred))
    r2 = r2_score(y_eval, y_pred)
    print(f"MAE on eval subset: {mae:.4f}")
    print(f"RMSE on eval subset: {rmse:.4f}")
    print(f"R^2 on eval subset: {r2:.4f}")

    shap_values_array = None
    base_values = None

    if shap is not None:
        background = X_train_matrix
        explainer = shap.LinearExplainer(model, background)
        shap_explanation = explainer(X_eval_matrix)
        shap_values_array = shap_explanation.values
        base_values = shap_explanation.base_values

        plot_df = ensure_dataframe(X_eval_matrix, feature_names)
        shap.summary_plot(
            shap_values_array,
            plot_df.values,
            feature_names=feature_names,
            show=False,
            max_display=args.max_display,
        )
        plt.tight_layout()
        beeswarm_path = output_dir / "theilsen_shap_beeswarm.png"
        plt.savefig(beeswarm_path, dpi=200)
        plt.close()
        print(f"Saved SHAP beeswarm plot to {beeswarm_path}")

        shap.summary_plot(
            shap_values_array,
            plot_df.values,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
            max_display=args.max_display,
        )
        plt.tight_layout()
        bar_path = output_dir / "theilsen_shap_bar.png"
        plt.savefig(bar_path, dpi=200)
        plt.close()
        print(f"Saved SHAP bar plot to {bar_path}")
    else:
        print(
            "shap package not installed; skipping visualization. Install shap to enable plots."
        )

    if shap_values_array is not None:
        mean_abs_shap = np.abs(shap_values_array).mean(axis=0)
        shap_summary = (
            pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        csv_path = output_dir / "theilsen_shap_feature_importance.csv"
        shap_summary.to_csv(csv_path, index=False)
        print(f"Saved mean(|SHAP|) ranking to {csv_path}")
        print("Top 10 features by |SHAP|:")
        print(shap_summary.head(10))

        np.save(output_dir / "theilsen_shap_values.npy", shap_values_array)
        if base_values is not None:
            np.save(output_dir / "theilsen_shap_base_values.npy", base_values)
            print("Persisted raw SHAP arrays for downstream analysis.")
    else:
        print("SHAP values not computed; CSV and numpy exports skipped.")


if __name__ == "__main__":
    main()
