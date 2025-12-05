import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score

try:
    import shap
except ImportError:  # pragma: no cover - optional dependency
    shap = None

from load_data import load_preprocessed_data
from xgboost_model import XGBoostThresholdClassifier


def sample_rows(
    X: pd.DataFrame, y: pd.Series, size: Optional[int], rng: np.random.Generator
):
    """Return a random subset (without replacement) if size is specified."""
    if size is None or size <= 0 or size >= len(X):
        return X.reset_index(drop=True), y.reset_index(drop=True)

    indices = rng.choice(len(X), size=size, replace=False)
    return (
        X.iloc[indices].reset_index(drop=True),
        y.iloc[indices].reset_index(drop=True),
    )


def ensure_numeric_for_plotting(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical columns to codes so matplotlib/shap can plot them."""
    plot_df = df.copy()
    categorical_cols = plot_df.select_dtypes(include="category").columns
    for col in categorical_cols:
        plot_df[col] = plot_df[col].cat.codes
    return plot_df


def main():
    parser = argparse.ArgumentParser(
        description="SHAP analysis for the tuned XGBoost classifier"
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
        default=1_000_000,
        help="Rows used to (re)fit the tuned model",
    )
    parser.add_argument(
        "--eval-sample-size",
        type=int,
        default=50_000,
        help="Rows used for SHAP evaluation subset",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.4125,
        help="Probability threshold for class predictions",
    )
    parser.add_argument(
        "--output-dir",
        default="../../results/xgboost_shap",
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

    X, y = load_preprocessed_data(path=args.data_path, n_rows=args.n_rows)
    print(f"Loaded {len(X):,} rows with {X.shape[1]} features")

    X_train, y_train = sample_rows(X, y, args.train_sample_size, rng)
    print(f"Training on {len(X_train):,} rows")

    model = XGBoostThresholdClassifier(threshold=args.decision_threshold)
    model.fit(X_train, y_train)

    X_eval, y_eval = sample_rows(X, y, args.eval_sample_size, rng)
    print(f"SHAP evaluation subset: {len(X_eval):,} rows")

    X_eval_processed = model.preprocessor.transform(X_eval)
    eval_proba = model.model.predict_proba(X_eval_processed)[:, 1]
    y_pred = (eval_proba >= args.decision_threshold).astype(int)

    auc = roc_auc_score(y_eval, eval_proba)
    print(f"ROC AUC on SHAP subset: {auc:.4f}")
    print("Classification report (SHAP subset):")
    print(classification_report(y_eval, y_pred, digits=3))

    d_eval = xgb.DMatrix(X_eval_processed, enable_categorical=True)
    booster = model.model.get_booster()
    shap_contribs = booster.predict(d_eval, pred_contribs=True)

    feature_names = list(X_eval_processed.columns)
    shap_values = shap_contribs[:, :-1]
    base_values = shap_contribs[:, -1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_summary = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    summary_path = output_dir / "xgboost_shap_feature_importance.csv"
    shap_summary.to_csv(summary_path, index=False)
    print(f"Saved mean(|SHAP|) ranking to {summary_path}")

    print("Top 10 features by |SHAP|:")
    print(shap_summary.head(10))

    if shap is not None:
        plot_matrix = ensure_numeric_for_plotting(X_eval_processed)
        shap.summary_plot(
            shap_values,
            plot_matrix.values,
            feature_names=feature_names,
            show=False,
            max_display=args.max_display,
        )
        plt.tight_layout()
        png_path = output_dir / "xgboost_shap_beeswarm.png"
        plt.savefig(png_path, dpi=200)
        plt.close()
        print(f"Saved SHAP beeswarm plot to {png_path}")

        shap.summary_plot(
            shap_values,
            plot_matrix.values,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
            max_display=args.max_display,
        )
        plt.tight_layout()
        bar_path = output_dir / "xgboost_shap_bar.png"
        plt.savefig(bar_path, dpi=200)
        plt.close()
        print(f"Saved SHAP bar plot to {bar_path}")
    else:
        print(
            "shap package not installed; skipped plot generation. Run `pip install shap` to enable plots."
        )

    np.save(output_dir / "xgboost_shap_values.npy", shap_values)
    np.save(output_dir / "xgboost_shap_base_values.npy", base_values)
    print("Persisted raw SHAP arrays for downstream analysis.")


if __name__ == "__main__":
    main()
