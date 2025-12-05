import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from xgboost import XGBClassifier
from math import sqrt


def hhmm_to_minutes(time_val):
    if pd.isna(time_val):
        return np.nan
    try:
        if isinstance(time_val, str) and ":" in time_val:
            parts = time_val.split(":")
            hour = int(parts[0])
            minute = int(parts[1])
            if hour == 24:
                return 0
            return hour * 60 + minute
        s = str(int(float(time_val))).zfill(4)
        if s == "2400":
            return 0
        hour = int(s[:2])
        minute = int(s[2:])
        return hour * 60 + minute
    except (ValueError, TypeError, IndexError):
        return np.nan


class FlightDataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.categorical_cols = ["op_unique_carrier", "origin", "dest"]
        self.cols_to_drop = [
            "origin_city_name",
            "origin_state_nm",
            "dest_city_name",
            "dest_state_nm",
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()

        cols_to_drop_existing = [
            col for col in self.cols_to_drop if col in X_copy.columns
        ]
        if cols_to_drop_existing:
            X_copy = X_copy.drop(columns=cols_to_drop_existing)

        if "dep_time" in X_copy.columns:
            X_copy["dep_time_minutes"] = X_copy["dep_time"].apply(hhmm_to_minutes)
            X_copy["dep_time_sin"] = np.sin(
                2 * np.pi * X_copy["dep_time_minutes"] / 1440.0
            )
            X_copy["dep_time_cos"] = np.cos(
                2 * np.pi * X_copy["dep_time_minutes"] / 1440.0
            )
            X_copy = X_copy.drop(columns=["dep_time", "dep_time_minutes"])

        if "month" in X_copy.columns:
            X_copy["month_sin"] = np.sin(2 * np.pi * X_copy["month"] / 12.0)
            X_copy["month_cos"] = np.cos(2 * np.pi * X_copy["month"] / 12.0)
            X_copy = X_copy.drop(columns=["month"])

        if "day_of_week" in X_copy.columns:
            X_copy["day_of_week_sin"] = np.sin(2 * np.pi * X_copy["day_of_week"] / 7.0)
            X_copy["day_of_week_cos"] = np.cos(2 * np.pi * X_copy["day_of_week"] / 7.0)
            X_copy = X_copy.drop(columns=["day_of_week"])

        if "day_of_month" in X_copy.columns:
            X_copy["day_of_month_sin"] = np.sin(
                2 * np.pi * X_copy["day_of_month"] / 31.0
            )
            X_copy["day_of_month_cos"] = np.cos(
                2 * np.pi * X_copy["day_of_month"] / 31.0
            )
            X_copy = X_copy.drop(columns=["day_of_month"])

        for col in self.categorical_cols:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].astype("category")

        return X_copy


class XGBoostThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.4125):
        self.threshold = threshold
        self.preprocessor = FlightDataPreprocessor()

        best_params = {
            "colsample_bytree": 0.7211248392548631,
            "gamma": 0.10445935880768009,
            "learning_rate": 0.14422870336481014,
            "max_depth": 11,
            "n_estimators": 838,
            "reg_alpha": 0.5183296523637367,
            "reg_lambda": 4.509492287711822,
            "subsample": 0.9222305853262613,
        }

        self.model = XGBClassifier(
            tree_method="hist",
            enable_categorical=True,
            random_state=42,
            n_jobs=-1,
            **best_params
        )

    def fit(self, X, y):
        X_processed = self.preprocessor.fit_transform(X)

        negative_count = len(y) - sum(y)
        positive_count = sum(y)
        if positive_count > 0:
            scale_pos_weight = sqrt(negative_count / positive_count)
        else:
            scale_pos_weight = 1

        self.model.set_params(scale_pos_weight=scale_pos_weight)

        self.model.fit(X_processed, y)
        return self

    def predict_proba(self, X):
        X_processed = self.preprocessor.transform(X)
        return self.model.predict_proba(X_processed)

    def predict(self, X):
        probas = self.predict_proba(X)
        return (probas[:, 1] >= self.threshold).astype(int)

    def save_model(self, path):
        import joblib

        joblib.dump(self, path)

    @staticmethod
    def load_model(path):
        import joblib

        return joblib.load(path)
