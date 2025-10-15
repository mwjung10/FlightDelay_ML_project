import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List


FEATURES = [
    "month",
    "day_of_month",
    "day_of_week",
    "op_unique_carrier",
    "origin",
    "origin_city_name",
    "origin_state_nm",
    "dest",
    "dest_city_name",
    "dest_state_nm",
    "dep_time",
    "distance"
]

TARGET = "is_arr_delayed"


def load_preprocessed_data(path: str = "./data/preprocessed_flight_data.csv",
                           features: List[str] = FEATURES) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the preprocessed flight data CSV and return X (features DataFrame) and y (target Series).

    This function will label-encode object (string) columns among the selected features in-place
    so the returned X contains numeric columns ready for classic ML models. It does not modify the
    CSV on disk.

    Parameters:
        path: Path to the preprocessed CSV file (default: ./data/preprocessed_flight_data.csv)
        features: List of feature column names to select (default: FEATURES)

    Returns:
        X: pd.DataFrame with selected feature columns (numeric)
        y: pd.Series with the target values
    """
    df = pd.read_csv(path)

    missing = set(features + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df_local = df.copy()

    for col in features:
        if df_local[col].dtype == 'object':
            le = LabelEncoder()
            df_local[col] = le.fit_transform(df_local[col].astype(str))

    X = df_local[features]
    y = df_local[TARGET]
    return X, y


__all__ = ["load_preprocessed_data", "FEATURES", "TARGET"]
