import pandas as pd
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


def load_preprocessed_data(path: str = "../../data/preprocessed_flight_data.csv",
                           features: List[str] = FEATURES,
                           n_rows: int=3500000) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the preprocessed flight data CSV and return X (features DataFrame) and y (target Series).

    This function returns the selected feature columns and the target column as-is from the
    preprocessed CSV. It does not perform any encoding or scaling — that should be handled by
    the caller so that preprocessing choices remain explicit.

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

    X = df[features]
    y = df[TARGET]
    return X, y


__all__ = ["load_preprocessed_data", "FEATURES", "TARGET"]
