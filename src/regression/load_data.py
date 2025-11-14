import os
import pandas as pd
from typing import Tuple, List

# Define the feature set, consistent with the classification task
FEATURES = [
    "month", "day_of_month", "day_of_week",
    "op_unique_carrier", "origin", "dest",
    "dep_time", "distance", "dep_delay"
]

# The target for regression is the arrival delay in minutes
TARGET = "arr_delay"


def load_regression_data(path: str = None,
                           features: List[str] = FEATURES,
                           n_rows: int = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the preprocessed flight data for the regression task.

    Returns X (features DataFrame) and y (target Series, 'arr_delay').
    """
    # If no path is provided, construct it relative to this file's location
    if path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.abspath(os.path.join(base_dir, "../../data/preprocessed_flight_data.csv"))

    print(f"Loading data from: {path}")

    df = pd.read_csv(path, nrows=n_rows)

    # Ensure all required columns are present
    required_cols = features + [TARGET]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in the loaded CSV: {missing}")

    X = df[features]
    y = df[TARGET]

    print("Data loaded successfully.")
    return X, y

__all__ = ["load_regression_data", "FEATURES", "TARGET"]
