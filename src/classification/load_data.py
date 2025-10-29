import os
import pandas as pd
from typing import Tuple, List

FEATURES = [
    "month", "day_of_month", "day_of_week",
    "op_unique_carrier", "origin", "origin_city_name",
    "origin_state_nm", "dest", "dest_city_name",
    "dest_state_nm", "dep_time", "distance"
]

TARGET = "is_arr_delayed"


def load_preprocessed_data(path: str = "../../data/preprocessed_flight_data.csv",
                           features: List[str] = FEATURES,
                           n_rows: int=3500000) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the preprocessed flight data CSV and return X (features DataFrame) and y (target Series).
    """

    # Automatically build the correct path relative to this file
    if path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.abspath(os.path.join(base_dir, "../../data/preprocessed_flight_data.csv"))

    print(f"📂 Loading data from: {path}")

    df = pd.read_csv(path)

    missing = set(features + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    X = df[features]
    y = df[TARGET]
    return X, y


__all__ = ["load_preprocessed_data", "FEATURES", "TARGET"]
