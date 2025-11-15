import os
import pandas as pd
from typing import Tuple, List


ALL_FEATURES = [
    "month", "day_of_month", "day_of_week",
    "op_unique_carrier", "origin",
    "dest","dep_time", "distance"
]



FEATURES_FOR_LCA = [
    "month", "day_of_month", "day_of_week",
    "op_unique_carrier", "origin", "dest",
]

TARGET = "is_arr_delayed"


def load_data_for_clustering(path: str = "../../data/preprocessed_flight_data.csv",
                             features: List[str] = FEATURES_FOR_LCA,
                             filter_for_delays: bool = True) -> pd.DataFrame:
    """
    Load data specifically for clustering.

    Args:
        path: Path to the CSV file.
        features: The list of features to include in the returned DataFrame.
        filter_for_delays: If True, filters the data to only include
                           rows where 'is_arr_delayed' is 1 (or True).
    """

    if not os.path.exists(path):
        print(f"Warning: File not found at {path}. Trying relative path logic...")
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.abspath(os.path.join(base_dir, path))
        except NameError:
            raise FileNotFoundError(f"Could not find data file. Please check path: {path}")

    print(f"Loading data from: {path}")

    cols_to_load = list(set(features + [TARGET]))
    df = pd.read_csv(path, usecols=cols_to_load)

    if filter_for_delays:
        print(f"Filtering for delayed flights (where {TARGET} == 1)...")
        df = df[df[TARGET] == 1].copy()
        print(f"Found {len(df)} delayed flights.")

    return df[features]