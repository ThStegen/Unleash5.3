import pandas as pd
import numpy as np
from typing import Optional, Union

# ExplicitFlex.py

def load_boiler_csv(path: str,
                    expected_rows: int = 96,
                    expected_cols: int = 121,
                    delimiter: str = ',') -> np.ndarray:
    """
    Load a CSV file containing boiler data arranged as expected_rows x expected_cols.
    Returns a numpy array of shape (expected_rows, expected_cols) with dtype float.

    Raises ValueError on unexpected shape or if conversion to float fails.
    """
    # read without headers, allow flexible parsing
    df = pd.read_csv(path, header=None, sep=delimiter, engine='python')
    # drop fully empty rows (e.g. trailing newline)
    df = df.dropna(how='all').reset_index(drop=True)

    if df.shape != (expected_rows, expected_cols):
        raise ValueError(f"CSV shape {df.shape} does not match expected ({expected_rows}, {expected_cols})")

    try:
        arr = df.values.astype(float)
    except ValueError as e:
        raise ValueError(f"Failed to convert CSV contents to float: {e}")

    return arr