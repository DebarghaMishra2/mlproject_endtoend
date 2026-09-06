import os
import sys
from src.pipeline.logger import logging

import pickle
import numpy as np


def save_object(file_path, obj):
    """
    Save a Python object to a file using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)

        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise e


def load_object(file_path):
    """
    Load a Python object from a pickle file.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise e


def evaluate_model(true, predicted):
    """
    Evaluate a regression model.
    """

    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score
    )

    mae = mean_absolute_error(true, predicted)

    rmse = np.sqrt(
        mean_squared_error(true, predicted)
    )

    r2 = r2_score(true, predicted)

    return mae, rmse, r2