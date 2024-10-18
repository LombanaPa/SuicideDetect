"""
Class to get type of input
"""
from typing import List

import numpy as np
import pandas as pd
from app.utils.fasttext.typing import ModelInput


class EstimatorMixin:
    """
        EstimatorMixin adds a method that checks wheter an input is of type np.ndarray
            or pd.Series and transforms it into a list, so every estimator can take
            either one of this data types as inputs.
    """

    def _check_data_type(self, X: ModelInput) -> List:
        """
            Validate if the data to predict is a List type.
            Oterwise transform it to a list.
            Args:
                X: list of items to predict. That could be or not
                    a ndarray, pandas series or just a list type.
            Return:
                A list type of items.
        """
        if isinstance(X, (np.ndarray, pd.Series)):
            X = X.tolist()

        return X
