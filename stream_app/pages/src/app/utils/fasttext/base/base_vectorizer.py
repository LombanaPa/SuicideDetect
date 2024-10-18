"""
Modelo to abstract implementation transformers.
"""
from typing import List, Optional

import attr
from sklearn.base import BaseEstimator, TransformerMixin
from app.utils.fasttext.base.estimator_mixin import EstimatorMixin
from app.utils.fasttext.typing import ModelInput


@attr.s
class BaseVectorizer(BaseEstimator, TransformerMixin, EstimatorMixin):
    """
        Abstract implementation for transformers.
    """

    def fit(self, X: ModelInput, y: Optional[ModelInput] = None) -> object:
        """
        Fits the transformer to the data.

        This method is intended to be overridden by subclasses. It takes in the data to be fitted and optionally 
        the target values (labels). The method should fit the transformer to the provided data.

        Parameters:
        X (ModelInput): The input data to fit.
        y (Optional[ModelInput], optional): The target values (labels) corresponding to X. Defaults to None.

        Returns:
        object: Returns an instance of the fitted transformer.
        """
        raise NotImplementedError()

    def transform(self, X: ModelInput) -> List:
        """
        Transforms the input data.

        This method is intended to be overridden by subclasses. It takes in the data to be transformed and returns
        the transformed data as a list.

        Parameters:
        X (ModelInput): The input data to transform.

        Returns:
        List: A list containing the transformed data.
        """
        raise NotImplementedError()
