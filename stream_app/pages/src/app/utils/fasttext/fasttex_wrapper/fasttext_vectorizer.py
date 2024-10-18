"""
Vectorizer text inputs
"""
from tempfile import NamedTemporaryFile
from stream_app.pages.src.app.utils.fasttext.fasttex_wrapper.typing import Any, Dict, List, Optional

import attr
import fasttext


from app.utils.fasttext.base import BaseVectorizer
from app.utils.fasttext.exceptions import ModelNotLoadException
from app.utils.fasttext.fasttex_wrapper.fasttext_base import BaseFasttext

from app.utils.fasttext.fasttex_wrapper.typing import FasttextInput


@attr.s
class FasttextVectorizer(BaseFasttext, BaseVectorizer):
    """
    A wrapper for fasttext vectorizer model.
    Extends BaseFasttext, adding all required args for training a non supervised
        fasttext model.

    Args:
        model: unsupervised fasttext model {cbow, skipgram} [skipgram].
    """

    model: str = attr.ib(default="skipgram")

    fasttext_model = attr.ib(default=None, init=False)

    def get_params(self) -> Dict:
        """
        Returns Dict with all the training parameters
        """
        params: Dict[Any, Any] = super().get_params()
        params.pop("fasttext_model", None)
        return params

    def fit(
        self, X: FasttextInput, y: Optional[FasttextInput] = None
    ) -> "FasttextVectorizer":
        """
        Trains a non supervised fasttext model.

        Args:
            X: A list with training data.
            y: List of targets. Not used in this method, keeped to comply with
                sklearn interface.

        Return:
            A trained model.

        """
        with NamedTemporaryFile(mode="w", suffix=".txt") as temp_file:
            self._dump_train_lines_to_file(X, temp_file.name)
            self.fasttext_model = fasttext.train_unsupervised(
                temp_file.name, **self.get_params()
            )
        return self

    def transform(self, X: FasttextInput) -> List:
        """
        Provides the fasttext vector representation of each input.

        Args:
            X: An array like data to transform.

        Returns:
            A list of predicted class.
        """
        if self.fasttext_model:
            X = self._check_data_type(X)
            Xtransformed = list(map(self.fasttext_model.get_sentence_vector, X))
            return Xtransformed
        else:
            raise ModelNotLoadException("There is not model loaded")
