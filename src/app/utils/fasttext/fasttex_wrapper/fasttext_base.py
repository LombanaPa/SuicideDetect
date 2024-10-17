"""
Define parameters to get train vectorizer using fasstext
"""
import multiprocessing
from tempfile import NamedTemporaryFile
from typing import Dict, List

import attr
import fasttext


@attr.s
class BaseFasttext:
    """
    Base class for all fasttext wrappers.
    Takes all common arguments between supervised and no supervised training,
        leaving the rest for child classes.
    The actual fasttext model should be under the attribute name `fasttext_model`,
        in order to make the class serializable through the `__getstate__` and
        `__setstate__` methods.
    It also provides the `_dump_train_lines_to_file` to write a file with all train
        data, in order to comply with fasttext training interface. The child class
        has the responsability to create and destroy such file.

    Args:
        lr: learning rate [0.05].
        dim: size of word vectors [100].
        ws: size of the context window [5].
        epoch: number of epochs [5].
        minCount: minimal number of word occurences [5].
        minn: min length of char ngram [3].
        maxn: max length of char ngram [6].
        neg: number of negatives sampled [5].
        wordNgrams: max length of word ngram [1].
        loss: loss function {ns, hs, softmax, ova} [ns].
        bucket: number of buckets [2000000].
        thread: nunber of threads [number of cpus].
        lrUpdateRate: rate of updates for the learning rate [100].
        t: sampling threshold [0.0001].
        verbose: verbose [2].
    """

    lr: float = attr.ib(default=0.05)
    dim: int = attr.ib(default=100)
    ws: int = attr.ib(default=5)
    epoch: int = attr.ib(default=5)
    minCount: int = attr.ib(default=5)
    minn: int = attr.ib(default=3)
    maxn: int = attr.ib(default=6)
    neg: int = attr.ib(default=5)
    wordNgrams: int = attr.ib(default=1)
    loss: str = attr.ib(default="ns")
    bucket: int = attr.ib(default=2000000)
    thread: int = attr.ib(default=multiprocessing.cpu_count())
    lrUpdateRate: int = attr.ib(default=100)
    t: float = attr.ib(default=0.0001)
    verbose: int = attr.ib(default=2)

    _buffer_size: int = attr.ib(default=int(1e9), init=False)

    def __setstate__(self, state: Dict) -> None:
        """
        Load a serialized fasstext model when pickle.loads is called.

        Args:
            state: the full object state serialized.
        """
        serialized_model = state["serialized_model"]
        state.pop("serialized_model")
        self.__dict__.update(state)

        with NamedTemporaryFile(suffix=".bin") as temp_file:
            temp_file.write(serialized_model)
            temp_file.flush()
            self.fasttext_model = fasttext.load_model(temp_file.name)

    def __getstate__(self) -> Dict:
        """
        Serialize a fasttext model and all the object instance
        when pickle.dumps is called.
        """
        _self = self.__dict__.copy()

        if self.fasttext_model:
            with NamedTemporaryFile(suffix=".bin") as temp_file:
                self.fasttext_model.save_model(temp_file.name)
                with open(temp_file.name, "rb") as f:
                    serialized_model = f.read()
                    _self["serialized_model"] = serialized_model
                    _self["fasttext_model"] = None

        return _self

    def _dump_train_lines_to_file(self, train_lines: List[str], path: str) -> None:
        """
        Dumps train dataset into a file in order to fit the fasttext model.

        Args:
            X: list of sentences used as features.
            path: path of the file.

        """
        with open(path, "w", buffering=self._buffer_size) as writer:
            for line in train_lines:
                writer.write("{}\n".format(line))
