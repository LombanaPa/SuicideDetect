"""
Typing inputs in fasttext
"""
from stream_app.pages.src.app.utils.fasttext.fasttex_wrapper.typing import List, Union

import numpy as np
import pandas as pd

FasttextInput = Union[List[str], np.ndarray, pd.Series]
