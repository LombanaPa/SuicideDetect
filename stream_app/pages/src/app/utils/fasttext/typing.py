"""
Typing inputs in fasttext
"""
from stream_app.pages.src.app.utils.fasttext.typing import List, Union

import numpy as np
import pandas as pd

ModelInput = Union[List, np.ndarray, pd.Series]
