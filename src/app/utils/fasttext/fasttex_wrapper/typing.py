"""
Typing inputs in fasttext
"""
from typing import List, Union

import numpy as np
import pandas as pd

FasttextInput = Union[List[str], np.ndarray, pd.Series]
