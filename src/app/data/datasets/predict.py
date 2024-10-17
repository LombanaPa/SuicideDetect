from keras.models import load_model
import pandas as pd
import numpy as np
from src.app.conf.parameters import dataframe_embeddings, ROUTES, MODEL_PARAMETERS, TRAINING_RESULTS

def predict(X):
    '''
    Predic using model
    Examples
    -------
    value = predict(np.arange(0,768).reshape(1, 768))
    '''
    modelo = load_model(ROUTES['route_results'] + TRAINING_RESULTS['name_model_bert'])
    if X.shape[0] >1:
        return modelo.predict(X.reshape(-1,1))
    else:
        return modelo.predict(X)