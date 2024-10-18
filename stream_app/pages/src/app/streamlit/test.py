import pandas as pd
from src.app.model.generate_embeddings import Embeddings
from src.app.conf.parameters import MODEL_EMBEDDINGS
from src.app.data.process_data import Preprocessing
from src.app.model.predict import predict
import numpy as np

dictio = [{'role': 'assistant', 'content': 'Hola soy Kamala, me alegra que busques esta linea para hablar... cuentame en que puedo ayudarte?'},
          {'role': 'user', 'content': 'Necesito hablar con alguien'},
          {'role': 'user', 'content': 'Siento que mi vida no vale nada'}]

filter_dict = [j['content'] for j  in [k for k in dictio if k['role']=='user']]

instance_embeddings = Embeddings(MODEL_EMBEDDINGS)
instance_preprocessing = Preprocessing()
probs = np.array([])
for k in filter_dict:
    text_prepro = instance_preprocessing.__call__(k)
    embeddings_step = np.array(instance_embeddings.get_embeddings(text_prepro))
    prediction = np.array(predict(embeddings_step.reshape(1, 768))[0])
    probs = np.concatenate((probs, prediction))
probs
