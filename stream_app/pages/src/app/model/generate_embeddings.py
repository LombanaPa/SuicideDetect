import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pages.src.app.conf.parameters import MODEL_EMBEDDINGS
from pages.src.app.conf.parameters import ROUTES
import logging
import numpy as np
import os
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 1. Load a pretrained Sentence Transformer model
class Embeddings:

    def __init__(self, model: str, dataframe: pd.DataFrame = None ) -> None:
        self.model = model
        logger.info(f"Cargado el modelo de sentecen transformer: {model}")
        self.SentenceModel = SentenceTransformer(self.model)
        self.dataframe = dataframe

    def get_embeddings(self, text:str) -> np.array:
        embedding = self.SentenceModel.encode(text)
        return list(embedding)

    def transform_all_text(self) -> pd.DataFrame:
        logger.info('Obteniendo los embeddings de todos los textos')
        tqdm.pandas()
        self.dataframe['embeddings'] = self.dataframe['text_clean'].progress_apply(lambda x: self.get_embeddings(x))
        return self.dataframe

    def run(self):
        df_process = self.transform_all_text()
        logger.info("Exportando el dataframe final")
        df_process.to_csv(ROUTES['route_dataset'] + ROUTES['dataframe_embeddings'])


if __name__ in '__main__':
    os.chdir(ROUTES['route_project'])
    df_clean = pd.read_parquet(ROUTES['route_dataset'] + ROUTES['name_clean_file'])
    obj_embeddings = Embeddings(MODEL_EMBEDDINGS, df_clean)
    obj_embeddings.run()
