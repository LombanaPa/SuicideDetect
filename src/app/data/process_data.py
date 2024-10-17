import pandas as pd
import numpy as np
import os
from src.app.data.transform import  (ToLower, RemoveHTML, RemovePunctuation,
                                     RemoveEmojis, RemoveAccents,
                                     RemoveEscapeSequences, ReplaceURLs, RemoveBlankSpaces)
from src.app.conf.parameters import  ROUTES
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Preprocessing:
    def __call__(self, text: str) -> str:
        tolower = ToLower()
        removehtml = RemoveHTML()
        removepuntuaction = RemovePunctuation()
        removemojis = RemoveEmojis()
        removeaccents = RemoveAccents()
        removescapesequences = RemoveEscapeSequences()
        replaceUrl = ReplaceURLs()
        remove_blank = RemoveBlankSpaces()

        text = tolower(text)
        text = removehtml(text)
        text = removepuntuaction(text)
        text = removemojis(text)
        text = removeaccents (text)
        text = removescapesequences(text)
        text = replaceUrl(text)
        text = remove_blank(text)
        return text

class TransformDataframe:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def process_text(self) -> pd.DataFrame:
        preprocessor_obj = Preprocessing()
        self.df['text_clean'] = self.df['Text'].apply(lambda x: preprocessor_obj(x))
        return df
    def create_dataset_clean(self) -> None:
        clean_df = self.process_text()
        logging.info("Exportando el dataset limpio")
        #clean_df.to_parquet("src/app/data/datasets/dataset_cleaned.parquet")
        clean_df.to_parquet(ROUTES['route_dataset'] + ROUTES['name_clean_file'])

if __name__ == '__main__':
    logger.info("Leyendo el Dataframe para entrenar")
    os.chdir(ROUTES['route_project'])
    df = pd.read_parquet(ROUTES['route_dataset'] + ROUTES['name_file'])
    logger.info("Transformando el dataset")
    resul_df = TransformDataframe(df)
    resul_df.create_dataset_clean()
    logger.info("Proceso Finalizado con exito!!! 🔧")















