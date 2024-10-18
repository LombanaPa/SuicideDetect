import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
import keras
import warnings
import numpy as np
from pages.src.app.conf.parameters import dataframe_embeddings, ROUTES, MODEL_PARAMETERS, TRAINING_RESULTS
import logging
import os
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.simplefilter('ignore')

class Modeltrain:
    def __init__(self, learning_rate, batch_size, epoch, dropout, validation_split, shape):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch = epoch
        self.dropout = dropout
        self.validation_split = validation_split
        self.shape = shape
    def read_data_for_train(self):
        logger.info("Reading Dataset...")
        df = pd.read_csv(ROUTES['route_dataset'] + dataframe_embeddings)
        return df

    def buil_model(self):

        # Neural Network
        model_nn = Sequential([
            layers.Input(shape=(self.shape,)),
            Dense(500, activation='relu'),
            Dropout(self.dropout),
            Dense(1025, activation='relu'),

            Dense(1025, activation='relu'),

            Dense(500, activation='relu'),

            Dense(1, activation='sigmoid')
        ])

        model_nn.compile(optimizer=keras.optimizers.Adam(self.learning_rate), loss='binary_crossentropy',
                         metrics=['accuracy'])

        logger.info("Structure or Neural Network")
        logger.info(model_nn.summary())

        return model_nn

    def callback(self):
        callback = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.01,
            patience=5,
            verbose=1,
            baseline=None,
            restore_best_weights=True,
        )
        return callback

    def train_model(self):
        df = self.read_data_for_train()
        callback = self.callback()
        logger.info("Transform embeddings to array")
        #df = df.sample(50)
        df['embeddings_final'] = df['embeddings'].apply(lambda x: eval(x))
        self.df =df.copy()
        X = df['embeddings_final'].to_numpy()
        X = np.array([np.array(j) for j in X])
        y = df['Label'].to_numpy()
        logger.info(f"Dimention array {X.shape}")
        model_nn = self.buil_model()
        logger.info("Star to train ....")
        history = model_nn.fit(X,y, epochs = self.epoch, batch_size = self.batch_size,
                               validation_split = self.validation_split,
                               )
        #pd.DataFrame(history.history).to_csv("src/app/results/curve_train.csv")
        pd.DataFrame(history.history).to_csv(ROUTES['route_results'] + TRAINING_RESULTS['save_train_results'])
        #model_nn.save("src/app/results/model_ber_copia.h5")
        model_nn.save(ROUTES['route_results'] + TRAINING_RESULTS['name_model_bert'])


if __name__ == '__main__':
    os.chdir(ROUTES['route_project'])
    clase = Modeltrain(**MODEL_PARAMETERS)
    clase.train_model()

