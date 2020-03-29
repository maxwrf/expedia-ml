import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
from keras import layers, models
from source.models.base_model import BaseModel
import os
from keras.layers import Dropout

# Disable tensorflow warnings concerning CPU optimization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class NeuralNetwork(BaseModel):
    def __init__(self, config, X, y, params=None):
        super().__init__(config, X, y, params)
        self.clf_name = 'NeuralNetwork'

    def get_default_model(self):
        return KerasClassifier

    def get_default_parameter(self):
        return {
            'build_fn': self.create_network,
            'epochs': 50,
            'batch_size': 100,
            'verbose': 0
        }

    def create_network(self):
        nn = models.Sequential()
        nn.add(layers.Dense(units=512,
                            activation='relu',
                            input_shape=(self.X.shape[1],)
                            ))
        nn.add(Dropout(0.25))
        nn.add(layers.Dense(units=256,
                            activation='relu'
                            ))
        nn.add(Dropout(0.25))
        nn.add(layers.Dense(units=256,
                            activation='relu'
                            ))
        nn.add(Dropout(0.25))
        nn.add(layers.Dense(units=128,
                            activation='relu'
                            ))
        nn.add(Dropout(0.25))
        nn.add(layers.Dense(units=64,
                            activation='relu'
                            ))
        nn.add(Dropout(0.25))
        nn.add(layers.Dense(units=64,
                            activation='relu'
                            ))
        nn.add(Dropout(0.25))
        nn.add(layers.Dense(units=32,
                            activation='relu'
                            ))
        nn.add(Dropout(0.25))
        nn.add(layers.Dense(units=len(pd.unique(self.y)),
                            activation='softmax'
                            ))
        nn.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=[self.config.get('Models', 'scoring')]
                   )
        return nn
