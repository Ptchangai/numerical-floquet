#This file turns the Floquet algorithms into Neural Network algorithms. 

import tensoflow as tf
#Helper libraries
import pandas as pd
import numpy as np
#Statistical data visualization library:
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner import HyperParameters
from kerastuner.tuners import RandomSearch
import random

def build_model():
    model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(3,)), 
    keras.layers.Dense(32, activation='sigmoid'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(8, activation='sigmoid'),
    keras.layers.Dense(2)
])
    model.compile(metrics='mean_absolute_error', loss='mean_squared_error' )
    model.summary()
    return model



def tune_build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(3,)))
    model.add(layers.Dense(units=hp.Int('units', min_value=8, max_value=64, step=8),
                           activation=hp.Choice('activation', values=['relu', 'tanh', 'sigmoid'])))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop']),
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
    return model

def tune_model(model,model_name='Model'):
    tuner = RandomSearch(build_model,
                         objective='mean_squared_error',
                         max_trials=10,
                         executions_per_trial=3,
                         directory='Tuning_'+model_name,
                         project_name='Neural_Floquet')
    
def train_model(model,Xdf,Ydf):

    model=build_model()
    history = model.fit(Xdf, 
                        Ydf, 
                        epochs=100,
                        batch_size=70,
                        validation_split=0.1,
                        shuffle=True)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    return

def test_model():
    return