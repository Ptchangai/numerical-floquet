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
from scipy.integrate import odeint #Replace by Assimulo integrator
#Todo: use numba

#Todo: add possibility for more models
def build_model(output_size=2):
    """Create architecture for our Neural Network ODE solver"""
    model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(3,)), 
    keras.layers.Dense(32, activation='sigmoid'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(8, activation='sigmoid'),
    keras.layers.Dense(output_size)
])
    model.compile(metrics='mean_absolute_error', loss='mean_squared_error' )
    model.summary()
    return model

def tune_build_model(hp):
    """Build tuneable model"""
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(3,)))
    model.add(layers.Dense(units=hp.Int('units', min_value=8, max_value=64, step=8),
                           activation=hp.Choice('activation', values=['relu', 'tanh', 'sigmoid'])))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop']),
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
    return model

def tune_model(model,model_name='Model', max_trials=10):
    """Finds best hyperparameters for model architecture"""
    tuner = RandomSearch(build_model,
                         objective='mean_squared_error',
                         max_trials=max_trials,
                         executions_per_trial=3,
                         directory='Tuning_'+model_name,
                         project_name='Neural_Floquet')
    
def train_model(model,Xdf,Ydf, epochs=100, batch_size=70, validation_split=0.1,shuffle=True):
    """Trains new model given a training dataset (Xdf, Ydf) and model architecture."""
    model = build_model()
    history = model.fit(Xdf, 
                        Ydf, 
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=validation_split,
                        shuffle=shuffle)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    return

def create_dataset(ODE, length, t0, N, parameters):
    """Generates training dataset with random values, integrating given ODE using odeint"""
    dataset = []
    for i in range(0,length):
            y0 = [random.uniform(0, 20), random.uniform(0, 20)] #(Initial conditions)
            h = random.uniform(0, 0.5)
            ti = random.randrange(1, N-1) #(=> Make it several points?)
            t = np.arange(t0,ti,h)
            sol = odeint(ODE, y0, t, args=parameters)
            sol_x1, sol_x2 = sol.T
            if np.abs(sol_x2[-1]) < 100 or np.abs(sol_x1[-1]) < 100:
                dataset.append([y0[0],y0[1],h,ti,sol_x1[-1],sol_x2[-1]])
    df = pd.DataFrame(dataset,columns=['X0','Y0','dt','t','X','Y'])
    Xdf = df.iloc[:, 0:4]
    Ydf = df.iloc[:, 4:]
    return [Xdf, Ydf]

def test_model():
    ...
    return