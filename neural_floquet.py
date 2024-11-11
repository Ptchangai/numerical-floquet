#This file turns the Floquet algorithms into Neural Network algorithms. 
import random

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
from scipy.integrate import odeint 
from numerical_integration import RK4step
#TODO: remplace scipy.integrate with by Assimulo integrator
#TODO: use numba

#TODO: add complete class for wider range of models.
class ModelDefinition():
     ...

#TODO: make architecture more flexible(parameters...) 
def build_model(input_size, output_size=2):
    """
    Create Sequential architecture for our Neural Network ODE solver.
    """
    model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(None, input_size)), 
    keras.layers.GRU(32, dropout=0.2, recurrent_dropout=0.2, return_sequences = True),
    keras.layers.GRU(64, dropout=0.2, recurrent_dropout=0.2),
    keras.layers.Dense(32, activation='sigmoid'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(8, activation='sigmoid'),
    keras.layers.Dense(output_size)
    ])
    
    model.compile(metrics='mean_absolute_error', loss='mean_squared_error' )
    model.summary()
    return model

def tune_build_model(hp):
    """
    Build tuneable model.
    """
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(3,)))
    model.add(layers.Dense(units=hp.Int('units', min_value=8, max_value=64, step=8),
                           activation=hp.Choice('activation', values=['relu', 'tanh', 'sigmoid'])))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop']),
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
    return model

def tune_model(model, model_name='Model', max_trials=10):
    """
    Find best hyperparameters for model architecture.
    """
    tuner = RandomSearch(build_model,
                         objective='mean_squared_error',
                         max_trials=max_trials,
                         executions_per_trial=3,
                         directory='Tuning_'+model_name,
                         project_name='Neural_Floquet')
    
def train_model(model, Xdf, Ydf, epochs=100, batch_size=70, validation_split=0.1, shuffle=True):
    """
    Train new model given a training dataset (Xdf, Ydf) and model architecture.
    """

    history = model.fit(Xdf, 
                        Ydf, 
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=validation_split,
                        shuffle=shuffle)

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    return model

##TODO: refactor.
##TODO: pandas is not needed and might slow the program down. Remove it (can be use for other things)
#TODO: Adapt y0 to any size
#TODO: keep h constant?
#TODO: make time series rather than collection of random single steps. This way we can use it for LSTM.
def generate_dataset(ode_func, param, num_samples=100, future_steps=5, t_step=0.001, state_range=(0,10), size=2):
    """
    Generate random data points of ODE integration
    """
    X = []
    y = []
    for _ in range(num_samples):
        current_state = [np.random.uniform(state_range[0], state_range[2], size=2)]
        states = [current_state]
        for _ in range(future_steps):
            next_state = RK4step(ode_func, current_state,0, t_step, param)
            states.append(next_state)
            current_state = next_state
        X.append(states[0])
        y.append(states[-1])

    return np.array(X), np.array(y)   

def create_dataset(ODE, length, t0, N, parameters):
    """
    Generate training dataset with random values, integrating given ODE using odeint.
    """
    dataset = []
    for i in range(0,length):
            y0 = [random.uniform(0, 20), random.uniform(0, 20)] #(Initial conditions)
            h = random.uniform(0, 0.5)
            ti = random.randrange(1, N-1) #(=> Make it several points?)
            t = np.arange(t0,ti,h)
            sol = odeint(ODE, y0, t, args=parameters)
            sol_x1, sol_x2 = sol.T
            if np.abs(sol_x2[-1]) < 100 or np.abs(sol_x1[-1]) < 100:
                dataset.append([y0[0], y0[1], h, ti, sol_x1[-1], sol_x2[-1]])
    df = pd.DataFrame(dataset,columns=['X0','Y0','dt','t','X','Y'])
    Xdf = df.iloc[:, 0:4]
    Ydf = df.iloc[:, 4:]
    return [Xdf, Ydf]

##TODO: if the mean and values are provided, use them. Otherwise make them up.
def normalize_data(data_values):
     """
     Normalize numpy array data_values.
     """
     mean = data_values.mean(axis=0)
     data_values -= mean
     std = data_values.std(axis=0) 
     data_values /= std
     return data_values, mean, std

def test_model():
    """
    Test model accuracy against numerical method accuracy.
    """
    ...
    return

if __name__ == "__main__":

    ODE, length, t0, N, parameters = ...

    Xdf, Ydf = create_dataset(ODE, length, t0, N, parameters)
    input_size = Xdf.shape[-1]
    output_size = Ydf.shape[-1]

    model = build_model(input_size, output_size )
    model = train_model()