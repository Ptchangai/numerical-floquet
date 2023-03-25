import tensoflow as tf
#Helper libraries
import pandas as pd
import numpy as np
#Statistical data visualization library:
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras

import random

def build_model():
    return

def tune_model():
    return

def train_model(model,Xdf,Ydf):
    history = model.fit(Xdf, Ydf, epochs=100,batch_size=70,validation_split=0.1, shuffle=True)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    return

def test_model():
    return