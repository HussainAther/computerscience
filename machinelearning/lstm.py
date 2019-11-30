import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import math

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

"""
Long Short-Term (long short term) Memory network or LSTM.
"""

np.random.seed(1234)

def create_dataset(dataset, look_back=1):
    """
    Create a dataset matrix from the array of values.
    """
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

# Load the dataset.
dataframe = pandas.read_csv("airline-passengers.csv", usecols=[1], engine="python")
dataset = dataframe.values
dataset = dataset.astype("float32")

# Normalize the dataset.
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Split into train and test sets.
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))
