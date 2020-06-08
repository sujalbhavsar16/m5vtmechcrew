# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 22:40:07 2020

@author: Chaitanya
"""

import os

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import random

from statsmodels.tsa.stattools import acf,pacf
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm
#from nbeats_keras.model import NBeatsNet
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from statsmodels.tsa.arima_model import ARIMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from numpy import linalg as LA
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import LevelsCreater as lc
import exovar as ex

from functions import series_to_supervised

from fbprophet import Prophet as proph

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def RMSSE(y_pred,y_test,y):
    y_pred = np.round(y_pred)
    y_test,y = y_test.values,y.values
    n = np.mean(np.square(y_test-y_pred))
    d = np.mean(np.square(y[1:]- y[0:y.shape[0]-1]))
    return np.sqrt(n/d)


np.random.seed(0)



ts = levels.get_level(sale,1).iloc[0,:]
look_back = 28
epochs = 3
train_fraction = 0.985



scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(ts.values.reshape(-1, 1))

train_size = int(len(dataset) * train_fraction)
test_size = len(dataset) - train_size
    
ts_xy = series_to_supervised(pd.DataFrame(dataset.squeeze()),
                             look_back,dropnan=True)



ts_x = ts_xy.iloc[:,:-1].values

ts_y = ts_xy.iloc[:,-1].values
trainX = ts_x[:train_size]
testX = ts_x[train_size:]

trainY = ts_y[:train_size]
testY = ts_y[train_size:]

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#print(trainX.shape)

#print(trainX.shape)
batch_size = 1
model = Sequential()
  
 
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(20))
model.add(Dense(look_back))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
  
model.fit(trainX, trainY,epochs = 3, batch_size = 1, verbose = 2)
    
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
    
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
  
print(RMSSE(testPredict.squeeze(),
      pd.Series(testY.squeeze()),pd.Series(trainY.squeeze())))

















