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
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import time

ex = ex.exovar()

path1 = os.path.join('Data2')
path2='sales_train_evaluation.csv'

path3 = 'calendar.csv'

sale = pd.read_csv(os.path.join(path1,path2), delimiter=",")

calendar = pd.read_csv(os.path.join(path1,path3), delimiter=",")

exogs = ex.calendar(calendar)

def RMSSE(y_pred,y_test,y):
    y_pred = np.round(y_pred)
    y_test,y = y_test.values,y.values
    n = np.mean(np.square(y_test-y_pred))
    d = np.mean(np.square(y[1:]- y[0:y.shape[0]-1]))
    return np.sqrt(n/d)




def lstm_model():
    look_back=28
    model = Sequential()   
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(20))
    model.add(Dense(look_back))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


np.random.seed(0)


def create_dataset(level,node,scaler):
    levels = lc.LevelsCreater()
    ts = levels.get_level(sale,level).iloc[node,:]

    look_back = 28
    
    train_fraction = 28

    
    dataset = scaler.fit_transform(ts.values.reshape(-1, 1))
    #print(len(dataset))
    
    
    ts_xy = series_to_supervised(pd.DataFrame(dataset.squeeze()),
                             look_back,dropnan=True)
    
    #print('tsxy',ts_xy.shape)
    
    train_size = int(len(ts_xy) - train_fraction)
    test_size = len(ts_xy) - train_size
    ts_x = ts_xy.iloc[:,:-1].values
    ts_y = ts_xy.iloc[:,-1].values
    trainX = ts_x[:train_size]
    testX = ts_x[train_size:]
    trainY = ts_y[:train_size]
    testY = ts_y[train_size:]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    print('train x shape',trainX.shape)
    return trainX,testX,trainY,testY


batch_size = 1

def train_lstm():
    preddf = pd.DataFrame([])
    levels = lc.LevelsCreater()
    exogs = ex.calendar(calendar)
    prederr = pd.DataFrame([])
    
    
    
    for i in range(12):
        data =  levels.get_level(sale,i+1)
        st = time.time()
        n = len(data)
        
        if i not in [9,10,11]:
            nodes = list(range(n))
        else:
            nodes = random.sample(range(n),100)
       
        m = 0
        
        for j in nodes:
            model = lstm_model()
            scaler = MinMaxScaler(feature_range=(0, 1))
            trainX,testX,trainY,testY = create_dataset(i+1,j,scaler)

            model.fit(trainX, trainY,epochs = 3, batch_size = 1, verbose = 2)    
            trainPredict = model.predict(trainX)
            testPredict = model.predict(testX)    
            trainPredict = scaler.inverse_transform(trainPredict)
            trainY = scaler.inverse_transform([trainY])
            testPredict = scaler.inverse_transform(testPredict)
            testY = scaler.inverse_transform([testY])
            
            
            
            
            m+=1
            print('ts number:',i+1,j+1,', number ts trained:',m)
    
            preddf[str(i+1)+'_'+ str(j+1)] = (np.array(testPredict.squeeze()))
            
            preddf.to_csv("results/lstm_predictions.csv")
            err = RMSSE(testPredict.squeeze(),
                         pd.Series(testY.squeeze()),pd.Series(trainY.squeeze()))
            
            prederr[str(i+1)+'_'+ str(j+1)] = [err]
            
            prederr.to_csv("results/lstm_errors.csv")
            print(err)
            model.save('lstm_models/'+str(i+1)+'_'+ str(j+1)+'.h5')
        en = time.time()
        
        print(f'level {i+1} finished, It took, {(en-st)/60} mins')
        #print('test x shape',testX.shape,testY.shape)


train_lstm()












