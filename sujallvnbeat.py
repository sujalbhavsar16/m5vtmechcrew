# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 18:06:50 2020

@author: Sujal Bhavsar
"""

import import_ipynb
import LevelsCreater as lc
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nbeats_keras.model import NBeatsNet
from numpy import linalg as LA
# from sklearn.utils import check_arrays
from statsmodels.tsa.arima_model import ARIMA
from pandas import DataFrame
from pandas import concat
plt.style.use(['amtlvt'])
def series_to_supervised(data, n_in=1, n_out=1, dropnan=False):

    n_vars = 1 if type(data) is list else data.shape[1]
    cols=pd.DataFrame()
    names=list()
    for i in range(n_in, 0, -1):
        df=pd.DataFrame()
        names=list()
        df=data.shift(i)
        names += [('%s(t-%d)' % (data.columns[j], i)) for j in range(n_vars)]
        df.columns=names
        cols = pd.concat([cols, df], axis=1, sort=False)
    for i in range(0, n_out):
        df=pd.DataFrame()
        names=list()
        df=data.shift(-i)
        if i == 0:
            names += [('%s(t)' % (data.columns[j])) for j in range(n_vars)]
        else:
            names += [('%s(t+%d)' % (data.columns[j], i)) for j in range(n_vars)]
            
        df.columns=names
        cols=pd.concat([cols,df],axis=1,sort=False)
    if dropnan:
        cols.dropna(inplace=True)
    return cols

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

levels=lc.LevelsCreater()

path1='Data'
path2='sales_train_validation.csv'

data = pd.read_csv(os.path.join(path1,path2), delimiter=",")
df = levels.level_3(data)
# rmse=[]
# mape=[]
# for mw in [10,15,20,25]:
#
#     newdf=series_to_supervised(pd.DataFrame(df.T['CA_1']),mw,1)
#     newdf=newdf.dropna(axis=0)
#     x=newdf[newdf.columns[:-1]].values.reshape(df.shape[1]-mw,mw,1)
#     y=newdf[newdf.columns[-1]].values.reshape(df.shape[1]-mw,1,1)
#
#
#
#
#
#     # https://keras.io/layers/recurrent/
#     num_samples, time_steps, input_dim, output_dim = df.shape[1]-mw, mw, 1, 1
#
#     model = NBeatsNet(backcast_length=time_steps, forecast_length=output_dim,
#                           stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK), nb_blocks_per_stack=2,
#                           thetas_dim=(4, 4), share_weights_in_stack=True, hidden_layer_units=64)
#
#     # Definition of the objective function and the optimizer.
#     model.compile_model(loss='mae', learning_rate=1e-5)
#
#     # Definition of the data. The problem to solve is to find f such as | f(x) - y | -> 0.
#     x = np.random.uniform(size=(num_samples, time_steps, input_dim))
#     y = np.mean(x, axis=1, keepdims=True)
#
#     # Split data into training and testing datasets.
#     c = num_samples // 10
#     x_train, y_train, x_test, y_test = x[c:], y[c:], x[:c], y[:c]
#
#     # Train the model.
#     model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=25, batch_size=128)
#
#     # Save the model for later.
#     model.save('n_beats_model.h5')
#
#     # Predict on the testing set.
#     predictions = model.predict(x_test)
#     print(predictions.shape)
#
#     # Load the model.
#     model2 = NBeatsNet.load('n_beats_model.h5')
#
#     predictions2 = model2.predict(x_test)
#     np.testing.assert_almost_equal(predictions, predictions2)
#
#     rmse.append(LA.norm(predictions.ravel()-y_test[:,0,0],2))
#     mape.append(mean_absolute_percentage_error(predictions.ravel(),y_test[:,0,0]))
#
#
# # fig,ax=plt.subplots()
# # ax.plot(predictions.ravel(),linestyle='--',label='prediction')
# # ax.plot(y_test[:,0,0],label='target')
# # ax.legend()
# # plt.show()
#
#
# fig,ax=plt.subplots()
# ax.plot([10,15,20,25],rmse)
#
# fig,ax=plt.subplots()
# ax.plot([10,15,20,25],mape)
#
# # print(LA.norm(predictions.ravel()-y_test[:,0,0],2))
# # print(mean_absolute_percentage_error(predictions.ravel(),y_test[:,0,0]))
#


model = ARIMA(df.iloc[0], order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
residuals = DataFrame(model_fit.resid)
residuals.plot(kind='kde')
predictions=model_fit.forecast()
# print(mean_absolute_percentage_error(predictions.ravel(),y_test[:,0,0]))
