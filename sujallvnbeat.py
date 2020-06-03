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
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from scipy.stats import zscore
import csv
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from sklearn.preprocessing import power_transform
from more_itertools import one
from scipy import stats
from functions import  series_to_supervised, mean_absolute_percentage_error, RMSSE,series_to_supervised
import exovar

# levels=lc.LevelsCreater()
#
# path1='Data'
# path2='sales_train_validation.csv'
#
# data = pd.read_csv(os.path.join(path1,path2), delimiter=",")
# df = levels.level_3(data)
# df=df.transpose()
# df=pd.DataFrame(df['CA_1'])
# df['d']=df.index
#
# path3='calendar.csv'
# calender=pd.read_csv(os.path.join(path1,path3))
# store_level_final = df.merge(calender, on='d')
#
# CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category",
#          "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
#         "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }
#
#
# for col_name, col_fit in CAL_DTYPES.items():
#         if col_name in store_level_final.columns:
#             store_level_final[col_name] = store_level_final[col_name].astype(col_fit)
#
# for col_name, col_fit in CAL_DTYPES.items():
#     if col_fit =='category':
#         store_level_final[col_name]=store_level_final[col_name].cat.codes.astype('int16')
#         store_level_final[col_name]-=store_level_final[col_name].min()
#
#
# new_store_level=store_level_final.drop(['d','date'],axis=1)
# print(new_store_level)
# # #...............code to observe RMSE and MAPE with different look-back window................
# # # rmse = []
# # # mape = []
# # # for mw in [10,15,20,30,35]:
# # #
# # #
# # #     # model prediction with explanatory variables
# # #     nsl_sts=series_to_supervised(new_store_level,mw,1)
# # #     nsl_sts=nsl_sts.dropna(axis=0)
# # #
# # #     t_column=[i for i in nsl_sts.columns if '(t)' in i]+[i for i in nsl_sts.columns if 'CA_1' in i]
# # #     exo_column=list(set(list(nsl_sts.columns))-set(t_column))
# # #
# # #     output_column='CA_1(t)'
# # #     input_column=list(set([i for i in nsl_sts.columns if 'CA_1' in i])-set(['CA_1(t)']))
# # #     x=nsl_sts[input_column].values.reshape(df.shape[0]-mw,mw,1)              #changed 1 to 0
# # #     y=nsl_sts[output_column].values.reshape(df.shape[0]-mw,1,1)
# # #     e=nsl_sts[exo_column].values.reshape(df.shape[0]-mw,mw,12)
# # #     num_samples, time_steps, input_dim, output_dim,exo = df.shape[0]-mw, mw, 1, 1,12
# # #     #nbeats code
# # #
# # #     model = NBeatsNet(exo_dim=exo,backcast_length=time_steps, forecast_length=output_dim,stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK), nb_blocks_per_stack=2,thetas_dim=(4, 4), share_weights_in_stack=True, hidden_layer_units=64)
# # #     model.compile_model(loss='mae', learning_rate=1e-5)
# # #     c = num_samples // 10
# # #     x_train, y_train, x_test, y_test,e_train,e_test = x[c:], y[c:], x[:c], y[:c],e[c:],e[:c]
# # #     model.fit([x_train,e_train], y_train, validation_data=([x_test,e_test], y_test), epochs=1500, batch_size=128)
# # #     # Save the model for later.
# # #     model.save('n_beats_model.h5')
# # #
# # #     # Predict on the testing set.
# # #     predictions = model.predict([x_test,e_test])
# # #     print(predictions.shape)
# # #
# # #     # Load the model.
# # #     # model2 = NBeatsNet.load('n_beats_model.h5')
# # #     #
# # #     # predictions2 = model2.predict([x_test,e_test])
# # #     # np.testing.assert_almost_equal(predictions, predictions2)
# # #
# # #     rmse.append(LA.norm(predictions.ravel()-y_test[:,0,0],2))
# # #     mape.append(mean_absolute_percentage_error(predictions.ravel(),y_test[:,0,0]))
# # #
# # #     fig,ax=plt.subplots()
# # #     ax.plot(predictions.ravel(),linestyle='--',label='prediction')
# # #     ax.plot(y_test[:,0,0],label='target')
# # #     ax.legend()
# # #     ax.set_title('with explanatory variable on tr= %f'%mw)
# # #     fig.savefig('Fig/nbeat_level3_1_tr%f.jpg'%mw)
# #
# #
# # # plt.show()
# # # #....................................activate from here............................
# # #
# # # erpd=pd.DataFrame({'RMSE':rmse,'MAPE':mape,'tr':[10,15,20,30,35]})
# # # erpd.set_index('tr')
# # # erpd.to_csv('Results/nbeat_level3_1.csv')
# # #
# # # fig2,ax1=plt.subplots()
# # # ax1.plot([10,15,20,30,35],rmse,marker='o',label='RMSE',color='tab:orange')
# # # ax1.tick_params(axis='y', labelcolor='tab:orange')
# # # ax1.set_xlabel('Training look-back window')
# # # ax1.set_ylabel('RMSE', color='tab:orange')
# # # ax2 = ax1.twinx()
# # # ax2.plot([10,15,20,30,35],mape,marker='o',label='MAPE',color='blue')
# # # ax2.set_ylabel('MAPE',color='blue')
# # # ax2.tick_params(axis='y', labelcolor='blue')
# # # # ax.legend()
# # # fig2.savefig('fig/nbeat_level3_1_er.jpg')
# #
# # #without explanatory variables
# # # data = pd.read_csv(os.path.join(path1,path2), delimiter=",")
# # # df = levels.level_3(data)
# # # rmse2=[]
# # # mape2=[]
# # # for mw in [5]:
# # #
# # #     newdf=series_to_supervised(pd.DataFrame(df.T['CA_1']),mw,1)
# # #     newdf=newdf.dropna(axis=0)
# # #     x=newdf[newdf.columns[:-1]].values.reshape(df.shape[1]-mw,mw,1)
# # #     y=newdf[newdf.columns[-1]].values.reshape(df.shape[1]-mw,1,1)
# # #
# # #
# # #
# # #
# # #
# # #     # https://keras.io/layers/recurrent/
# # #     num_samples, time_steps, input_dim, output_dim = df.shape[1]-mw, mw, 1, 1
# # #
# # #     model = NBeatsNet(backcast_length=time_steps, forecast_length=output_dim,
# # #                           stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK), nb_blocks_per_stack=2,
# # #                           thetas_dim=(4, 4), share_weights_in_stack=True, hidden_layer_units=64)
# # #
# # #     # Definition of the objective function and the optimizer.
# # #     model.compile_model(loss='mae', learning_rate=1e-5)
# # #
# # #     # Definition of the data. The problem to solve is to find f such as | f(x) - y | -> 0.
# # #     # x = np.random.uniform(size=(num_samples, time_steps, input_dim))
# # #     # y = np.mean(x, axis=1, keepdims=True)
# # #
# # #     # Split data into training and testing datasets.
# # #     c = num_samples // 10
# # #     x_train, y_train, x_test, y_test = x[c:], y[c:], x[:c], y[:c]
# # #
# # #     # Train the model.
# # #     model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=25, batch_size=128)
# # #
# # #     # Save the model for later.
# # #     model.save('n_beats_model.h5')
# # #
# # #     # Predict on the testing set.
# # #     predictions = model.predict(x_test)
# # #     print(predictions.shape)
# # #
# # #     # Load the model.
# # #     # model2 = NBeatsNet.load('n_beats_model.h5')
# # #
# # #     # predictions2 = model2.predict(x_test)
# # #     # np.testing.assert_almost_equal(predictions, predictions2)
# # #
# # #     rmse2.append(LA.norm(predictions.ravel()-y_test[:,0,0],2))
# # #     mape2.append(mean_absolute_percentage_error(predictions.ravel(),y_test[:,0,0]))
# # #
# # #
# # # fig,ax=plt.subplots()
# # # ax.plot(predictions.ravel(),linestyle='--',label='prediction')
# # # ax.plot(y_test[:,0,0],label='target')
# # # ax.set_title('without explanatory variable')
# # # ax.legend()
# # # plt.show()
# # #
# # #
# # # fig,ax=plt.subplots()
# # # ax.plot([10,15,20,25],rmse)
# # #
# # # fig,ax=plt.subplots()
# # # ax.plot([10,15,20,25],mape)
# # #
# # # # print(LA.norm(predictions.ravel()-y_test[:,0,0],2))
# # # # print(mean_absolute_percentage_error(predictions.ravel(),y_test[:,0,0]))
# # #
# #
# # # p: The number of lag observations included in the model, also called the lag order.
# # # d: The number of times that the raw observations are differenced, also called the degree of differencing.
# # # q: The size of the moving average window, also called the order of moving average.
# #
# # # model = ARIMA(df.iloc[0], order=(5,1,0))
# # # model_fit = model.fit(disp=0)
# # # print(model_fit.summary())
# # # residuals = DataFrame(model_fit.resid)
# # # residuals.plot(kind='kde')
# # # predictions=model_fit.forecast()
# # # plot_acf(df.iloc[0].values)
# # # # print(mean_absolute_percentage_error(predictions.ravel(),y_test[:,0,0]))
# # # plt.plot(model_fit.predict(1,df.iloc[0].shape[0],typ='levels'),label='prediction')
# # # plt.plot(df.iloc[0].values,label='target')
# # # plt.legend()
# # # plt.show()
# #
# #
# # # X = df.iloc[0].values
# # # result = adfuller(X)
# # # print('ADF Statistic: %f' % result[0])
# # # print('p-value: %f' % result[1])
# # # print('Critical Values:')
# #
# #
# #
# # # Prediction with normalized time-series
# # # scaler = MinMaxScaler()
# # # scaler.fit(new_store_level['CA_1'])
# # # new_store_level['CA_1']=scaler.fit_transform(new_store_level['CA_1'])
# # # new_store_level['CA_1']=zscore(new_store_level['CA_1'])
# #
# # # new_store_level['CA_1']=(new_store_level['CA_1']-new_store_level['CA_1'].min())/(new_store_level['CA_1'].max()-new_store_level['CA_1'].min())
# #
# # #Boxcox transformation
# # # new_store_level['CA_1'],_=stats.boxcox(new_store_level['CA_1'])
# #
# #
#
# rmse = []
# mape = []
# for mw in [30]:
#
#
#     # model prediction with explanatory variables
#     nsl_sts=series_to_supervised(new_store_level,mw,1)
#     nsl_sts=nsl_sts.dropna(axis=0)
#
#     t_column=[i for i in nsl_sts.columns if '(t)' in i]+[i for i in nsl_sts.columns if 'CA_1' in i]
#     exo_column=list(set(list(nsl_sts.columns))-set(t_column))
#
#     output_column='CA_1(t)'
#     input_column=list(set([i for i in nsl_sts.columns if 'CA_1' in i])-set(['CA_1(t)']))
#     x=nsl_sts[input_column].values.reshape(df.shape[0]-mw,mw,1)              #changed 1 to 0
#     y=nsl_sts[output_column].values.reshape(df.shape[0]-mw,1,1)
#     e=nsl_sts[exo_column].values.reshape(df.shape[0]-mw,mw,12)
#     num_samples, time_steps, input_dim, output_dim,exo = df.shape[0]-mw, mw, 1, 1,12
#     #nbeats code
#
#     model = NBeatsNet(exo_dim=exo,backcast_length=time_steps, forecast_length=output_dim,stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK), nb_blocks_per_stack=2,thetas_dim=(4, 4), share_weights_in_stack=True, hidden_layer_units=64)
#     model.compile_model(loss='mae', learning_rate=1e-5)
#     c = 121
#     x_train, y_train, x_test, y_test,e_train,e_test = x[:-c], y[:-c], x[-c:], y[-c:],e[:-c],e[-c:]
#     model.fit([x_train,e_train], y_train, validation_data=([x_test,e_test], y_test), epochs=500, batch_size=128)
#     # Save the model for later.
#     model.save('n_beats_model.h5')
#
#     # Predict on the testing set.
#     predictions = model.predict([x_test,e_test])
#     print(predictions.shape)
#
#     # Load the model.
#     # model2 = NBeatsNet.load('n_beats_model.h5')
#     #
#     # predictions2 = model2.predict([x_test,e_test])
#     # np.testing.assert_almost_equal(predictions, predictions2)
#     error=predictions.ravel()-y_test[:,0,0]
#     rmse.append(LA.norm(predictions.ravel()-y_test[:,0,0],2))
#     mape.append(mean_absolute_percentage_error(predictions.ravel(),y_test[:,0,0]))
#
#     # fig,ax=plt.subplots()
#     # ax.plot(predictions.ravel(),linestyle='--',label='prediction')
#     # ax.plot(y_test[:,0,0],label='target')
#     # ax.legend()
#     # ax.set_title('with explanatory variable on tr= %f'%mw)
#     # # fig.savefig('Fig/nbeat_level3_1_tr%f_zscorenormalized.jpg'%mw)
#     # fig,ax=plt.subplots()
#     # sm.graphics.tsa.plot_acf(error, lags=40)
#     # fig, ax = plt.subplots()
#     # ax.hist(error,bin=100)
#     # plt.show()
#
# # f = csv.writer('Results/nbeat_level3_1.csv', lineterminator='\n')
# # f.write("Zscore normalized: RMSE is %s, MAPE is %s" %{rmse, mape})



# print(mape)
def s_nbeat(level=1,exo=False,node=0,normalized=False,n_in=30,n_out=1,batch_size=128,optimizer='adam', loss='mse',test=200,epochs=500,decomposition=False):
    if exo==True:

        path1 = 'Data'
        path2 = 'sales_train_validation.csv'
        path3 = 'calendar.csv'
        sale = pd.read_csv(os.path.join(path1, path2), delimiter=",")
        calendar = pd.read_csv(os.path.join(path1, path3))
        levels=lc.LevelsCreater()
        df = levels.get_level(sale, level)
        df = df.transpose()
        df = pd.DataFrame(df[df.columns[node]])
        ex = exovar.exovar()
        ts_level = lc.LevelsCreater().get_level(sale, level)
        salecal = ex.salecal(ts_level, calendar, node)
        if level==1:
            salecal=salecal.rename(columns={salecal.columns[0]:'zero'})
        print(salecal.shape)
        bigx=series_to_supervised(salecal,n_in,n_out,parse=False,dropnan=True)
        # x, y, e = series_to_supervised(salecal, n_in, n_out, dropnan=True, parse=False)
        y_columsn = str(salecal.columns[0]) + '(t)'
        look_back_columns = list(set([i for i in bigx.columns if str(salecal.columns[0]) in i]) - set([y_columsn]))
        # look_back_columns.remove(y_columsn)
        exo_column = [i for i in bigx.columns if '(t)' not in i and i not in look_back_columns]
        x,y,e=bigx[look_back_columns],bigx[y_columsn],bigx[exo_column]
        print(x.columns)
        print(e.columns)
        print(y.shape)
        x = x.values.reshape(salecal.shape[0]-n_in , n_in, 1)
        y = y.values.reshape(salecal.shape[0]-n_in , 1, 1)
        e = e.values.reshape(salecal.shape[0]-n_in , n_in, 12)
        # x = x.values.reshape(x.shape[0], n_in, 1)
        # y = y.values.reshape(y.shape[0], 1, 1)
        # e = e.values.reshape(e.shape[0], n_in, 12)
        num_samples, time_steps, input_dim, output_dim, exo_dim = x.shape[0] , n_in, 1, 1, 12
        model = NBeatsNet(exo_dim=exo_dim, backcast_length=time_steps, forecast_length=output_dim,
                          stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK), nb_blocks_per_stack=2,
                          thetas_dim=(4, 4), share_weights_in_stack=True, hidden_layer_units=64)
        model.compile_model(loss='mae', learning_rate=1e-5)
        c = test
        x_train, y_train, x_test, y_test, e_train, e_test = x[:-c], y[:-c], x[-c:], y[-c:], e[:-c], e[-c:]
        model.fit([x_train, e_train], y_train, validation_data=([x_test, e_test], y_test), epochs=epochs, batch_size=batch_size)
        predictions = model.predict([x_test, e_test])
        print(predictions.shape)
        print(y_test.shape)

    return mean_absolute_percentage_error(predictions, y_test), LA.norm(predictions - y_test, 2,axis=0), RMSSE(predictions,
                                                                                                        y_test,
                                                                                                        df.values)

#


if __name__ == "__main__":
    mape,rmse,rmsse=s_nbeat(level=1,node=0,normalized=True,decomposition=True,exo=True)
    print('MAPE on testing set: %f' %mape)
    print('RMSE on testing set: %f' %rmse)
    print('RMSSE on testin set: %f'%rmsse)

