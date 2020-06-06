# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 22:39:03 2020

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

from fbprophet import Prophet as proph

import time


def RMSSE(y_pred,y_test,y):
    y_pred = np.round(y_pred)
    y_test,y = y_test.values,y.values
    n = np.mean(np.square(y_test-y_pred))
    d = np.mean(np.square(y[1:]- y[0:y.shape[0]-1]))
    return np.sqrt(n/d)

ex = ex.exovar()
path1 = os.path.join('Data2')
path2='sales_train_evaluation.csv'




path3 = 'calendar.csv'

sale = pd.read_csv(os.path.join(path1,path2), delimiter=",")

calendar = pd.read_csv(os.path.join(path1,path3), delimiter=",")

#exogs = ex.salecal(sale,calendar)


def train_fbpro():
    preddf = pd.DataFrame([])
    levels = lc.LevelsCreater()
    exogs = ex.calendar(calendar)
    #print('exogs len =',len(exogs['event_name_1']),exogs)
    
    
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
            
            ts = data.iloc[j,:]
            #print(len(ts))
            ts = pd.DataFrame(ts)
            ts = ts.rename(columns={ts.columns[0]:'y'})
            ts['ds'] = (calendar['date'].values)[:len(ts)]
            ts['event_name_1'] = (exogs['event_name_1'].values)[:len(ts)]
            ts['event_type_1'] = (exogs['event_type_1'].values)[:len(ts)]
            ts['event_name_2'] = (exogs['event_name_2'].values)[:len(ts)]
            ts['event_type_2'] = (exogs['event_type_2'].values)[:len(ts)]

            #print(np.sum(ts['event_name_1'].isna()))
            ts_train = ts.iloc[:,:]
            
            m+=1
            
            model = proph(daily_seasonality=True)
            model.add_country_holidays(country_name = 'US')
            model.add_regressor('event_name_1')
            model.add_regressor('event_type_1')
            model.add_regressor('event_name_2')
            model.add_regressor('event_type_2')
            
            #print(ts_train.head())
            model.fit(ts_train)
            future = model.make_future_dataframe(periods=28)
            future['event_name_1'] = exogs['event_name_1']
            future['event_type_1'] = exogs['event_type_1']
            future['event_name_2'] = exogs['event_name_2']
            future['event_type_2'] = exogs['event_type_2']
            
            y_pred = np.round(model.predict(future))
            #print(np.round(y_pred['yhat'].iloc[0]),type(np.round(y_pred['yhat'])))
   
            #err = RMSSE(np.round(y_pred['yhat'].iloc[(len(ts)+1):]),ts['y'].iloc[(len(ts)+1):],ts_train['y'])
            
            #print('RMSSE =', err)
            print('ts number:',i+1,j+1,', number ts trained:',m)
    
            preddf[str(i+1)+'_'+ str(j+1)] = (y_pred['yhat'].iloc[(len(ts)):]).values
            
            preddf.to_csv("results/fb_predictions(10,11,12).csv")
            
        
        en = time.time()
        
        print(f'level {i+1} finished, It took, {(en-st)/60} mins')
        
        
        
        

        
   
train_fbpro() 


#print(y_pred['yhat'].shape)

