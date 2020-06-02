

import os

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


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


path1 = os.path.join('Data')
path2='sales_train_validation.csv'





path3 = 'calendar.csv'

sale = pd.read_csv(os.path.join(path1,path2), delimiter=",")



class st_models:
  def __init__(self,ts,split = None,exog = None):
    self.ts= ts.astype("float64")
    self.split = split
    self.exog_ =exog
    if split!=None:


      self.ts_train = self.ts.iloc[:int(np.floor(self.ts.shape[0]*self.split))]
      self.ts_test = self.ts.iloc[int(np.floor(self.ts.shape[0]*self.split)):]
      self.ts_train = self.ts_train.astype("int64")
      self.ts_test = self.ts_test.astype("int64")



      self.horizon = self.ts_test.shape[0]


    else:
      self.ts_train = self.ts
      self.horizon = 1

  
    if np.sum(np.sum(((self.exog_!=None)))):
      self.exog = self.exog_.iloc[:int(np.floor(self.exog_.shape[0]*self.split))]
      self.exog_test = self.exog_.iloc[int(np.floor(self.exog_.shape[0]*self.split)):]
    else:
      self.exog = self.exog_
      self.exog_test = self.exog_
  def arima_pred(self,order=(1,0,1),pretrained=False,path = None,plot = False,decomp=False):
    if not pretrained:
     
      model1 = ARIMA(self.ts_train,order=order,exog = self.exog)

      if decomp:
        
        predictions = self.decompose(model = model1,pretrained = False)

      else:
        model = model1.fit()
      
        predictions =  model.forecast(self.horizon,exog=self.exog_test)
     
      if self.split!=None:
        err = 100*abs((predictions[0] - self.ts_test.iloc[:len(predictions[0])])/self.ts_test.iloc[:len(predictions[0])]).mean()
        #print('test mape = ',err)
        rmsse = self.RMSSE(predictions[0],self.ts_test,self.ts_train)
      if plot:
        self.plot_model(model,order)
      return predictions,rmsse



    else:
      for p in path:
        if decomp:
          predictions = self.decompose(pretrained= True,path=path)
        
        else:
          model =ARIMAResults.load(p)
          predictions =  model.forecast(self.horizon,exog = self.exog_test)
      
        


        
        if self.split!=None:
          err = 100*abs((predictions[0] - self.ts_test.iloc[:len(predictions[0])])/self.ts_test.iloc[:len(predictions[0])]).mean()
          #print('test mape = ',err)
          rmsse = self.RMSSE(predictions[0],self.ts_test,self.ts_train)
        if plot:
          self.plot_model(model,order)
        return predictions,rmsse


  def sarima_pred(self,order=(1,0,1),seorder=(0,0,1,8),pretrained=False,path = None,plot = False,decomp=False):
    if not pretrained:

      model1 = SARIMAX(self.ts_train,order=order,seasonal_order =seorder,exog = self.exog)
      if decomp:
        predictions = self.decompose(model = model1,pretrained=False)

      else:
        model = model1.fit()
        predictions =  np.array(model.forecast(self.horizon,exog = self.exog_test))
      
      if self.split!=None:
        err = 100*abs((predictions - self.ts_test.iloc[:len(predictions)])/self.ts_test.iloc[:len(predictions)]).mean()
        #print('test mape = ',err)
        rmsse = self.RMSSE(predictions,self.ts_test,self.ts_train)
                      
      if plot:
        self.plot_model(model)
      return predictions,rmsse

    else:
      
      
      model =SARIMAXResults.load(path)
      predictions =  np.array(model.forecast(self.horizon,exog = self.exog_test))
     
      if self.split!=None:
        err = 100*abs((predictions - self.ts_test.iloc[:len(predictions)])/self.ts_test.iloc[:len(predictions)]).mean()
        #print('test mape = ',err)
        rmsse = self.RMSSE(predictions,self.ts_test,self.ts_train)
      if plot:
        self.plot_model(model)
      return predictions,rmsse

  
  def plot_model(self,model,order):
    rss = np.sum(np.square(model.fittedvalues - self.ts_train))
    rmse = np.sqrt(rss/len(self.ts_train))
    mape = 100*np.mean(np.abs((model.fittedvalues - self.ts_train)/(self.ts_train)))
    plt.plot(self.ts_train,color='blue')
    plt.plot(model.fittedvalues,color = 'red')
    plt.title(f'For model with {order}, rmse = {rmse},mape = {mape}')
    plt.show()
    plt.close()

  def RMSSE(self,y_pred,y_test,y):
    y_pred = np.round(y_pred)
    y_test,y = y_test.values,y.values
    n = np.mean(np.square(y_test-y_pred))
    d = np.mean(np.square(y[1:]- y[0:y.shape[0]-1]))
    return np.sqrt(n/d)



def test_stationarity(ts,window):
    rolling_mean = ts.rolling(window = window,center=False).mean()
    
    rolling_std = ts.rolling(window = window,center=False).std()
    
    
    
    orignal = plt.plot(ts, color = 'blue', label = 'Original')
    mean = plt.plot(rolling_mean, color = 'red', label = 'Rolling Mean')
    std = plt.plot(rolling_std,color = 'black', label = 'Rolling Std')
    plt.legend()
    plt.show()
    plt.close()
    
    dftest = adfuller(ts)
    test_output= pd.DataFrame(dftest[0:4],index = ['Test Statistic','p-value','# Lags Used','Number of Observations Used'],columns=['Value'])

    for key,value in dftest[4].items():
        test_output = pd.concat([test_output,pd.DataFrame([value],index=[f'Critical Value at {key}'],columns=['Value'])])
    print(test_output)



def RMSSE(y_pred,y_test,y):
    y_pred = np.round(y_pred)
    y_test,y = y_test.values,y.values
    n = np.mean(np.square(y_test-y_pred))
    d = np.mean(np.square(y[1:]- y[0:y.shape[0]-1]))
    return np.sqrt(n/d)

#ts = list(ts1.iloc[:1813])
#pred= []

#for i in range(12):
#fit3 = ExponentialSmoothing(pd.Series(ts), seasonal_periods=7,damped=False, trend='add', seasonal='add').fit(use_boxcox=True)
#p = fit3.forecast(100)
  
  
  #pred.append(p)
  #ts.append(p)

#RMSSE(ts1.iloc[1813:],pd.Series(p),ts1.iloc[:1813])





#X,y = create_dataset(ts1.iloc[:1901])

#print(type(X[188]))


def RMSSE(y_pred,y_test,y):
    y_pred = np.round(y_pred)
    y_test,y = y_test.values,y.values
    n = np.mean(np.square(y_test-y_pred))
    d = np.mean(np.square(y[1:]- y[0:y.shape[0]-1]))
    return np.sqrt(n/d)

def do_lstm_model(df, ts, look_back, epochs,type_ = None, train_fraction = 0.05):
  
  # Import packages
  import numpy
  import matplotlib.pyplot as plt
  from pandas import read_csv
  import math
  from keras.models import Sequential
  from keras.layers import Dense
  from keras.layers import LSTM
  from sklearn.preprocessing import MinMaxScaler
  from sklearn.metrics import mean_squared_error

  # Convert an array of values into a dataset matrix
  def create_dataset(dataset, look_back=1):
    """
    Create the dataset
    """
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
      a = dataset[i:(i+look_back), 0]
      dataX.append(a)
      dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

  # Fix random seed for reproducibility
  numpy.random.seed(7)

  # Get dataset
  dataset = df[ts].values
  dataset = dataset.astype('float32')

  # Normalize the dataset
  scaler = MinMaxScaler(feature_range=(0, 1))
  dataset = scaler.fit_transform(dataset.reshape(-1, 1))
  
  # Split into train and test sets
  train_size = int(len(dataset) * train_fraction)
  test_size = len(dataset) - train_size
  train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
  
  # Reshape into X=t and Y=t+1
  look_back = look_back
  trainX, trainY = create_dataset(train, look_back)
  testX, testY = create_dataset(test, look_back)
  
  # Reshape input to be [samples, time steps, features]
  if type_ == 'regression with time steps':
    trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
  elif type_ == 'stacked with memory between batches':
    trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
  else:
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
  
  # Create and fit the LSTM network
  batch_size = 1
  model = Sequential()
  
  if type_ == 'regression with time steps':
    model.add(LSTM(4, input_shape=(look_back, 1)))
  elif type_ == 'memory between batches':
    model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
  elif type_ == 'stacked with memory between batches':
    model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
    model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
  else:
    model.add(LSTM(4, input_shape=(1, look_back)))
  
  model.add(Dense(1))
  model.compile(loss='mean_squared_error', optimizer='adam')

  if type_ == 'memory between batches' or type_ == 'stacked with memory between batches':
    for i in range(100):
      model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
      model.reset_states()
  else:
    model.fit(trainX, trainY,epochs = epochs, batch_size = 1, verbose = 2)
  
  # Make predictions
  if type_ == 'memory between batches' or type_ == 'stacked with memory between batches':
    trainPredict = model.predict(trainX, batch_size=batch_size)
    testPredict = model.predict(testX, batch_size=batch_size)
  else:
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
  
  # Invert predictions
  trainPredict = scaler.inverse_transform(trainPredict)
  trainY = scaler.inverse_transform([trainY])
  testPredict = scaler.inverse_transform(testPredict)
  testY = scaler.inverse_transform([testY])
  
  # Calculate root mean squared error
  trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
  
  print('Shape = ',trainY[0].shape, trainPredict.shape,trainY)
  testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
  print('Test Score: %.2f RMSE' % (testScore))
  
  # Shift train predictions for plotting
  trainPredictPlot = numpy.empty_like(dataset)
  trainPredictPlot[:, :] = numpy.nan
  trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
  
  # Shift test predictions for plotting
  testPredictPlot = numpy.empty_like(dataset)
  testPredictPlot[:, :] = numpy.nan
  testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
  
  # Plot baseline and predictions
  plt.plot(scaler.inverse_transform(dataset))
  plt.plot(trainPredictPlot)
  plt.plot(testPredictPlot)
  plt.show()
  plt.close()

  return




import warnings
warnings.filterwarnings('ignore')
import time
def train_st():
  preddf = pd.DataFrame([])
  predorder = pd.DataFrame([])
  levels = lc.LevelsCreater()
  for i in range(12):
    data =  levels.get_level(sale,i+1)
    st = time.time()

  
    n = len(data)

    for j in range(n):
      ts = data.iloc[j,:]
      print(i,j)
      arima_model = st_models(ts,split=0.895)
      try:
        _,err = arima_model.arima_pred(order = (10,0,1),pretrained=False,plot=False)
        order = (10,0,1)

      except:
        try:
          _,err = arima_model.arima_pred(order = (9,0,1),pretrained=False,plot=False)
          order = (9,0,1)
        except:
          try:
            _,err = arima_model.arima_pred(order = (8,0,1),pretrained=False,plot=False)
            order = (8,0,1)
          except:
            try:
              _,err = arima_model.arima_pred(order = (7,0,1),pretrained=False,plot=False)
              order = (7,0,1)
            except:
              try:
                _,err = arima_model.arima_pred(order = (6,0,1),pretrained=False,plot=False)
                order = (6,0,1)

              except:
                try:
                  _,err = arima_model.arima_pred(order = (5,0,1),pretrained=False,plot=False)
                  order = (5,0,1)

                except:
                  _,err = arima_model.arima_pred(order = (4,0,1),pretrained=False,plot=False)
                  order = (4,0,1)

      preddf[str(i+1)+'_'+ str(j+1)] = [err]
      predorder[str(i+1)+'_'+ str(j+1)] = [order]
      print(order)

  
    print(f'level {i+1} finished')
    

 


    en = time.time()

    print((en-st)/60)
  preddf.to_csv("results/arima_pred.csv")
  predorder.to_csv("results/arima_order.csv")

train_st()
