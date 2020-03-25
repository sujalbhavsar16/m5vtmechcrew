import pandas as pd
import numpy as np
from tslearn.utils import to_time_series_dataset
from tslearn.metrics import cdist_dtw
from scipy.spatial.distance import cdist

sale=pd.read_csv('sales_train_validation.csv')
print(sale.columns)

lowlevelsale=sale.drop(columns=['id','item_id','dept_id','cat_id','store_id','state_id'])
print(lowlevelsale.shape)

time_series=[]
for i in range(lowlevelsale.shape[0]):
    time_series.append(lowlevelsale.iloc[i].values)

formatted_time_series = to_time_series_dataset(time_series)

print(np.array(time_series).shape)

TS=np.array(time_series)

dtwrelation=cdist(TS,TS)

print(dtwrelation.shape)

print(np.argsort(dtwrelation)[:2000])

#need to find first few thousand elemnts which have lower values in dtwrelation matrix
cidx=np.argsort(dtwrelation)[:2000]
np.savetxt('cidx.csv',cidx,delimiter=',')
# print(formatted_time_series.shape)
#
# dtwrelation=cdist(formatted_time_series[:100],formatted_time_series[:100])
#
# print(dtwrelation)
