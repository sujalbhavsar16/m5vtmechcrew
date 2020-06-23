import pandas as pd
import numpy as np
import LevelsCreater as lc
import os
from tslearn.clustering import KShape
from tslearn.utils import to_time_series_dataset
from scipy.spatial.distance import cdist

path1 = 'Data2 '
path2 = 'sales_train_validation.csv'
sale=pd.read_csv(os.path.join('Data2','sales_train_validation.csv'))
levels=lc.LevelsCreater()
df = levels.get_level(sale, 12)

print(df)

# ...........Making input environment..............
my_time_series=[]

for i in range(df.shape[0]):
# for i in range(50):
    my_time_series.append(df.iloc[i].values)
formatted_dataset = to_time_series_dataset(my_time_series)

print(formatted_dataset.shape)

ks=KShape(n_clusters=100,verbose=True)
# ks=KShape(n_clusters=10,verbose=True)
y_pred=ks.fit_predict(formatted_dataset)
print(y_pred)
centroid=ks.cluster_centers_
centroid=centroid.reshape((centroid.shape[0],centroid.shape[1]))
print(centroid.shape)
# np.savetxt("Results/centroid.csv", centroid, delimiter=",")
pd.DataFrame(centroid).to_csv('Results/centroid_lv10.csv')

#
D=cdist(formatted_dataset.reshape((formatted_dataset.shape[0],formatted_dataset.shape[1])),centroid)
print(D.shape)
selected_ts=np.argmin(D.T,axis=1)
pd.DataFrame(selected_ts).to_csv('Results/selected_ts_lv12.csv')
pd.DataFrame(y_pred).to_csv('Results/clusterprediction_lv12.csv')
print(selected_ts.shape)
print(selected_ts)