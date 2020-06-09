from sujallvntcn import s_tcn
from sujallvnbeat import s_nbeat
from functions import get_max_node
import  numpy as np
import pandas as pd
import time
import os
import LevelsCreater as lc
from functions import series_to_supervised
import exovar
from nbeats_keras.model import NBeatsNet

#..............................activate from here..................
# s_atuto=pd.DataFrame(index=['Nbeat','Ntcn'])
# case=1
# test=True
# if test==True:
#     level_range=[10]
# else:
#     level_range=np.arange(1,13)
#
# for level in level_range:
#
#     if level==10 or level==11 or level==12:
#         node_size=np.random.choice(list(range(get_max_node(level))),100)
#         print('running')
#     else:
#         node_size=list(range(get_max_node(level)))
#     print('.....................................',node_size,'....................................')
#     for node in node_size:
#         start=time.time()
#         print(level,node)
#         print('case:',case)
#         _,_,rmsse_nbeat=s_nbeat(level=level,node=node,normalized=True,decomposition=True,exo=True,n_in=30,n_out=1)
#         _,_,rmsse_ntcn=s_tcn(level=level,node=node,normalized=True,decomposition=True,exo=True,n_in=10,n_out=1)
#         s_atuto[str(level)+'_'+str(node)]=[rmsse_nbeat,rmsse_ntcn]
#         s_atuto.to_csv('Data/s_auto.csv')
#         case+=1
#         end=time.time   ()
#         ts=end-start
#         print('Time spend: %f' %(ts))





#........................getting predictions................................

n_beat_model=next(os.walk('n_beat_models'))[2]
# print(n_beat_model)
node=int(n_beat_model[0][:-3].split('_')[-1])
level=int(n_beat_model[0][:-3].split('_')[-2])
print(level)

path1 = 'Data2 '
path2 = 'sales_train_validation.csv'
path3 = 'calendar.csv'
path4='sell_prices.csv'
sale = pd.read_csv('Data2/sales_train_validation.csv')
calendar = pd.read_csv('Data2/calendar.csv')
price = pd.read_csv('Data2/sell_prices.csv')
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
bigx=series_to_supervised(salecal,30,1,parse=False,dropnan=True)
# x, y, e = series_to_supervised(salecal, n_in, n_out, dropnan=True, parse=False)
y_columsn = str(salecal.columns[0]) + '(t)'
look_back_columns = list(set([i for i in bigx.columns if str(salecal.columns[0]) in i]) - set([y_columsn]))
# look_back_columns.remove(y_columsn)
exo_column = [i for i in bigx.columns if '(t)' not in i and i not in look_back_columns]
x,y,e=bigx[look_back_columns],bigx[y_columsn],bigx[exo_column]
x = x.values.reshape(salecal.shape[0]-30 , 30, 1)
y = y.values.reshape(salecal.shape[0]-30 , 1, 1)
e = e.values.reshape(salecal.shape[0]-30 , 30, 12)
model = NBeatsNet.load(os.path.join('n_beat_models',n_beat_model[0]))

prediction = model.predict([x,e])
print(prediction.shape)
print(y.shape)
# for level in level_range:
#     if level==10 or level==11 or level==12:
#         node_size=np.random.choice(list(range(get_max_node(level))),100)
#         print('running')
#     else:
#         node_size=list(range(get_max_node(level)))
#
#     for node in node_size:
