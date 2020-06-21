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
import tensorflow as tf

#..............................activate from here..................
# TCN=False
# if TCN == False:
#     s_atuto = pd.DataFrame(index=['Nbeat'])
#     case = 1
#     test = True
#     if test == True:
#         level_range = [12]
#     else:
#         level_range = np.arange(1, 13)
#
#     for level in level_range:
#
#         if level == 10:
#             node_size=set(pd.read_csv('Results/selected_ts_lv10.csv',index_col=0).values.ravel())
#             # node_size = np.random.choice(list(range(get_max_node(level))), 100)
#             print('running')
#         elif level==11:
#             node_size = set(pd.read_csv('Results/selected_ts_lv11.csv', index_col=0).values.ravel())
#         elif level==12:
#             node_size = set(pd.read_csv('Results/selected_ts_lv12.csv', index_col=0).values.ravel())
#         else:
#             node_size = list(range(get_max_node(level)))
#         print('.....................................', node_size, '....................................')
#         for node in node_size:
#             start = time.time()
#             print(level, node)
#             print('case:', case)
#             _, _, rmsse_nbeat = s_nbeat(level=level, node=node, normalized=True, decomposition=True, exo=True, n_in=30,
#                                         n_out=1)
#             # _, _, rmsse_ntcn = s_tcn(level=level, node=node, normalized=True, decomposition=True, exo=True, n_in=10,
#             #                          n_out=1)
#             s_atuto[str(level) + '_' + str(node)] = [rmsse_nbeat]
#             s_atuto.to_csv('Data/s_auto.csv')
#             case += 1
#             end = time.time()
#             ts = end - start
#             print('Time spend: %f' % (ts))


#............................................#..............................#
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
n_beat_model=n_beat_model[::-1]
# print
print(len(n_beat_model))
path1 = 'Data2 '
path2 = 'sales_train_validation.csv'
path3 = 'calendar.csv'
path4 = 'sell_prices.csv'
sale = pd.read_csv('Data2/sales_train_evaluation.csv')
calendar = pd.read_csv('Data2/calendar.csv')
price = pd.read_csv('Data2/sell_prices.csv')
# levels=lc.LevelsCreater()
# df = levels.get_level(sale, level)
# df = df.transpose()
# df = pd.DataFrame(df[df.columns[node]])
ex = exovar.exovar()

def dicofnode(s_prediction):
    s_prediction[s_prediction[s_prediction.columns[0]] == 1].index
    groupts = {}
    for i in range(100):
        groupts['cluster_' + str(i)] = s_prediction[s_prediction[s_prediction.columns[0]] == i].index.ravel()
    return groupts
def get_key(val,groupts):
    for key,value in groupts.items():
        if val in [x for x in value]:
            return key



test=False
if test==True:
    length=[0,1,2,3]
    horizon=range(5)
else:
    length=range(len(n_beat_model))
    horizon=range(28)

s_prediction=pd.DataFrame()
for i in length:
    print(n_beat_model[i])
    node=int(n_beat_model[i][:-3].split('_')[-1])
    level=int(n_beat_model[i][:-3].split('_')[-2])
    print('Level:',level)
    print('node:',node)



    #......................update ts_level and run in for loop aha............#
    model = NBeatsNet.load(os.path.join('n_beat_models', n_beat_model[0]))
    if level==12 or level==11:
        pass
    elif level==10:
        clus_pred = pd.read_csv('Results/clusterprediction_lv{level:d}.csv'.format(level=level), index_col=0)
        groupts=dicofnode(clus_pred)
        othernode=groupts[get_key(node,groupts)]

        for node in othernode:
            print('node:',node)
            an = []
            ts_level = lc.LevelsCreater().get_level(sale, level)
            for h in horizon:
                print('horizon-', h)
                # print(ts_level.shape)
                salecal = ex.salecal(ts_level, calendar, node)
                if level == 1:
                    salecal = salecal.rename(columns={salecal.columns[0]: 'zero'})
                print(salecal.shape)
                x = salecal[salecal.columns[0]].values[-30:].reshape(1, 30, 1)
                x = tf.cast(x, tf.float32)
                e = salecal[salecal.columns[1:]].values[-30:].reshape(1, 30, len(salecal.columns[1:]))
                e = tf.cast(e, tf.float32)

                prediction = model.predict([x, e])
                # print(prediction.shape)
                # print(prediction.ravel().shape)

                ts_level['d_' + str(int(ts_level.columns[-1].split('_')[1]) + 1)] = np.zeros(get_max_node(level))
                ts_level.iloc[node, -1] = int(prediction.ravel())
                an.append(int(prediction.ravel()))
            s_prediction[str(level) + '_' + str(node)] = an
            s_prediction.to_csv('Results/s_prediction.csv')

    else:

        an=[]
        ts_level = lc.LevelsCreater().get_level(sale, level)
        for h in horizon:
            print('horizon-',h)
            # print(ts_level.shape)
            salecal = ex.salecal(ts_level, calendar, node)
            if level==1:
                salecal=salecal.rename(columns={salecal.columns[0]:'zero'})
            print(salecal.shape)
            x=salecal[salecal.columns[0]].values[-30:].reshape(1,30,1)
            x= tf.cast(x,tf.float32)
            e=salecal[salecal.columns[1:]].values[-30:].reshape(1,30,len(salecal.columns[1:]))
            e= tf.cast(e,tf.float32)



            prediction = model.predict([x,e])
            # print(prediction.shape)
            # print(prediction.ravel().shape)

            ts_level['d_'+str(int(ts_level.columns[-1].split('_')[1])+1)]=np.zeros(get_max_node(level))
            ts_level.iloc[node,-1]=int(prediction.ravel())
            an.append(int(prediction.ravel()))
        s_prediction[str(level)+'_'+str(node)]=an
        s_prediction.to_csv('Results/s_prediction.csv')
    # print(ts_level)
# print(s_prediction)
