from tensorflow.keras.layers import Dense
from tensorflow.keras import Input, Model
from functions import series_to_supervised
from functions import mean_absolute_percentage_error
from functions import RMSSE
from functions import get_max_node
from tcn import TCN, tcn_full_summary
import import_ipynb
import LevelsCreater as lc
import pandas as pd
import os
from numpy import linalg as LA
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose

def s_nbeat(level=1,exo=False,node=0,normalized=False,n_in=30,n_out=1,batch_size=100,optimizer='adam', loss='mse',test=200,epochs=100,decomposition=False):

    max_node=get_max_node(level)
    if node>(max_node-1):
        print(f'Exceed the limit: Maximum number of node for level {level} is {max_node}')
        return None,None,None
    print('level %s' %level+':'+ 'node %s' %node)
    levels = lc.LevelsCreater()
    path1 = 'Data'
    path2 = 'sales_train_validation.csv'
    sale = pd.read_csv(os.path.join(path1, path2), delimiter=",")
    if exo==False:
        df = levels.get_level(sale,level)

        df = df.transpose()
        df = pd.DataFrame(df[df.columns[node]])
        print('shape of a time-series is %s' %{df.shape})
    # df['d'] = df.index
    #
    # path3 = 'calendar.csv'
    # calender = pd.read_csv(os.path.join(path1, path3))
    # store_level_final = df.merge(calender, on='d')
    # new_store_level=store_level_final.drop(['d','date'],axis=1)
    # mw=30
        if decomposition==True:
            res=seasonal_decompose(df.values, model='additive',period=1)
            res=[res.resid,res.seasonal,res.trend]
            ytot=[]
            for df in res:
                print('dealing with decompositioin')
                df=pd.DataFrame(df)
                nsl_sts = series_to_supervised(df, n_in, n_out, dropnan=True)
                x = nsl_sts[nsl_sts.columns[:n_in]].values
                y = nsl_sts[nsl_sts.columns[-n_out:]].values
                if normalized == True:
                    scaler = StandardScaler()
                    x = scaler.fit_transform(x)

                print('input shape to model: %s' % {x.shape})
                print('output shape to model: %s' % {y.shape})

                x_reshape = x.reshape((x.shape[0], x.shape[1], 1))
                y_reshape = y.reshape((y.shape[0], 1))

                batch_size, timesteps, input_dim = batch_size, n_in, n_out
                i = Input(batch_shape=(batch_size, timesteps, input_dim))

                o = TCN(return_sequences=False)(i)  # The TCN layers are here.
                o = Dense(1)(o)

                m = Model(inputs=[i], outputs=[o])
                m.compile(optimizer, loss)

                tcn_full_summary(m, expand_residual_blocks=False)

                c = test
                x_train, y_train, x_test, y_test = x_reshape[:-c], y_reshape[:-c], x_reshape[-c:], y_reshape[-c:]
                print('training x size: %s' % {x_train.shape})
                print('training y size: %s' % {y_train.shape})
                m.fit(x_train, y_train, epochs, validation_split=0.2)

                y_pred_test = np.round(m.predict(x_test))
                print('shape of y_pred of decomposition : %s' %{y_pred_test.shape})
                ytot.append(y_pred_test)
            y_pred_test=np.sum(ytot,axis=0)
            print('Final shape of y_pred: %s'%{y_pred_test.shape})
        else:

            nsl_sts=series_to_supervised(df,n_in,n_out,dropnan=True)
            x=nsl_sts[nsl_sts.columns[:n_in]].values
            y=nsl_sts[nsl_sts.columns[-n_out:]].values
            if normalized==True:
                scaler = StandardScaler()
                x=scaler.fit_transform(x)

            print('input shape to model: %s' %{x.shape})
            print('output shape to model: %s' % {y.shape})

            x_reshape=x.reshape((x.shape[0],x.shape[1],1))
            y_reshape=y.reshape((y.shape[0],1))

            batch_size, timesteps, input_dim = batch_size, n_in, n_out
            i = Input(batch_shape=(batch_size, timesteps, input_dim))

            o = TCN(return_sequences=False)(i)  # The TCN layers are here.
            o = Dense(1)(o)

            m = Model(inputs=[i], outputs=[o])
            m.compile(optimizer, loss)

            tcn_full_summary(m, expand_residual_blocks=False)


            c = test
            x_train, y_train, x_test, y_test = x_reshape[:-c], y_reshape[:-c], x_reshape[-c:], y_reshape[-c:]
            print('training x size: %s'%{x_train.shape})
            print('training y size: %s' %{y_train.shape})
            m.fit(x_train, y_train, epochs, validation_split=0.2)

            y_pred_test=np.round(m.predict(x_test))

    return mean_absolute_percentage_error(y_pred_test,y_test),LA.norm(y_pred_test-y_test,2),RMSSE(y_pred_test,y_test,df.values)

# CAL_DTYPES = {"event_name_1": "category", "event_name_2": "category", "event_type_1": "category",
#               "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
#               "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32'}
#
# for col_name, col_fit in CAL_DTYPES.items():
#     if col_name in store_level_final.columns:
#         store_level_final[col_name] = store_level_final[col_name].astype(col_fit)
#
# for col_name, col_fit in CAL_DTYPES.items():
#     if col_fit == 'category':
#         store_level_final[col_name] = store_level_final[col_name].cat.codes.astype('int16')
#         store_level_final[col_name] -= store_level_final[col_name].min()
#
# new_store_level = store_level_final.drop(['d', 'date'], axis=1)
# mw=30
# nsl_sts=series_to_supervised(new_store_level,mw,1)




# def get_x_y(size=1000):
#     import numpy as np
#     pos_indices = np.random.choice(size, size=int(size // 2), replace=False)
#     x_train = np.zeros(shape=(size, timesteps, 1))
#     y_train = np.zeros(shape=(size, 1))
#     x_train[pos_indices, 0] = 1.0
#     y_train[pos_indices, 0] = 1.0
#     return x_train, y_train

if __name__ == "__main__":
    mape,rmse,rmsse=s_nbeat(level=3,node=0,normalized=True,decomposition=True)
    print('MAPE on testing set: %f' %mape)
    print('RMSE on testing set: %f' %rmse)
    print('RMSSE on testin set: %f'%rmsse)

