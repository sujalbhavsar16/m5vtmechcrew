from tensorflow.keras.layers import Dense
from tensorflow.keras import Input, Model

from tcn import TCN, tcn_full_summary
import import_ipynb
import LevelsCreater as lc
import pandas as pd
import os

def series_to_supervised(data, n_in=1, n_out=1, dropnan=False):
    n_vars = 1 if type(data) is list else data.shape[1]
    cols = pd.DataFrame()
    names = list()
    for i in range(n_in, 0, -1):
        df = pd.DataFrame()
        names = list()
        df = data.shift(i)
        names += [('%s(t-%d)' % (data.columns[j], i)) for j in range(n_vars)]
        df.columns = names
        cols = pd.concat([cols, df], axis=1, sort=False)
    for i in range(0, n_out):
        df = pd.DataFrame()
        names = list()
        df = data.shift(-i)
        if i == 0:
            names += [('%s(t)' % (data.columns[j])) for j in range(n_vars)]
        else:
            names += [('%s(t+%d)' % (data.columns[j], i)) for j in range(n_vars)]

        df.columns = names
        cols = pd.concat([cols, df], axis=1, sort=False)
    if dropnan:
        cols.dropna(inplace=True)
    return cols


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


levels = lc.LevelsCreater()

path1 = 'Data'
path2 = 'sales_train_validation.csv'

data = pd.read_csv(os.path.join(path1, path2), delimiter=",")
df = levels.level_3(data)
df = df.transpose()
df = pd.DataFrame(df['CA_1'])
# df['d'] = df.index
#
# path3 = 'calendar.csv'
# calender = pd.read_csv(os.path.join(path1, path3))
# store_level_final = df.merge(calender, on='d')
# new_store_level=store_level_final.drop(['d','date'],axis=1)
mw=30
nsl_sts=series_to_supervised(df,mw,1,dropnan=True)
x=nsl_sts[nsl_sts.columns[:-1]].values
y=nsl_sts[nsl_sts.columns[-1]].values

x_reshape=x.reshape((x.shape[0],x.shape[1],1))
y_reshape=y.reshape((y.shape[0],1))



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


batch_size, timesteps, input_dim = 128, 30, 1

def get_x_y(size=1000):
    import numpy as np
    pos_indices = np.random.choice(size, size=int(size // 2), replace=False)
    x_train = np.zeros(shape=(size, timesteps, 1))
    y_train = np.zeros(shape=(size, 1))
    x_train[pos_indices, 0] = 1.0
    y_train[pos_indices, 0] = 1.0
    return x_train, y_train

i = Input(batch_shape=(batch_size, timesteps, input_dim))

o = TCN(return_sequences=False)(i)  # The TCN layers are here.
o = Dense(1)(o)

m = Model(inputs=[i], outputs=[o])
m.compile(optimizer='adam', loss='mse')

tcn_full_summary(m, expand_residual_blocks=False)

c = 121
x_train, y_train, x_test, y_test = x_reshape[:-c], y_reshape[:-c], x_reshape[-c:], y_reshape[-c:]

m.fit(x_train, y_train, epochs=100, validation_split=0.2)