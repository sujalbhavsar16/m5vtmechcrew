import pandas as pd
import numpy as np

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


