import pandas as pd
import numpy as np
import LevelsCreater as lc
import os

def series_to_supervised(data, n_in=1, n_out=1, dropnan=False,parse=False):
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

    if parse == False:
        return cols
    else:
        if n_vars==1:
            print('Sorry: exogenous variables are missing')
        else:
            exo_columns = [i for i in cols.columns if '(t)' in i and data.columns[0] not in i]
            y_columsn = data.columns[0] + '(t)'
            look_back_columns = list(set([i for i in cols.columns if data.columns[0] in i]) - set(y_columsn))
        return cols[look_back_columns],cols[y_columsn],cols[exo_columns]

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100




def RMSSE(y_pred,y_test,y):
    y_test,y = y_test,y
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


def get_max_node(level):
    levels = lc.LevelsCreater()
    path1 = 'Data'
    path2 = 'sales_train_validation.csv'
    sale = pd.read_csv(os.path.join(path1, path2), delimiter=",")
    df = levels.get_level(sale, level)
    df = df.transpose()
    return len(df.columns)

if __name__ == "__main__":
    from functions import series_to_supervised
    import exovar
    import os
    import pandas as pd
    import LevelsCreater as lc
    from functions import series_to_supervised

    path1 = 'Data'
    path2 = 'calendar.csv'
    path3 = 'sales_train_validation.csv'
    calendar = pd.read_csv(os.path.join(path1, path2))
    sale = pd.read_csv(os.path.join(path1, path3))
    ex = exovar.exovar()
    # calendar=ex.calendar(calendar)
    levels = lc.LevelsCreater()
    level6 = levels.get_level(sale, 6)
    salecal = ex.salecal(level6, calendar, 1)
    x, y, e = series_to_supervised(salecal, 5, 1, dropnan=True, parse=True)
    print(x)
    print(e)
    print(y)