class exovar:
    def __init__(self):
        pass
    def calendar(self,calendar_data):
        CAL_DTYPES = {"event_name_1": "category", "event_name_2": "category", "event_type_1": "category",
                      "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
                      "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32',
                      'snap_WI': 'float32'}
        for col_name, col_fit in CAL_DTYPES.items():
            if col_name in calendar_data.columns:
                calendar_data[col_name] = calendar_data[col_name].astype(col_fit)

        for col_name, col_fit in CAL_DTYPES.items():
            if col_fit == 'category':
                calendar_data[col_name] = calendar_data[col_name].cat.codes.astype('int16')
                calendar_data[col_name] -= calendar_data[col_name].min()
        calendar_data = calendar_data.drop(['date'], axis=1)

        return calendar_data


    def salecal(self,sale_data,calendar_data,node=1):
        CAL_DTYPES = {"event_name_1": "category", "event_name_2": "category", "event_type_1": "category",
                      "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
                      "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32',
                      'snap_WI': 'float32'}
        for col_name, col_fit in CAL_DTYPES.items():
            if col_name in calendar_data.columns:
                calendar_data[col_name] = calendar_data[col_name].astype(col_fit)

        for col_name, col_fit in CAL_DTYPES.items():
            if col_fit == 'category':
                calendar_data[col_name] = calendar_data[col_name].cat.codes.astype('int16')
                calendar_data[col_name] -= calendar_data[col_name].min()
        sale_data=sale_data.T[[sale_data.T.columns[node]]]
        sale_data.index.name = 'd'
        store_level_final = sale_data.merge(calendar_data, on='d')
        new_store_level = store_level_final.drop(['d', 'date'], axis=1)
        return new_store_level

    def salcaltwo(self,sale_data,calendar_data,price_data,node):
        # CAL_DTYPES = {"event_name_1": "category", "event_name_2": "category", "event_type_1": "category",
        #               "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        #               "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32',
        #               'snap_WI': 'float32'}
        # for col_name, col_fit in CAL_DTYPES.items():
        #     if col_name in calendar_data.columns:
        #         calendar_data[col_name] = calendar_data[col_name].astype(col_fit)
        #
        # for col_name, col_fit in CAL_DTYPES.items():
        #     if col_fit == 'category':
        #         calendar_data[col_name] = calendar_data[col_name].cat.codes.astype('int16')
        #         calendar_data[col_name] -= calendar_data[col_name].min()

        calendar_data=self.calendar(calendar_data)
        sale_data = sale_data.T[[sale_data.T.columns[node]]]
        sale_data.index.name = 'd'
        calendar_data=calendar_data.merge(price_data, on='wm_yr_wk')
        store_level_final = sale_data.merge(calendar_data, on='d')
        new_store_level = store_level_final.drop(['d'], axis=1)
        return new_store_level



if __name__ == "__main__":
    import pandas as pd
    import os
    import LevelsCreater as lc
    import exovar as ex
    path1 = 'Data2'
    path2 = 'calendar.csv'
    path3 = 'sales_train_evaluation.csv'
    path4='sell_prices.csv'
    calendar = pd.read_csv(os.path.join(path1, path2))
    sale = pd.read_csv(os.path.join(path1, path3))
    price=pd.read_csv(os.path.join(path1,path4))
    levels = lc.LevelsCreater()
    level6 = levels.get_level(sale, 6)
    ex = ex.exovar()
    leveltwithexo=ex.salcaltwo(sale, calendar, price, 1)
    print(leveltwithexo)
