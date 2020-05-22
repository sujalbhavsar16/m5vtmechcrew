#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 30,5
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[13]:


class LevelsCreater:
    def __init__(self):
        pass
    def get_level(self,sales_data,level):
        if level==11:
            df = sales_data.groupby(['item_id', 'state_id'], as_index=False).sum()
            df=df.set_index(df.columns[0])
            df.index=df.index+'_'+df[df.columns[0]]
            df=df.drop(df.columns[0],axis=1)
        elif level==10:
            df= sales_data.groupby(['item_id'], as_index=False).sum()
            df=df.set_index(df.columns[0])
        elif level==9:
            df= sales_data.groupby(['store_id', 'dept_id'], as_index=False).sum()
            df=df.set_index(df.columns[0])
            df.index=df.index+'_'+df[df.columns[0]]
            df=df.drop(df.columns[0],axis=1)
        elif level==8:
            df= sales_data.groupby(['store_id', 'cat_id'], as_index=False).sum()
            df=df.set_index(df.columns[0])
            df.index=df.index+'_'+df[df.columns[0]]
            df=df.drop(df.columns[0],axis=1)
        elif level==7:
            df= sales_data.groupby(['state_id', 'dept_id'], as_index=False).sum()
            df=df.set_index(df.columns[0])
            df.index=df.index+'_'+df[df.columns[0]]
            df=df.drop(df.columns[0],axis=1)
        elif level==6:
            df= sales_data.groupby(['state_id', 'cat_id'], as_index=False).sum()
            df=df.set_index(df.columns[0])
            df.index=df.index+'_'+df[df.columns[0]]
            df=df.drop(df.columns[0],axis=1)
        elif level==5:
            df= sales_data.groupby(['dept_id'], as_index=False).sum()
            df=df.set_index(df.columns[0])
        elif level==4:
            df= sales_data.groupby(['cat_id'], as_index=False).sum()
            df=df.set_index(df.columns[0])
        elif level==3:
            df= sales_data.groupby(['store_id'], as_index=False).sum()
            df=df.set_index(df.columns[0])
        elif level==2:
            df= sales_data.groupby(['state_id'], as_index=False).sum()
            df=df.set_index(df.columns[0])
        elif level==1:
            df= (pd.DataFrame(self.level_2(sales_data).sum())).transpose()
        else:
            df=sales_data[[t for t in sales_data.columns if 'd_' in t]]
            
        return df
    
    def level_11(self,sales_data):
        df = sales_data.groupby(['item_id', 'state_id'], as_index=False).sum()
        return df.set_index(df.columns[0])
    def level_10(self,sales_data):
        df= sales_data.groupby(['item_id'], as_index=False).sum()
        return df.set_index(df.columns[0])
    
    def level_9(self,sales_data):
        df= sales_data.groupby(['store_id', 'dept_id'], as_index=False).sum()
        return df.set_index(df.columns[0])
    
    def level_8(self,sales_data):
        df= sales_data.groupby(['store_id', 'cat_id'], as_index=False).sum()
        return df.set_index(df.columns[0])
    
    def level_7(self,sales_data):
        df= sales_data.groupby(['state_id', 'dept_id'], as_index=False).sum()
        return df.set_index(df.columns[0])
    def level_6(self,sales_data):
        df= sales_data.groupby(['state_id', 'cat_id'], as_index=False).sum()
        return df.set_index(df.columns[0])
    def level_5(self,sales_data):
        df= sales_data.groupby(['dept_id'], as_index=False).sum()
        return df.set_index(df.columns[0])
    def level_4(self,sales_data):
        df= sales_data.groupby(['cat_id'], as_index=False).sum()
        return df.set_index(df.columns[0])
    def level_3(self,sales_data):
        df= sales_data.groupby(['store_id'], as_index=False).sum()
        return df.set_index(df.columns[0])
    def level_2(self,sales_data):
        df= sales_data.groupby(['state_id'], as_index=False).sum()
        return df.set_index(df.columns[0])
    def level_1(self,sales_data):
        df= (pd.DataFrame(self.level_2(sales_data).sum())).transpose()
        return df        


# In[14]:


# Generatelev = LevelsCreater()
# level11 = Generatelev.level_11(sale)
# level10 = Generatelev.level_10(sale)
# level9 = Generatelev.level_9(sale)
# level8 = Generatelev.level_8(sale)
# level7 = Generatelev.level_7(sale)
# level6 = Generatelev.level_6(sale)
# level5 = Generatelev.level_5(sale)
# level4 = Generatelev.level_4(sale)
# level3 = Generatelev.level_3(sale)
# level2 = Generatelev.level_2(sale)
# level1 = Generatelev.level_1(sale)
        

