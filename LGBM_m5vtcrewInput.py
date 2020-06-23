#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ''' S.M.Ali Mousavi
# last saved at 2020-05-19 20:19:04 '''

# Legend:
#     HC : Helper Class
#     HF : Helper Function


# In[2]:


# General imports
import numpy as np
import pandas as pd
import os, sys, gc, time, warnings, pickle, psutil, random
import lightgbm as lgb

# custom imports
from multiprocessing import Pool        # Multiprocess Runs

warnings.filterwarnings('ignore')


# In[3]:


import import_ipynb
import matplotlib.pyplot as plt
import LevelsCreater as lc
from sklearn.metrics import mean_absolute_error as MAE
from functions import mean_absolute_percentage_error, RMSSE, test_stationarity
from numpy import linalg as LA
from exovar import exovar
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.metrics import mean_absolute_error as MAE 


# # HF: Seeding

# In[4]:


########################### Helpers
#################################################################################
## Seeder
# :seed to make all processes deterministic     # type: int
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)

    
## Multiprocess Runs
def df_parallelize_run(func, t_split):
    num_cores = np.min([N_CORES,len(t_split)])
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, t_split), axis=1)
    pool.close()
    pool.join()
    return df


# # LGBM Model Hyper Parameters setting

# In[ ]:


########################### Model params
#################################################################################
### https://lightgbm.readthedocs.io/en/latest/Parameters.html ###
### https://neptune.ai/blog/lightgbm-parameters-guide
import lightgbm as lgb
lgb_params = {
                    'boosting_type': 'gbdt',
                    'objective': 'tweedie',
                    'tweedie_variance_power': 1.1,
                    'metric': 'rmse',
                    'subsample': 0.5,
                    'subsample_freq': 1,
                    'learning_rate': 0.03,
                    'num_leaves': 2**11-1,
                    'min_data_in_leaf': 2**12-1,
                    'feature_fraction': 0.5,
                    'max_bin': 100,
                    'n_estimators': 1400,
                    'boost_from_average': False,
                    'verbose': 0, # < 0: Fatal, = 0: Error (Warning), = 1: Info, > 1: Debug
                } 

# Let's look closer on params

## 'boosting_type': 'gbdt'
# we have 'goss' option for faster training
# but it normally leads to underfit.
# Also there is good 'dart' mode
# but it takes forever to train
# and model performance depends 
# a lot on random factor 
# https://www.kaggle.com/c/home-credit-default-risk/discussion/60921

## 'objective': 'tweedie'
# Tweedie Gradient Boosting for Extremely
# Unbalanced Zero-inflated Data
# https://arxiv.org/pdf/1811.10192.pdf
# and many more articles about tweediie
#
# Strange (for me) but Tweedie is close in results
# to my own ugly loss.
# My advice here - make OWN LOSS function
# https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/140564
# https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/143070
# I think many of you already using it (after poisson kernel appeared) 
# (kagglers are very good with "params" testing and tuning).
# Try to figure out why Tweedie works.
# probably it will show you new features options
# or data transformation (Target transformation?).

## 'tweedie_variance_power': 1.1
# default = 1.5
# set this closer to 2 to shift towards a Gamma distribution
# set this closer to 1 to shift towards a Poisson distribution
# my CV shows 1.1 is optimal 
# but you can make your own choice

## 'metric': 'rmse'
# Doesn't mean anything to us
# as competition metric is different
# and we don't use early stoppings here.
# So rmse serves just for general 
# model performance overview.
# Also we use "fake" validation set
# (as it makes part of the training set)
# so even general rmse score doesn't mean anything))
# https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/133834

## 'subsample': 0.5
# Serves to fight with overfit
# this will randomly select part of data without resampling
# Chosen by CV (my CV can be wrong!)
# Next kernel will be about CV

##'subsample_freq': 1
# frequency for bagging
# default value - seems ok

## 'learning_rate': 0.03
# Chosen by CV
# Smaller - longer training
# but there is an option to stop 
# in "local minimum"
# Bigger - faster training
# but there is a chance to
# not find "global minimum" minimum

## 'num_leaves': 2**11-1
## 'min_data_in_leaf': 2**12-1
# Force model to use more features
# We need it to reduce "recursive"
# error impact.
# Also it leads to overfit
# that's why we use small 
# 'max_bin': 100

## l1, l2 regularizations
# https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c
# Good tiny explanation
# l2 can work with bigger num_leaves
# but my CV doesn't show boost
                    
## 'n_estimators': 1400
# CV shows that there should be
# different values for each state/store.
# Current value was chosen 
# for general purpose.
# As we don't use any early stopings
# careful to not overfit Public LB.

##'feature_fraction': 0.5
# LightGBM will randomly select 
# part of features on each iteration (tree).
# We have maaaany features
# and many of them are "duplicates"
# and many just "noise"
# good values here - 0.5-0.7 (by CV)

## 'boost_from_average': False
# There is some "problem"
# to code boost_from_average for 
# custom loss
# 'True' makes training faster
# BUT carefull use it
# https://github.com/microsoft/LightGBM/issues/1514
# not our case but good to know cons


# # Setting global variables

# In[6]:


########################### Vars
#################################################################################
VER = 1                          # Our model version
SEED = 42                        # We want all things
seed_everything(SEED)            # to be as deterministic 
lgb_params['seed'] = SEED        # as possible
N_CORES = psutil.cpu_count()     # Available CPU cores


#LIMITS and const
TARGET      = 'sales'            # Our target
START_TRAIN = 0                  # We can skip some rows (Nans/faster training)
END_TRAIN   = 1913               # End day of our train set
P_HORIZON   = 12                 # Prediction horizon #vtM5crew 12days #DarkMagic=28days
USE_AUX     = True               # Use or not pretrained models

#FEATURES to remove
## These features lead to overfit
## or values not present in test set
remove_features = ['id','state_id','store_id',
                   'date','wm_yr_wk','d',TARGET]
mean_features   = ['enc_cat_id_mean','enc_cat_id_std',
                   'enc_dept_id_mean','enc_dept_id_std',
                   'enc_item_id_mean','enc_item_id_std'] 

#PATHS for Features
Absolute = '' # 'G://My Drive/Machine Learning/Kaggle Projects/M5/m5vtmechcrew/'
Cache    = 'Cache/'
ORIGINAL = 'Data2/'
SALES    = 'sales_train_evaluation.csv'
Calendar = 'calendar.csv' # calendar csv from kaggle
BASE     = Absolute+'m5-simple-fe/grid_part_1.pkl'
PRICE    = Absolute+'m5-simple-fe/grid_part_2.pkl'
CALENDAR = Absolute+'m5-simple-fe/grid_part_3.pkl'  # calendar csv edited
LAGS     = 'm5-lags-features/lags_df_28.pkl'
MEAN_ENC = 'm5-custom-features/mean_encoding_df.pkl'


# AUX(pretrained) Models paths
AUX_MODELS = '../input/m5-aux-models/'


#STORES ids
STORES_IDS = pd.read_csv(Absolute+ORIGINAL+SALES)['store_id']
STORES_IDS = list(STORES_IDS.unique())


#SPLITS for lags creation
SHIFT_DAY  = 28
N_LAGS     = 15
LAGS_SPLIT = [col for col in range(SHIFT_DAY,SHIFT_DAY+N_LAGS)]
ROLS_SPLIT = []
for i in [1,7,14]:
    for j in [7,14,30,60]:
        ROLS_SPLIT.append([i,j])


# # HC: Level Creator

# In[7]:


class LevelsCreator:
    def __init__(self):
        pass
    
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


# # HF: Loading data

# In[8]:


# import LevelsCreater as lc

# SALES='sales_train_validation.csv'
sale=pd.read_csv(Absolute+ORIGINAL+SALES); sale.head()


# In[9]:


# levels = LevelsCreator()
# level3 = levels.level_3(sale); level3.head()


# In[10]:


# t = level3.loc[['CA_1']].T.rename_axis('days', axis=1); tt=t.shift(2); t.head()


# # HF: Categorical_to_sparse 
# (Calendar exogenic variables to Sparse)

# In[11]:


def categorical_to_sparse(calendar_data,node=1):
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
    return calendar_data


# # HF: series_to_supervised

# In[12]:


# series_to_supervised(LevelsCreator().level_3(sale), n_in=3)


# In[13]:


# # draft
# LevelsCreator().level_3(sale)


# In[14]:


# # draft
data_ = lc.LevelsCreater().get_level(sale,level=2); data_.head()


# In[15]:


# # draft
# data =LevelsCreator().level_6(sale); data.head()


# In[16]:


# # draft
# series_to_supervised_l3(data2, node=2, n_in=3, n_out=2, exo=True, dropnan=False)


# In[18]:


def series_to_supervised_l3(data, node='CA_1', n_in=1, n_out=1, exo=False, dropnan=False):

    n_vars = 1 if type(data) is list else data.shape[1]
    data_shifted=pd.DataFrame()
    names=list()
    # Selecting a store of interest, 
    # Transposing the days from being column to being the index
    data=data.iloc[[node]].T
    
    # looping over the lag window(n_in) and making a new column
    for i in range(n_in, 0, -1):
        df=pd.DataFrame() #is this step necessary ?
        col_shifted=data.shift(i)
        col_shifted.columns=[f'{node}_t-{i}']
        data_shifted = pd.concat([data_shifted,col_shifted], axis=1, sort=False)

#     data_shifted.columns=col_names
    
    for i in range(0, n_out):
        df=pd.DataFrame()
        names=list()
        df=data.shift(-i)
#         if i == 0:
        names =[f'{node}_t_out-{i}'] #[('%s(t)' % (data.columns[j])) for j in range(n_vars)]
#         else:
#             names = [('%s(t+%d)' % (data.columns[j], i)) for j in range(n_vars)]
            
        df.columns=names
        data_shifted=pd.concat([data_shifted,df],axis=1,sort=False)
    

        
   # exogenic/categorical variables
    if exo==True:        
        calendar_data = pd.read_csv(ORIGINAL+Calendar) # reading the kaggle Calendar.csv file
        calendar_data =categorical_to_sparse(calendar_data,node=node) # encode the categorial variables in Calendar file
        data_shifted = data_shifted.merge(calendar_data, left_index=True, right_on='d') # index in data_shifted is same format as "d" column in Calendar, format: d_1*
        data_shifted = data_shifted.drop(['d', 'date'], axis=1) #dropping column 'd' and column 'date'
        
    # dropping the Nans if dropnan=True
    if dropnan:
        data_shifted.dropna(inplace=True)
        
  
    #reseting index from d_1...d_1913 to o...1912
    data_shifted.reset_index(drop=True, inplace=True)   
    # Renaming the index column from store_id to days
    data_shifted.rename_axis('days', axis=1, inplace=True) 
    
    return data_shifted


# In[19]:


series_to_supervised_l3(data_, n_in=2,node=0, exo=True)


# # HF: Train_test split  & LGB train_test DataSet builder

# In[97]:


def train_validate_splitter_l3(data, node='CA_1', n_in=2, n_out=1, exo=False, dropnan=False):
    # making supervised data
    Supervised_data = series_to_supervised_l3(data, node=node,n_in=n_in, n_out=n_out, exo=exo, dropnan=dropnan)
    nan_mask = Supervised_data.isna().any(axis=1)
#     Target_data = data.iloc[[node]].T.reset_index(drop=True)
    Feature_data = Supervised_data[Supervised_data.columns[:len(Supervised_data.columns)]]
    Target_data = Supervised_data[Supervised_data.columns[-n_out:]]
    
    #dropping rows that has one or more nan features 
    Feature_data = Feature_data[~nan_mask]
    Target_data = Target_data[~nan_mask]
#     END_TRAIN = END_TRAIN-len(nan_mask) Not#?? sure why it gives error if we redefine the END_TRAIN here it was defined in setting global variable section

    # Train (All data less than 1913)
    # "Validation" (Last 12 days - not real validation set)
    X_train = Feature_data.iloc[:END_TRAIN-len(nan_mask)-P_HORIZON]
    X_validate = Feature_data.iloc[END_TRAIN-len(nan_mask)-P_HORIZON:]
    y_train = Target_data.iloc[:END_TRAIN-len(nan_mask)-P_HORIZON]
    y_validat = Target_data.iloc[END_TRAIN-len(nan_mask)-P_HORIZON:]
    
    return X_train, X_validate, y_train, y_validat

# LGB library needs inputs in specific format of train_test dataset
# dataset includes in it X or features and y or label
def lgb_train_validate_set_builder(X_train, X_validate, y_train, y_validat):
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_validate, label=y_validat)
    return train_data, valid_data


# # Training Predict Plot `MAE`

# In[21]:


from sklearn.metrics import mean_absolute_error as MAE


# In[22]:


# #draft
# C=[]
# A = [1,1,1]
# B = [2,2,2]
# C.append(A)
# print(C)
# C.append(B)
# print(C)
# np.sum(C,axis=0)


# In[23]:


## for purpose of review the LGBM hyperparameters is copied from the section "LGBM Model Hyper Parameters setting"
# lgb_params = {
#                     'boosting_type': 'gbdt',
#                     'objective': 'tweedie',
#                     'tweedie_variance_power': 1.1,
#                     'metric': 'rmse',
#                     'subsample': 0.5,
#                     'subsample_freq': 1,
#                     'learning_rate': 0.03,
#                     'num_leaves': 2**11-1,
#                     'min_data_in_leaf': 2**12-1,
#                     'feature_fraction': 0.5,
#                     'max_bin': 100,
#                     'n_estimators': 1400,
#                     'boost_from_average': False,
#                     'verbose': -1,
#                 } 

def model_LGBM_vizualize(level=1,exo=False,node=0,normalized=False,n_in=3,n_out=1, lgb_params= lgb_params, decomposition=False):
    start = time.time()
    
    error_MAE =[]
    error_MAP =[]
    error_RMSE = []
    error_RMSSE = []
    # make the level data frame 
    levels = lc.LevelsCreater()
    data_level = levels.get_level(sale,level=level);

    lag_range = range(1,n_in)
    
    def LGBM_(data, lag):
            
            X_train, X_validate, y_train, y_validat=             train_validate_splitter_l3(data_level, node=node, n_in=lag,n_out=n_out, exo=exo, dropnan=False)

            train_data, valid_data = lgb_train_validate_set_builder(
                                         X_train, X_validate, y_train, y_validat
                                          )

            # Launch seeder again to make lgb training 100% deterministic
            # with each "code line" np.random "evolves" 
            # so we need (may want) to "reset" it
            seed_everything(SEED)
            estimator = lgb.train(lgb_params,
                                  train_data,
                                  valid_sets = [valid_data],
                                  verbose_eval = 1
                                  )

            # Save model - it's not real '.bin' but a pickle file
            # estimator = lgb.Booster(model_file='model.txt')
            # can only predict with the best iteration (or the saving iteration)
            # pickle.dump gives us more flexibility
            # like estimator.predict(TEST, num_iteration=100)
            # num_iteration - number of iteration want to predict with, 
            # NULL or <= 0 means use best iteration

            model_name = 'lgb_model_'+'node'+str(node)+'_v'+str(VER)+'.bin' #just saves last trained model
            # pickling the trained models
            if not os.path.exists(Absolute+Cache):
                os.mkdir(Absolute+Cache)
            pickle.dump(estimator, open(Absolute+Cache+model_name, 'wb'))

            # in-sample prediction
            y_predict = estimator.predict(X_train)
            print(MAE(y_train, y_predict))
            
            return y_train, y_predict
        
    # preparign for RMSE y argument
    df = levels.get_level(sale,level=level)
    df = df.transpose()
    df = pd.DataFrame(df[df.columns[node]])
        
    if decomposition==False:
        for lag in lag_range:
            print(lag)
            y_train, y_predict = LGBM_(data_level, lag)
            # adding each iteration error to a list of errors
            error_MAE.append(MAE(y_train, y_predict))
            error_MAP.append(mean_absolute_percentage_error(y_predict,y_train))

    if decomposition==True:
        decomposed=seasonal_decompose(data_level.values, model='additive',period=1)
        decomposed=[decomposed.resid,decomposed.seasonal,decomposed.trend]

        for lag in lag_range:
            print(lag)
            
            y_train_ = [] #pd.DataFrame()
            y_predict_ = []# pd.DataFrame()
            for composition in decomposed:
                print('dealing with decompositioin')
                dff=pd.DataFrame(composition)
                y_train_composition, y_predict_composition = LGBM_(composition, lag)
#                 print(np.shape(y_train_composition))
#                 print(y_train_composition.values)
                y_train_.append(y_train_composition.values) # y_train_ is a list of list, each element is y_train_composition
                y_predict_.append(y_predict_composition)
#             print([np.shape(element) for element in y_train_])

            # adding all the predictions from each composition
            y_train = np.sum(y_train_, axis=0) 
            y_predict = np.sum(y_predict_, axis=0)
            
            # check point on the shapes
            print(f'y_predict shape: {y_predict.shape}')
            print(f'y_train shape: {y_train.shape}')
            print(f'df shape: {df.shape}')
            end = time.time()
            ts = end - start
            print('Time spend: %f' % (ts))
            
            # adding each iteration error to a list of errors
            error_MAE.append(MAE(y_train, y_predict))
            error_MAP.append(mean_absolute_percentage_error(y_predict,y_train))
            error_RMSE.append(LA.norm(y_predict-y_train,2))
            error_RMSSE.append(RMSSE(y_predict,y_train,df.values))

        #below is to feed the RMSEE function fix this later
#         Target_data = data_level.iloc[[node]].T.reset_index(drop=True)
        


    return error_MAP,error_RMSE,error_RMSSE 


# 1862, 1862, 1913

# In[110]:


error_MAP, error_RMSE, error_RMSSE  = model_LGBM_vizualize(level=2,node=0,exo=True,normalized=False,n_in=5,n_out=1, lgb_params= lgb_params, decomposition=True)


# In[25]:


error_MAP, error_RMSE, error_RMSSE  = model_LGBM_vizualize(level=6,node=0,exo=True,normalized=False,n_in=40,n_out=1, lgb_params= lgb_params, decomposition=True)


# In[26]:


# print(f'error_MAE min {min(error_MAE)}\n  and max {max(error_MAE)}\n (max-min)/max:  {(max(error_MAE)-min(error_MAE))/max(error_MAE)*100:.2f}%')
print(f'error_MAP min {min(error_MAP)}\n  and max {max(error_MAP)}\n (max-min)/max:  {(max(error_MAP)-min(error_MAP))/max(error_MAP)*100:.2f}%')
print(f'error_RMSE min {min(error_RMSE)}\n  and max {max(error_RMSE)}\n (max-min)/max:  {(max(error_RMSE)-min(error_RMSE))/max(error_RMSE)*100:.2f}%')
print(f'error_RMSSE min {min(error_RMSSE)}\n  and max {max(error_RMSSE)}\n (max-min)/max:  {(max(error_RMSSE)-min(error_RMSSE))/max(error_RMSSE)*100:.2f}%')


# In[27]:


lag_range = range(1,40)

plt.subplot(311)
plt.plot(lag_range,np.array(error_MAP)*1000, 'r-')
plt.title('MAP', fontsize=20)

plt.xticks(fontsize=20); plt.yticks(fontsize=20)

plt.subplot(312)
plt.plot(lag_range,np.array(error_RMSE)*1000, 'b-')
plt.title('error_RMSE Error', fontsize=20)
plt.xticks(fontsize=20); plt.yticks(fontsize=20)
plt.xlabel('n_in', fontsize=20)

plt.subplot(313)
plt.plot(lag_range,np.array(error_RMSSE)*1000, 'b-')
plt.title('error_RMSSE Error', fontsize=20)
plt.xticks(fontsize=20); plt.yticks(fontsize=20)
plt.xlabel('n_in', fontsize=20)

plt.tight_layout()


# # Saving CSV for submission

# # HF: running the selected nodes and saving into DataFrame

# In[101]:


## for purpose of review the LGBM hyperparameters is copied from the section "LGBM Model Hyper Parameters setting"
# lgb_params = {
#                     'boosting_type': 'gbdt',
#                     'objective': 'tweedie',
#                     'tweedie_variance_power': 1.1,
#                     'metric': 'rmse',
#                     'subsample': 0.5,
#                     'subsample_freq': 1,
#                     'learning_rate': 0.03,
#                     'num_leaves': 2**11-1,
#                     'min_data_in_leaf': 2**12-1,
#                     'feature_fraction': 0.5,
#                     'max_bin': 100,
#                     'n_estimators': 1400,
#                     'boost_from_average': False,
#                     'verbose': -1,
#                 } 

def model_LGBM(level=1,exo=False,node=0,normalized=False,n_in=3,n_out=1, lgb_params= lgb_params, decomposition=False):

#     error_MAE =[]
#     error_MAP =[]
#     error_RMSE = []
#     error_RMSSE = []
    # make the level data frame 
    levels = lc.LevelsCreater()
    data_level = levels.get_level(sale,level=level);

#     lag_range = range(1,n_in)
    
    def LGBM_(data, lag, compose=-1):
            '''compose -1 is for model decomposition=False and compose 0,1,2 correspond to residual, seasonal, and trend
            compose value is used to distinguish between saved files.
            the estimator weights after training is pickle dumped (saved)
            '''
            
            X_train, X_validate, y_train, y_validat=             train_validate_splitter_l3(data_level, node=node, n_in=lag,n_out=n_out, exo=exo, dropnan=False)

            train_data, valid_data = lgb_train_validate_set_builder(
                                         X_train, X_validate, y_train, y_validat
                                          )

            # Launch seeder again to make lgb training 100% deterministic
            # with each "code line" np.random "evolves" 
            # so we need (may want) to "reset" it
            seed_everything(SEED)
            estimator = lgb.train(lgb_params,
                                  train_data,
                                  valid_sets = [valid_data],
                                  verbose_eval = 1
                                  )

            # Save model - it's not real '.bin' but a pickle file
            # estimator = lgb.Booster(model_file='model.txt')
            # can only predict with the best iteration (or the saving iteration)
            # pickle.dump gives us more flexibility
            # like estimator.predict(TEST, num_iteration=100)
            # num_iteration - number of iteration want to predict with, 
            # NULL or <= 0 means use best iteration

            model_name = 'lgb_model_'+'level'+str(level)+'_node'+str(node)+'_n-in'+str(lag)+'_decmpose'+str(compose)+'_v'+str(VER)+'.bin' #just saves last trained model
            # pickling the trained models
            if not os.path.exists(Absolute+Cache):
                os.mkdir(Absolute+Cache)
            pickle.dump(estimator, open(Absolute+Cache+model_name, 'wb'))

            # in-sample prediction
            y_predict = estimator.predict(X_train)
#             print(MAE(y_train, y_predict))
            
            return y_train, y_predict
        
    # preparign for RMSE y argument
#     df = levels.get_level(sale,level=level)
#     df = df.transpose()
#     df = pd.DataFrame(df[df.columns[node]])
        
    if decomposition==False:
        lag = n_in
        if level == 1:
            data_level.index=['zero']
#         for lag in lag_range:
#             print(lag)
        y_train, y_predict = LGBM_(data_level, lag ,compose=-1) 
            # adding each iteration error to a list of errors
#             error_MAE.append(MAE(y_train, y_predict))
#             error_MAP.append(mean_absolute_percentage_error(y_predict,y_train))

        # check point on the shapes
        print(f'y_predict shape: {y_predict.shape}')
        print(f'y_train shape: {y_train.shape}')
#         print(f'df shape: {df.shape}')
#         RMSSE_value = RMSSE(y_predict,y_train,df.values)
#         print(f'RMSEE leve_{level} Node_{node}: {RMSSE_value}')


    if decomposition==True:
        if level == 1:
            data_level.index=['zero']
#         print(data_level.head())
        decomposed=seasonal_decompose(data_level.values, model='additive',period=1)
        decomposed=[decomposed.resid,decomposed.seasonal,decomposed.trend]

#         for lag in lag_range:
#             print(lag)
        lag = n_in    
        y_train_ = [] #pd.DataFrame()
        y_predict_ = []# pd.DataFrame()
        for i, composition in enumerate(decomposed):
            print('dealing with decompositioin')
            dff=pd.DataFrame(composition)
            y_train_composition, y_predict_composition = LGBM_(composition, lag, compose=i)
#                 print(np.shape(y_train_composition))
#                 print(y_train_composition.values)
            y_train_.append(y_train_composition.values) # y_train_ is a list of list, each element is y_train_composition
            y_predict_.append(y_predict_composition)
#             print([np.shape(element) for element in y_train_])

        # adding all the predictions from each composition
        y_train = np.sum(y_train_, axis=0) 
        y_predict = np.sum(y_predict_, axis=0)

        # check point on the shapes
        print(f'y_predict shape: {y_predict.shape}')
        print(f'y_train shape: {y_train.shape}')
#         print(f'df shape: {df.shape}')
#         RMSSE_value = RMSSE(y_predict,y_train,df.values)
#         print(f'RMSEE leve_{level} Node_{node}: {RMSSE_value}')

#         # adding each iteration error to a list of errors
#         error_MAE.append(MAE(y_train, y_predict))
#         error_MAP.append(mean_absolute_percentage_error(y_predict,y_train))
#         error_RMSE.append(LA.norm(y_predict-y_train,2))
#         error_RMSSE.append(RMSSE(y_predict,y_train,df.values))

    #below is to feed the RMSEE function fix this later
#         Target_data = data_level.iloc[[node]].T.reset_index(drop=True)
        

    return #RMSSE_value #error_MAP,error_RMSE,error_RMSSE 


# In[ ]:


#testing
level=6; node=0
rmsse_lgbm = model_LGBM(level=level,node=node,exo=True,normalized=False,n_in=40,n_out=1, 
                   lgb_params= lgb_params, decomposition=True)


# In[ ]:


data_ = lc.LevelsCreater().get_level(sale,level=1); data_.index=['zero']; data_.head()


# In[103]:


from functions import get_max_node
import  numpy as np
import pandas as pd
import time

TCN=False
if TCN == False:
    s_atuto = pd.DataFrame(index=['LGBM'])
    case = 1
    test = False
    if test == True:
        level_range = [12]
    else:
        level_range = np.arange(1, 13)

    for level in level_range:

        if level == 10:
            node_size=set(pd.read_csv('results/selected_ts_lv10.csv',index_col=0).values.ravel())
            # node_size = np.random.choice(list(range(get_max_node(level))), 100)
            print('running')
        elif level==11:
            node_size = set(pd.read_csv('results/selected_ts_lv11.csv', index_col=0).values.ravel())
        elif level==12:
            node_size = set(pd.read_csv('results/selected_ts_lv12.csv', index_col=0).values.ravel())
        else:
            node_size = list(range(get_max_node(level)))
        
        print(f'.....................................node size length={len(node_size)}....................................')
        for node in node_size:
            start = time.time()
            print(f'level={level},node={node}')
            print('case:', case)
            
            rmsse_lgbm = model_LGBM(level=level,node=node,exo=True,normalized=False,n_in=20,n_out=1,lgb_params= lgb_params, decomposition=False)
#             _, _, rmsse_nbeat = s_nbeat(level=level, node=node, normalized=True, decomposition=True, exo=True, n_in=30,
#                                         n_out=1)
            # _, _, rmsse_ntcn = s_tcn(level=level, node=node, normalized=True, decomposition=True, exo=True, n_in=10,
            #                          n_out=1)
            s_atuto[str(level) + '_' + str(node)] = [rmsse_lgbm]
            if not os.path.exists('results/Ali'):
                os.mkdir('results/Ali')
            s_atuto.to_csv('results/Ali/LGBM_auto.csv')
            case += 1
            end = time.time()
            ts = end - start
            print('Time spend: %f' % (ts))




# s_atuto=pd.DataFrame(index=['LGBM'])
# case=1
# for level in np.arange(1,13):

#     if level==10 or level==11 or level==12:
#         node_size=np.random.choice(range(get_max_node(level)),100)
#         print('running')
#     else:
#         node_size=list(range(get_max_node(level)))
#     print(f'.....................................node size length={len(node_size)}....................................')
#     for node in node_size:
#         start=time.time()
#         print(f'level={level},node={node}')
#         print('case:',case)
#         rmsse_lgbm = model_LGBM(level=level,node=node,exo=True,normalized=False,n_in=30,n_out=1,lgb_params= lgb_params, decomposition=True)
# #         rmsse_nbeat=s_nbeat(level=level,node=node,normalized=True,decomposition=True,exo=True,n_in=30,n_out=1)

#         s_atuto[str(level)+'_'+str(node)]=[rmsse_lgbm]
#         s_atuto.to_csv('Data/s_auto_LGBM.csv')
#         case+=1
#         end=time.time   ()
#         ts=end-start
#         print('Time spend: %f' %(ts))


# In[ ]:


# ########################### Export
# #################################################################################
# # Reading competition sample submission and
# # merging our predictions
# # As we have predictions only for "_validation" data
# # we need to do fillna() for "_evaluation" items
# submission = pd.read_csv(ORIGINAL+'sample_submission.csv')[['id']]
# submission = submission.merge(all_preds, on=['id'], how='left').fillna(0)
# submission.to_csv('submission_v'+str(VER)+'.csv', index=False)


# In[ ]:


RMSSE_LGBM = pd.read_csv('results/Ali/LGBM_auto.csv'); RMSSE_LGBM


# # Predicting from saved model

# In[104]:


from glob import glob


# In[105]:


# putting the pathes to all models without decomposition ('decompose-1' means decomposition=False)
model_list = glob('./Cache/*decmpose-1_v1.bin')
model_list[0]
# len(model_list)


# In[111]:


# testing prediction method of LGBM on loaded saved model
# creating some data
level = 12
node =21537
n_in = 20
n_out = 1
exo = True
dropnan=True
levels = lc.LevelsCreater()
data_level = levels.get_level(sale,level=level);

X_train, X_validate, y_train, y_validat= train_validate_splitter_l3(data_level, node=node, n_in=n_in, n_out=n_out, exo=exo, dropnan=dropnan)

# if you give data set (or the below train_data which is LGMB dataset format) the predict method will give error
# one need to give the predict a raw data format
train_data, valid_data = lgb_train_validate_set_builder(
                             X_train, X_validate, y_train, y_validat
                              )

X_test = series_to_supervised_l3(data_level, node=node, n_in=n_in, n_out=n_out, exo=exo, dropnan=dropnan)


# In[114]:


# load the saved models.
with open(model_list[0], 'rb') as f:
    estimator = pickle.load(f)
    print(f'X_validate shape:\n{np.array(X_validate).shape}')
    print(f'X_test shape:\n{X_test.shape}')
    print(f'X_train shape:\n{X_train.shape}')
    print(f'predicted X_test format:\n{np.array(estimator.predict(X_test)).shape}')
    


# In[ ]:




