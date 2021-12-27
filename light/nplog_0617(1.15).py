import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
import optuna
from sklearn.model_selection import KFold,cross_val_score,train_test_split,cross_val_predict
from sklearn.metrics import mean_absolute_error

from lightgbm import LGBMRegressor,LGBMClassifier 

from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
import warnings
import time
from sklearn.metrics import f1_score, roc_auc_score, classification_report

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

train = pd.read_csv('train.csv',index_col=0)
test = pd.read_csv('test.csv',index_col=0)
submission = pd.read_csv('sample_submission.csv',index_col=0)


train_dst = train.filter(regex='_dst$',axis=1)
test_dst = test.filter(regex='_dst$',axis=1)

train_src = train.filter(regex='_src$',axis=1)
test_src = test.filter(regex='_src$',axis=1)

train_rho = train['rho']
test_rho = test['rho']

dst_list =  train.filter(regex='_dst$',axis=1).columns
src_list =  train.filter(regex='_src$',axis=1).columns

train_dst = train_dst.interpolate(methods='linear',axis = 1)
test_dst = test_dst.interpolate(methods='linear',axis = 1)

for col in dst_list:
    train_dst[col] = train_dst[col] * ((train['rho']^2)*100)
    test_dst[col] = test_dst[col] * ((test['rho']^2)*100)


train_x = pd.DataFrame(np.log10(np.array(train_src)/np.array(train_dst)))
test_x = pd.DataFrame(np.log10(np.array(test_dst)/np.array(test_src)))

tr_ds = pd.DataFrame(np.log10(np.array(train_dst)))
tr_sr = pd.DataFrame(np.log10(np.array(train_src)))

test_x.index = test_rho.index


train_x = pd.concat([train_rho,tr_ds,train_x],axis=1)
test_x = pd.concat([test_rho,test_dst,test_x],axis=1)

train_x = train_x.replace(np.inf, 0)
train_x = train_x.replace(-np.inf, 0)
train_x = train_x.replace(np.NaN, 1)

test_x = test_x.replace(np.inf, 0)
test_x = test_x.replace(-np.inf, 0)
test_x = test_x.replace(np.NaN, 1)

train_y = train[list(submission)]


 params = {'learning-rate' : 0.01,
           'max_depth' : 16,
           'boosting_type' : 'gbdt',
           'objective' : 'regression',
           'metric' : 'mae',
           'is_training_metric' : True,
           'nem_leaves' : 144,
           'feature_fracton' : 0.9,
           'bagginf_fraction' : 0.7,
           'bagginf_freq' : 5,
           'seed' : 2020}

from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import cross_val_score

model = MultiOutputRegressor(lgb.LGBMRegressor(**params, random_state = 0), n_jobs = -1)
model.fit(train_x, train_y)
light_pre = model.predict(test_x)
# mae(light_pre, test_y)   # 1.19866

base_model=lgb.LGBMRegressor(**params,random_state=0)

multi_model=MultiOutputRegressor(base_model, n_jobs = -1)

def model_scoring_cv(model, x, y, cv=10):
    start=time.time()
    score=cross_val_score(model, x, y, cv=cv, scoring='neg_mean_absolute_error').mean()
    stop=time.time()
    print(f"Validation Time : {round(stop-start, 3)} sec")
    return score

model_scoring_cv(multi_model,train_x,train_y) 



## [ lightgbm parameter : reg_alpha, reg_lambda]
# L1 : 가중치의 절대값에 비례하는 비용이 추가

# L2규제: 가중치의 제곱에 비례하는 비용이 추가(weight decay)

# reg_alpha
# - L1의 weight
# - 숫자가 클수록 보수적

# reg_lambda
#  - L2의 weight
#  - 숫자가 클수록 보수적

# reg_lambda : L2 regularization
# reg_alpha : L1 regularization
