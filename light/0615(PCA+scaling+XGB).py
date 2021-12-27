import os
import pandas as pd
import numpy as np

import sklearn 
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor 

from sklearn.preprocessing import StandardScaler

import time
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error as mae

train = pd.read_csv('train.csv', index_col = 0)
test = pd.read_csv('test.csv', index_col = 0 )
sub = pd.read_csv('sample_submission.csv', index_col = 0)

# 데이터 보간 
train = train.interpolate(method = 'linear', axis = 1)
test = test.interpolate(method = 'linear', axis = 1)

# 데이터 분리
X = train.iloc[:,:71]
y = train.iloc[:,71:]

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 0)



# scale
m_ss = StandardScaler()
m_ss.fit(train_x)
train_x_sc = m_ss.transform(train_x)
test_x_sc = m_ss.transform(test_x)

# lightgbm params 설정
lgb.LGBMRegressor(params)  # parameter값 확인
 params = {'learning-rate' : 0.01,
           'max_depth' : -1,
           'boosting_type' : 'gbdt',
           'objective' : 'regression',
           'metric' : 'mae',
           'is_training_metric' : True,
           'nem_leaves' : 144,
           'feature_fracton' : 0.9,
           'bagginf_fraction' : 0.7,
           'bagginf_freq' : 5,
           'seed' : 2020}

model = MultiOutputRegressor(lgb.LGBMRegressor(**params, random_state = 0), n_jobs = -1)
model.fit(train_x_sc, train_y)
light_pre = model.predict(test_x_sc)
mae(light_pre, test_y)   # 1.2139664910417782


##=> 스케일 시 기존 mae값(1.19)보다 높은 값(1.21) 출력

###score
base_model=lgb.LGBMRegressor(**params,random_state=0)

multi_model=MultiOutputRegressor(base_model, n_jobs = -1)



def model_scoring_cv(model, x, y, cv=10):
    start=time.time()
    score=cross_val_score(model, x, y, cv=cv, scoring='neg_mean_absolute_error').mean()
    stop=time.time()
    print(f"Validation Time : {round(stop-start, 3)} sec")
    return score


model_scoring_cv(multi_model,train_x_sc,train_y)  #- 1.21939164540293


###### PCA
from sklearn.decomposition import PCA

m_pca = PCA()
m_pca.fit(train_x)

train_x_pca = m_pca.transform(train_x)
test_x_pca = m_pca.transform(test_x)

model = MultiOutputRegressor(lgb.LGBMRegressor(**params, random_state = 0), n_jobs = -1)
model.fit(train_x_pca, train_y)
light_pre = model.predict(test_x_pca)
mae(light_pre, test_y)   # 1.5838890402706525  # 기존 mae값 보다 저하

model.feature_importances_



#####
import xgboost as xgb


lgb.LGBMRegressor(params)  # parameter값 확인
params1 = {'learning-rate' : 0.01,
           'max_depth' : 20,
           'boosting_type' : 'gbdt',
           'objective' : 'reg:linear',
           'metric' : 'mae',
           'is_training_metric' : True,
           'nem_leaves' : 144,
           'feature_fracton' : 0.9,
           'bagginf_fraction' : 0.7,
           'bagginf_freq' : 5,
           'seed' : 2020}

model = MultiOutputRegressor(xgb.XGBRFRegressor(**params1, random_state=0), n_jobs = -1)
model.fit(train_x, train_y)
preds = model.predict(test_x)
mae(preds, test_y)  # 1.211640226204764

