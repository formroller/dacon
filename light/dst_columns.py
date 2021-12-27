import os
os. getcwd()
os.chdir('.\dacon')

import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sub = pd.read_csv('sample_submission.csv')

##### interpolate
train = train.interpolate(method = 'linear', axis =1)
test = test.interpolate(method = 'linear', axis = 1)


# dst columns
import sklearn
from sklearn.model_selection import train_test_split

dst= train.loc[:,'650_dst':]

X = dst.loc[:,'650_dst':'990_dst']
y = dst.loc[:,'hhb':]
train_x, test_x, train_y, test_y = train_test_split(X, y, random_state = 0)

#### model
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor

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


model = MultiOutputRegressor(lgb.LGBMRegressor(**params, random_state = 0), n_jobs = -1)
model.fit(train_x, train_y)
light_pre = model.predict(test_x)

#### mae
from sklearn.metrics import mean_absolute_error as mae
mae(light_pre, test_y)   # 1.3971867430339497


# ==> mae(평균 절대 오차값) 증가, 
# dst columns만 사용 -> 예측력 감소
