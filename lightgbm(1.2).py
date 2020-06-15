# https://machinelearningmastery.com/multi-output-regression-models-with-python/
#  multioutput basic model
import os
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split

# scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# 모델 선택
import sklearn
from sklearn.linear_model import LinearRegression as lr
from sklearn.neighbors import KNeighborsRegressor as knn_R
from sklearn.linear_model import LogisticRegression as LR
# 선형회귀모형 적합
import statsmodels.api as sm

# 측정값
from sklearn.metrics import mean_absolute_error as mae

# 시각화
import matplotlib.pyplot as plt
import mglearn
os.getcwd()
os.chdir('.\dacon')
print(sklearn.__version__)

train = pd.read_csv('train.csv', index_col = 'id')
test = pd.read_csv('test.csv')
test2 = pd.read_csv('test.csv', index_col = 'id')
sub = pd.read_csv('sample_submission.csv')
sub2 = pd.read_csv('sample_submission.csv', index_col = 'id', na_values=0)


test_set = pd.merge(test,sub)
test_set = test_set.set_index('id')  # 최종 test set

# 원자료 분리
X = train.iloc[:,:71]
y = train.iloc[:,71:]

X = X.interpolate(method = 'linear', axis = 1)  # 데이터 보간
test2 = test2.interpolate(method = 'linear', axis = 1)

# =============================================================================
# linear regression
# =============================================================================
# case2) 
 # - train -> X(feature) , y(target)분리 
 # - 학습(fit)후 test_set (test2) y값 예측
train_x, test_x, train_y, test_y = train_test_split(X,y, test_size=0.25, random_state=0)
print(len(train_x))
print(len(train_y))

print(len(test_x))
print(len(test_y))

linear= lr()
    linear.fit(train_x, train_y)
LR = LR()


pd.Series(y)

x2 = sm.add_constant(X)
model = sm.OLS(y, x2)
result = model.fit()
print(result.summary())

y_pred = linear.predict(test_x)
print(y_pred)
print(list(test_y))

print('정확도 : ' metrics.accuracy_score(y_test, y_pred))
print('정확도 :', metrics.accuracy_score(y_test, y_pred))



# #####lightgbm
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor as MOR

train_ds = lgb.Dataset(train_x, label = train_y)
test_ds = lgb.Dataset(test_x, label = test_y)


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
 

lgb.LGBMRegressor(params)


light1= lgb.train(params, train_ds, 1000, test_ds, verbose_eval = 100, early_stopping_rounds = 100)
model =  MOR(lgb.train(paramsa, train_ds, 1000, test_ds, verbose_eval = 100, early_stopping_rounds = 100))


model = MOR(lgb.LGBMRegressor(random_state = 0, ), n_jobs = -1)
model.fit(train_x, train_y)
light_pre = model.predict(test_x)
mae(light_pre, test_y)   # 1.2027633417328685




MOR(lgb.train(params, train_ds, 1000, test_ds, verbose_eval = 100, early_stopping_rounds = 10))


MOR(lgb.LGBMRegressor(**params, n_estimators = 1000))
################

params = {'learning-rate' : 0.01,
          'max_depth' : 20,
          'boosting' : 'dart',
          'objective' : 'regression',
          'metric' : 'mae',
          'is_training_metric' : True,
          'nem_leaves' : 144,
          'feature_fracton' : 0.9,
          'bagginf_fraction' : 0.7,
          'bagginf_freq' : 5,
          'seed' : 2020}








