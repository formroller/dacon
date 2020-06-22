import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import nan as NA
from pandas import Series 
from pandas import DataFrame

from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor

from lightgbm import LGBMRegressor, LGBMClassifier
import time

train = pd.read_csv('train.csv',index_col=0)
test = pd.read_csv('test.csv',index_col=0)
submission = pd.read_csv('sample_submission.csv',index_col=0)

train_dst = train.filter(regex='_dst$',axis=1) 
test_dst = test.filter(regex='_dst$',axis=1)

train_src = train.filter(regex='_src$',axis=1)
test_src = test.filter(regex='_src$',axis=1)

train_rho = train['rho']
test_rho = test['rho']


train_y = train.iloc[:,71:75]

train_dst = train_dst.interpolate(method='linear', axis=1) #선형보간 실행
train_dst.fillna(1E-99, inplace=True)                      #na값을 극소값으로 치환 

test_dst = test_dst.interpolate(method='linear', axis=1)
test_dst.fillna(1E-99, inplace=True)

train_x = np.log10(np.array(train_src) / np.array(train_dst))  #log(src/dst)   

train_x = DataFrame(train_x).replace(np.inf, 0)         #inf,-inf,Na값 치환
train_x = train_x.replace(-np.inf, 0)
train_x = train_x.replace(np.NaN, 1)


test_x = np.log10(np.array(test_src) / np.array(test_dst))        

test_x = DataFrame(test_x).replace(np.inf, 0)
test_x = test_x.replace(-np.inf, 0)
test_x = test_x.replace(np.NaN, 1)


test_x.index=test.index
