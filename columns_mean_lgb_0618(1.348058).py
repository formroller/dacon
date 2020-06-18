# 가설
##20nm 단위로 그룹
 # - 배음대와 중첩대를 갖는 완만한 곡선의 형태로 나타난다

import pandas as pd
import numpy as np
import os 
 os.chdir('.\dacon')

train = pd.read_csv('train.csv', index_col = ['id','rho'])
test = pd.read_csv('train.csv', index_col = ['id','rho'])

train = train.drop('id', axis = 1)
pd.read_csv??

#interpolate
train = train.interpolate('linear',axis = 1)
test = test.interpolate('linear', axis = 1)
## 
import matplotlib.pyplot as plt
# plt.plot((train['650_dst'].groupby(train['hhb']).mean()))
# plt.plot((train['660_dst'].groupby(train['hhb']).mean()))

train.stack()

dst = [k for k in train.columns if 'dst' in k]
src = [k for k in train.columns if 'src' in k]

train_dst = train[dst]
train_src = train[src]
train = train.sort_values(by = 'rho')

## plt.plot dst
plt.plot(train.groupby('rho')[dst].mean())
train.groupby('rho')[dst].mean().idxmax()
train.groupby('rho')['ca'].mean()
train.groupby('rho')['na'].mean()

fig, axes = plt.subplots(2, 1)
axes[0].plot(train.pivot_table(index = 'rho', values = dst, aggfunc = mean))
axes[0].legend(dst)
axes[1].plot(train.pivot_table(index = 'rho', values = src, aggfunc = mean))
axes[1].legend(src)


# (train.iloc[:,i] + train.iloc[:,i+1])/2

for i in arange(0,len(train_dst.columns)+1):
    np.array(train.iloc[:,i] + train.iloc[:,i+1])/2

##### 변수 가공(a1+b1+c)
#dst
a = train_dst.iloc[:,range(0,35,3)]
b = train_dst.iloc[:,range(1,35,3)]
c = train_dst.iloc[:,range(2,35,3)]
a.shape
b.shape
c.shape
a1 = a.iloc[:,:11]
b1 = b.iloc[:,:11]
a1.shape

dt_dst = pd.DataFrame((np.array(a1) + np.array(b1)  +np.array(c))/3)
#src
s1 = train_src.iloc[:,range(0,35,3)]
s2 = train_src.iloc[:,range(1,35,3)]
s3 = train_src.iloc[:,range(2,35,3)]
s1.shape
s2.shape
s3.shape
a_s1 = s1.iloc[:,:11]
b_s2 =s2.iloc[:,:11]
a1.shape

dt_src = pd.DataFrame((np.array(a_s1) + np.array(b_s2)  +np.array(s3))/3)


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = np.log(dt_src/dt_dst)
y = train.iloc[:,71:]

X = pd.DataFrame(np.where(X == inf,1,np.where(X== -inf,-1,np.where(pd.isna(X),0,X))))

train_x, test_x, train_y, test_y = train_test_split(X, y, random_state = 0)
## scaling
m_ss = StandardScaler()
m_ss.fit(train_x)

train_x_sc = m_ss.transform(train_x)
test_x_sc = m_ss.transform(test_x)

X.info()
sum(pd.isna(X))



## lightgbm modeling
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error as mae
from sklearn.multioutput import MultiOutputRegressor
lgb.LGBMRegressor(params)  # parameter값 확인
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

model.fit(train_x_sc, train_y)

light_pre = model.predict(test_x_sc)
mae(light_pre, test_y)   # 1.19866



### for(lgm params)

rate = [0.05,0.025, 0.01, 0.1, 0.3, 0.5]
range(1,30,1)
depth = np.arange(-1,20, dtype = 'int')


col_rate =[]; col_depth = []; col_mae = []
for i in rate:
    for j in range(-5,20):
         params = {'learning-rate' :rate,
           'max_depth' : j,
           'boosting_type' : 'gbdt',
           'objective' : 'regression',
           'metric' : 'mae',
           'is_training_metric' : True,
           'nem_leaves' : 144,
           'feature_fracton' : rate,
           'bagginf_fraction' : rate,
           'bagginf_freq' : depth,
           'seed' : 2020}
         
         col_rate.append(i)
         col_depth.append(j)
         
         model = MultiOutputRegressor(lgb.LGBMRegressor(**params, random_state = 0), n_jobs = -1)
         model.fit(train_x_sc, train_y)
         light_pre = model.predict(test_x_sc)
         score = mae(light_pre, test_y)
         col_mae.append(score)
         mae_score = pd.DataFrame({'col_rate':col_rate,
                          'col_depth' : col_depth,
                          'col_mae' : col_mae})
         print(mae_score.sort_values(by = 'col_mae', ascending = True))

##  => 1.348058



################

