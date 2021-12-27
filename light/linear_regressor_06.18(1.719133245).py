import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

train = pd.read_csv('train.csv', index_col = 0)
test = pd.read_csv('test.csv', index_col = 0)

train = train.interpolate('linear', axis = 1)
train = train.fillna(method = 'ffill', axis = 1)
test = test.interpolate('linear', axis = 1)
test.fillna(method = 'ffill', axis = 1)

train.loc[:,sum(pd.isna(train)) >0]

X = train.iloc[:,:71]
y = train.iloc[:,71:]

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 0)

lr = LinearRegression()
lr.fit(x_train, y_train)
lr_pred = lr.predict(x_test)
print(r2_score(y_test, lr_pred))  # 0.002937053919089855
                                  # 0.002937053919089855    
print(lr.score(x_test, y_test))   # 0.002579377930869767 
mae(lr_pred, y_test) # 1.7191332457637285  interploate axis = 1

#######

train = pd.read_csv('train.csv', index_col = 0)
test = pd.read_csv('test.csv', index_col = 0)

train = train.interpolate('linear', axis = 1)
train = train.fillna(method = 'bfill', axis = 1)
test = test.interpolate('linear', axis = 1)
test.fillna(method = 'bfill', axis = 1)

train.loc[:,sum(pd.isna(train)) >0]

X = train.iloc[:,:71]
y = train.iloc[:,71:]

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 0)

lr = LinearRegression()
lr.fit(x_train, y_train)
lr_pred = lr.predict(x_test)

print(r2_score(y_test, lr_pred))

mae( y_test, lr_pred)


###
train = pd.read_csv('train.csv', index_col = 0)
test = pd.read_csv('test.csv', index_col = 0)

train = train.interpolate('polynomial', axis = 1)
train = train.fillna(method = 'ffill', axis = 1)
test = test.interpolate('polynomial', axis = 1)
test.fillna(method = 'ffill', axis = 1)


X = train.iloc[:,:71]
y = train.iloc[:,71:]

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 0)

lr = LinearRegression()
lr.fit(x_train, y_train)
lr_pred = lr.predict(x_test)
print(r2_score(y_test, lr_pred))  # 0.002937053919089855
                                  # 0.002937053919089855    
mae(lr_pred, y_test) # 1.7191332457637285  interploate axis = 1


