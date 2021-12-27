import os

import pandas as pd

train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)
sub = pd.read_csv('sample_submission.csv', index_col = 0)

#
train = train.interpolate('linear', axis = 1)
train = train.fillna(1e-99, axis = 1)

test = test.interpolate('linear', axis = 1)
test.fillna(1e-99, axis = 1)

##
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as mae

X = train.iloc[:,:71]
y = train.iloc[:,71:]

train_x, test_x, train_y, test_y = train_test_split(X, y, random_state = 0)

lr = LinearRegression(normalize=True)
lr.fit(train_x, train_y)
test_set = test


my_pred = lr.predict(test_set)  # test_set 예측

y_pred = lr.predict(test_x)
mae(test_y, y_pred)
lr.import
## 
import matplotlib.pyplot as plt
plt.scatter(test_y, y_pred, alpha = 0.4)
plt.xlabel('Actual cost')
plt.ylabel('Predict cost')
plt.title('Multiple Linear Regression')

## 회귀계수 확인
import numpy as np

plt.scatter(X,y, alpha = 0.4)


## 번외) svm실시
from sklearn import svm
from sklearn.decomposition import PCA

X, y 
train_x, test_x, train_y, test_y = train_test_split(X, y, random_state = 0)
pca = PCA(n_components=4, whiten = True)
pca.fit(train_x)
train_x_pca = pca.transform(train_x)
test_x_pca = pca.transform(test_x)

pca_train = pd.DataFrame(data = train_x_pca, columns = ['prin1', 'prin2'])

sum(pca.explained_variance_ratio_)

fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel('prin1', fontsize = 15)
ax.set_ylabel('prin2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
y.columns
target = y.columns
targets = ['hhb','hbo2','ca','na']
colors = ['red','green','blue','black']

# for target, color in zip(targets,colors):
#     indicesToKeep = finalDf['target'] == target
#     ax.scatter(finalDf.loc[indicesToKeep, 'principal component1']
#                , finalDf.loc[indicesToKeep, 'principal component2']
#                , c = color
#                , s = 50)
# ax.legend(targets)
# ax.grid()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(train_x)
train_x_sc = sc.transform(train_x)
test_x_sc = sc.transform(test_x)

##
fitcecoc(train_x, train_y)
m_svm = fitcecoc(train_x, train_y)
m_svm = svm.SVC(kernel='rbf')
m_svm.fit(train_x_sc, train_y)

from sklearn.svm import LinearSVR
svm_reg = LinearSVR(epsilon = 15,random_state=0)
model = MultiOutputRegressor(svm_reg)
model.fit(train_x, train_y)
pred_y = model.predict(test_x)

mae(test_x, pred_y)
