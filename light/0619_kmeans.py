import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

train_src= train.iloc[:,2:37]
sum(pd.isna(train_src))

train_dst = train.iloc[:,37:72]
dst_na = sum(pd.isna(train_dst))
plt.plot(dst_na, rotation = 90)


# dst_na plot(+grid)
plt.plot(dst_na)
plt.xticks(rotation = 90)
plt.grid()

# interploate
train = train.interpolate('linear', axis = 1)
a1= train.loc[:,sum(pd.isna(train)) > 0]
train[a1]

test = test.interpolate('linear', axis = 1)


# kmeans cluster
from sklearn.cluster import KMeans

m_km = KMeans(n_clusters = 3)
label = m_km.fit(train)
y_predict = m_km.fit_predict(train)

train['cluster'] = y_predict
print(train)

np.unique(train.cluster)


#scree plot
KMeans()
from scipy.spatial.distance import cdist

distortions = []
K = np.arange(1,10)
for k in K:
    m_km = KMeans(n_clusters = k).fit(train)
    m_km.fit(train)
    distortions.append(sum(np.min(cdist(train, m_km.cluster_centers_, 'euclidean'), axis=1)) / train.shape[0])

### plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

#####
colormap = { 0: 'red', 1: 'green', 2: 'blue'}
colors = train.apply(lambda row: colormap[row.cluster], axis=1)
ax = train.plot(kind='scatter', x=1, y=2, alpha=0.1, s=300, c=colors)
train.iloc[:,76]
train

## scatterplot

plt.grid()

plt.scatter(train.cluster, np.arange(10000), c=np.arange(10000), label = 'A')
plt.legend(train.cluster)
train.cluster.head(100)
print(sum(train.cluster==0),sum(train.cluster==1),sum(train.cluster==2),sum(train.cluster==3))

## 무의미
