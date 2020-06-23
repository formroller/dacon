#  multioutput basic model tutorial
# https://machinelearningmastery.com/multi-output-regression-models-with-python/

# lightgbm parameters
# https://nurilee.com/lightgbm-definition-parameter-tuning/

# xgbsoot tutorial
# https://www.datacamp.com/community/tutorials/xgboost-in-python

# RandomForest parameter description
# https://woolulu.tistory.com/28

# 측정지표
# chukycheese.github.io/translation/statistics/absolute-error-and-mean-absolute-error/

# 국내 논문
# http://www.riss.or.kr/index.do

# [참고문헌]
# A time-domain NIR brain imager applied in functional stimulation experiments
# Non-invasive NIR spectroscopy of human brain function during exercise.
# Optical windows for head tissues in near‐infrared and short‐wave infrared regions: Approaching transcranial light applications.

# [R] 데이터 종류에 따른 분석 기법
# https://rpubs.com/jmhome/datatype_analysis

# 모델 분할
from sklearn.model_selection import train_test_split

# 데이터 군집
from sklearn.cluster import KMeans

# 모델
import sklearn
from sklearn.linear_model import LinearRegression as lr
import lightgbm as lgb  # ** lgb.LGBMregressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
from skelanr.ensenble import RandomForest

# 측정값
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score  # 결정계수

# 상태 bar
from tqdm import tqdm

# 시각화
# https://zzsza.github.io/development/2018/08/24/data-visualization-in-python/

###### 
# 변수 중요도 계산 in python
https://machinelearningmastery.com/calculate-feature-importance-with-python/
https://wikidocs.net/16882

# feature selection
https://subinium.github.io/feature-selection/
