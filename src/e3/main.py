import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn import *
from sklearn.cluster import KMeans
from sklearn.datasets import make_regression
from sklearn.ensemble import BaggingClassifier, RandomForestRegressor, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, minmax_scale

RANDOM_STATE = 11
print("###################Task 1###################")
print("Part a")
data_01 = sklearn.datasets.load_breast_cancer(return_X_y=False, as_frame=True)
x1_train, x1_test, y1_train, y1_test = sklearn.model_selection.train_test_split(data_01.data, data_01.target,
                                                                                train_size=0.80,
                                                                                random_state=RANDOM_STATE)

print("Part b")
kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE)
kmeans = kmeans.fit(x1_train)
# print("K-means labels:\n", kmeans.labels_)
kmeans_pred = kmeans.predict(x1_test)
# print("K-means prediction:\n", kmeans_pred)

ari = sklearn.metrics.adjusted_rand_score(y1_test, kmeans_pred)
print("ARI score: ", ari)
# The ARI score is round about 0.522, and it computes a similarity measure between two clusterings by considering all
# pairs of samples and counting pairs that are assigned in the same or different clusters in the predicted and true
# clusterings.

print("Part c")
kmeans_bagging = BaggingClassifier(base_estimator=KMeans(n_clusters=2, random_state=RANDOM_STATE), n_estimators=20,
                                   max_samples=0.3, max_features=1.0, bootstrap=True,
                                   bootstrap_features=True, random_state=RANDOM_STATE)
kmeans_bagging = kmeans_bagging.fit(x1_train, y1_train)
kmeans_bagging_pred = kmeans_bagging.predict(x1_test)
# print(kmeans_bagging_pred)

ari_bag = sklearn.metrics.adjusted_rand_score(y1_test, kmeans_bagging_pred)
print("ARI score bagging: ", ari_bag)

print("Part d")
# The performance is the same as without bagging. I think bagging does more sence by using another algorithm than
# kmeans for example DecisionTree


print("###################Task 2###################")
print("Part a")
data_02, target_02 = make_regression(n_samples=1000, n_features=20, n_informative=10, noise=0.2,
                                     random_state=RANDOM_STATE)

print("Part b")
pipeline = Pipeline([('scaler', StandardScaler()), (
'randomForestRegressor', RandomForestRegressor(n_estimators=50, max_leaf_nodes=10, random_state=RANDOM_STATE))])

# A random forst is a meta estimator that fits a number of classifying decision trees on various sub-samples
# of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

print("Part c")
x2_train, x2_test, y2_train, y2_test = sklearn.model_selection.train_test_split(data_02, target_02, test_size=0.20,
                                                                                random_state=RANDOM_STATE)

pipeline = pipeline.fit(x2_train, y2_train)
pipeline_pred = pipeline.predict(x2_test)
print(pipeline_pred)

mae = mean_absolute_error(y2_test, pipeline_pred, sample_weight=None, multioutput='uniform_average')
print("Mean Absolute Error: ", mae)

print("Part d")

#TODO:

print("###################Task 3###################")

print("Part a")
data_03 = sklearn.datasets.load_breast_cancer(return_X_y=False, as_frame=True)

# Scale data with MinMaxScaler
data_03_scale = minmax_scale(data_03.data)
data_03.data = data_03_scale

x3_train, x3_test, y3_train, y3_test = sklearn.model_selection.train_test_split(data_03.data, data_03.target, train_size=0.80, random_state=RANDOM_STATE)
#print(x3_train, x3_test, y3_train, y3_test)

print("Part b")
log_reg = LogisticRegression(random_state=RANDOM_STATE)
log_reg = log_reg.fit(x3_train, y3_train)
log_reg_pred = log_reg.predict(x3_test)

accuracy_log_reg = accuracy_score(y3_test, log_reg_pred)
print("Logistic regression accuracy in percent: ", accuracy_log_reg * 100)


print("Part c")
adaboost = AdaBoostClassifier(base_estimator=LogisticRegression(random_state=RANDOM_STATE), random_state=RANDOM_STATE, n_estimators=20)
adaboost = adaboost.fit(x3_train, y3_train)


print("Part d")
adaboost_pred = adaboost.predict(x3_test)

accuracy_ada = accuracy_score(y3_test, adaboost_pred)
print("AdaBoost accuracy in percent: ", accuracy_ada * 100)

# The boosted approach has a lower accuracy than the adaboost classifier 94.74% < 96.49%

# The AdaBoostClassifier begins by fitting a classifier on the original dataset and then fits additional copies of the
# classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that
# subsequent classifiers focus more on difficult cases.













