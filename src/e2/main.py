import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn import *
from sklearn.datasets.tests.data import openml
from sklearn.svm import *
from sklearn.preprocessing import *
from sklearn.pipeline import Pipeline, make_pipeline

print("############################Task 1############################")
data_01 = sklearn.datasets.fetch_openml("banknote-authentication", return_X_y=False, as_frame=True)
print(data_01.frame)

print("############################Task 2############################")
print("Part a")
data_02 = sklearn.datasets.load_breast_cancer(return_X_y=False, as_frame=True)
# print(data_02.frame)

x2_train, x2_test, y2_train, y2_test = sklearn.model_selection.train_test_split(data_02.data, data_02.target,
                                                                                test_size=0.20, random_state=11)
print(x2_train, x2_test, y2_train, y2_test)

print("Part b")
# Pipeline 1
pipeline01 = make_pipeline(StandardScaler(), LinearSVC(C=1.0, max_iter=1000, loss='hinge', random_state=11))
pipeline01.fit(x2_train, y2_train)
score01 = pipeline01.score(x2_test, y2_test)
print("Pipeline 1 output: ", score01)

# Pipeline 2
pipeline02 = make_pipeline(StandardScaler(), SVC(C=1.0, max_iter=1000, random_state=11, degree=3, kernel='poly'))
pipeline02.fit(x2_train, y2_train)
score02 = pipeline02.score(x2_test, y2_test)
print("Pipeline 2 output: ", score02)

# Pipeline 3
pipeline03 = make_pipeline(StandardScaler(), SVC(C=1.0, random_state=11, degree=3, kernel='poly', coef0=1))
pipeline03.fit(x2_train, y2_train)
score03 = pipeline03.score(x2_test, y2_test)
print("Pipeline 3 output: ", score03)

print("Part c")
# Comparison: