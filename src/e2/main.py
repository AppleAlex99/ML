import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import numpy as np
from sklearn import *
from sklearn.datasets.tests.data import openml
from sklearn.model_selection import GridSearchCV
from sklearn.svm import *
from sklearn.preprocessing import *
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import PolynomialCountSketch
from sklearn.tree import DecisionTreeRegressor

print("############################Task 1############################")
print("Part a")
data_01 = sklearn.datasets.fetch_openml("banknote-authentication", return_X_y=False, as_frame=True)
print(data_01.frame)

print("Part b")
tsne = sklearn.manifold.TSNE(n_components=2, learning_rate='auto', init='pca', random_state=11).fit_transform(data_01.data)
print(tsne)
tsne_result = pd.DataFrame(tsne, columns=['tsne1', 'tsne2'])

data_01.target = data_01.target.astype(int)
plt.scatter(tsne_result['tsne1'], tsne_result['tsne2'], c=data_01.target)
TODO: plt.show()
# Suitable for linear SVM classification:
# yes, the dataset is is suitable for SVM classification (1372, 2)

print("Part c")
x1_train, x1_test, y1_train, y1_test = sklearn.model_selection.train_test_split(data_01.data, data_01.target,
                                                                                test_size=0.20, random_state=11)
# print(x1_train, x1_test, y1_train, y1_test)

print("Part d")
pipeline_01_01 = Pipeline(
    [('scaler', RobustScaler()), ('linearsvc', LinearSVC(C=1, max_iter=10000, loss='hinge', random_state=11))])
pipeline_01_01.fit(x1_train, y1_train)
# I use the RobustScaler, because it is not getting influenced
# by a few number of very large marginal outliers

print("Part e")
score_01_01 = pipeline_01_01.score(x1_test, y1_test)
print("Score Test Pipeline: ", score_01_01)

score_01_02 = pipeline_01_01.score(x1_train, y1_train)
print("Score Train Pipeline: ", score_01_02)
# The value of Train Split is higher than the test split so the pipeline is overfitting

# Confusion Matrix
cm_test = sklearn.metrics.confusion_matrix(y1_test, pipeline_01_01.predict(x1_test))
print("Confusion Matrix of Test Split:\n", cm_test)
cm_train = sklearn.metrics.confusion_matrix(y1_train, pipeline_01_01.predict(x1_train))
print("Confusion Matrix of Train Split:\n", cm_train)

# # text here

print("############################Task 2############################")
print("Part a")
data_02 = sklearn.datasets.load_breast_cancer(return_X_y=False, as_frame=True)
# print(data_02.frame)

x2_train, x2_test, y2_train, y2_test = sklearn.model_selection.train_test_split(data_02.data, data_02.target,
                                                                                test_size=0.20, random_state=11)
# print(x2_train, x2_test, y2_train, y2_test)

print("Part b")
# Pipeline 1
pipeline_02_01 = Pipeline([('scaler', RobustScaler()),
                           ('linearsvc01', LinearSVC(C=1, max_iter=1000, loss='hinge', random_state=11))])

pipeline_02_01.fit(x2_train, y2_train)

# Pipeline 2
pipeline_02_02 = Pipeline([('scaler', RobustScaler()), ('polySketch', PolynomialCountSketch(degree=3)),
                           ('linearsvc02', LinearSVC(C=1, max_iter=1000, loss='hinge', random_state=11))])

pipeline_02_02.fit(x2_train, y2_train)

# Pipeline 3
pipeline_02_03 = Pipeline([('scaler', RobustScaler()),
                           ('svc', SVC(C=1, kernel='poly', degree=3, coef0=1, random_state=11))])

pipeline_02_03.fit(x2_train, y2_train)

print("Part c")
# Pipeline 1 Score
score_02_01 = pipeline_02_01.score(x2_test, y2_test)
print("Task 2 Pipeline 1 output: ", score_02_01)

# Pipeline 2 Score
score_02_02 = pipeline_02_02.score(x2_test, y2_test)
print("Task 2 Pipeline 2 output: ", score_02_02)

# Pipeline 3 Score
score_02_03 = pipeline_02_03.score(x2_test, y2_test)
print("Task 2 Pipeline 3 output: ", score_02_03)

# Comparison:
# The first and the third pipelines are having the same value,
# so i will improve the 1 pipeline to get a higher value

print("Part d")
# Grid search with Pipeline 1
parameters = {'linearsvc01__max_iter': [100, 1000, 10000], 'linearsvc01__C': [0.1, 1, 10, 100]}
grid_search = GridSearchCV(pipeline_02_01, parameters)
grid_search.fit(data_02.data, data_02.target)

print(grid_search.best_params_)
print(grid_search.best_score_)

# The Score is improved by less than 0.0060

print("############################Task 3############################")
print("Part a")
data_03, target_03 = sklearn.datasets.make_classification(n_samples=10, n_features=5, n_classes=2, random_state=11)
print(data_03)
print(target_03)

print("############################Task 4############################")
print("Part a")
data_04 = sklearn.datasets.fetch_openml("banknote-authentication", return_X_y=False, as_frame=True)

x4_train, x4_test, y4_train, y4_test = sklearn.model_selection.train_test_split(data_04.data, data_04.target,
                                                                                test_size=0.20, random_state=11)

# print(x4_train, x4_test, y4_train, y4_test)

print("Part b")
decision_tree = tree.DecisionTreeClassifier(max_depth=2, random_state=11)
decision_tree = decision_tree.fit(x4_train, y4_train)
# decision_tree_pred = decision_tree.predict(x4_test)
# print(decision_tree_pred)

sklearn.tree.plot_tree(decision_tree)
TODO: plt.show()

print("Part c")

params_dc = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'max_depth': np.arange(2, 15)}
grid_search_dc = GridSearchCV(decision_tree, params_dc)
grid_search_dc.fit(data_04.data, data_04.target)

print(grid_search_dc.best_params_)
print(grid_search_dc.best_score_)

print("Part d")
# Apply some gaussian noise to targets
np.random.seed(11)
noise = np.random.normal(0, 0.2, data_04.target.shape)
print(noise)

print("Part e")
params_dc_e = {'splitter': ['best', 'random'], 'max_depth': np.arange(2, 15)}
decision_tree_reg = DecisionTreeRegressor()
grid_search_dc_e = GridSearchCV(decision_tree_reg, params_dc_e)
grid_search_dc_e.fit(data_04.data, noise)

print(grid_search_dc_e.best_params_)
print(grid_search_dc_e.best_score_)