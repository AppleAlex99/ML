import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from keras.layers import Dense, Dropout
import keras
from sklearn import *
from sklearn.preprocessing import *

df_t1 = sklearn.datasets.load_wine(return_X_y=True, as_frame=True)
data = df_t1[0]
print("Data: \n", data)

target = df_t1[1]
print("Target: \n", target)

print("-----------------------------------------1a)-----------------------------------------")

print("Number of rows: ", len(data.index))

print("Attributes: ", len(data.columns))

print("Attributes names: ", data.columns)

print("Datatypes:\n", data.dtypes)

##################

print("Min: ", data["ash"].min())

print("Max: ", data["ash"].max())

print("Median: ", data["ash"].median())

print("Mean: ", data["ash"].mean())

print("Classes: ", len(target.unique()))

print("Examples associated to Class: \n", target.value_counts())

print("-----------------------------------------1b)-----------------------------------------")

ash = data["ash"]

plt.hist(ash, bins=10, edgecolor='black')
# plt.show()
print("Histogram!!!")

plt.boxplot(ash)
# plt.show()
print("BoxPlot!!!")
# Most of the data is in between 2.25 and 2.50

print("-----------------------------------------2a)-----------------------------------------")

data['substracted_phenols'] = data['total_phenols'] - data['nonflavanoid_phenols']
print(data['substracted_phenols'])

data['alcohol'] = data['alcohol'] + 1.0

print("-----------------------------------------2b)-----------------------------------------")
data_shuffled = data.sample(n=len(data), random_state=11)

random_data = data.sample(n=20, random_state=11)
np.random.seed(11)
noise = np.random.normal(0, 1, random_data.shape)
random_data = random_data + noise

data_shuffled = data_shuffled.append(random_data)

# Add label
data_shuffled['labels'] = target

print(data_shuffled)

print("-----------------------------------------2c)-----------------------------------------")
data_shuffled['ash'] = sklearn.preprocessing.minmax_scale(data_shuffled['ash'], feature_range=(0, 1), axis=0, copy=True)
print(data_shuffled['ash'])
# with minmax u can

# Split the dataframe
training_set_shuffled, test_set_shuffled = sklearn.model_selection.train_test_split(data_shuffled, test_size=0.15, random_state=11)

# print(training_set_shuffled)
# print(test_set_shuffled)


print("-----------------------------------------3a)-----------------------------------------")
df_t3 = sklearn.datasets.load_wine(return_X_y=True, as_frame=True)

data_temp = df_t3[0]
target_temp = df_t3[1]

data_temp['labels'] = target_temp
data_temp['labels'] = data_temp['labels'].replace(to_replace=[0, 1, 2], value=["class-0", "not-class-0", "not-class-0"])

# print(data_temp['labels'])

training_set, test_set = sklearn.model_selection.train_test_split(data_temp, test_size=0.15, random_state=11)
# print(training_set)
# print(test_set)

# decision tree
x_temp = training_set.iloc[:, :-1]
y_temp = training_set['labels']

tree = sklearn.tree.DecisionTreeClassifier()
tree = tree.fit(x_temp, y_temp)
tree_pred = tree.predict(test_set.iloc[:, :-1])
# print(tree_pred)

# confusion matrix
matrix_temp = sklearn.metrics.confusion_matrix(test_set['labels'], tree_pred, labels=None, sample_weight=None, normalize=None)
print(matrix_temp)

# 26/27 = 96.29% accuracy
print("Accuracy for not-preprocessed data is: ", 26/27)


print("-----------------------------------------3b)-----------------------------------------")
training_set_shuffled['labels'] = training_set_shuffled['labels'].replace(to_replace=[0, 1, 2], value=["class-0", "not-class-0", "not-class-0"])
test_set_shuffled['labels'] = test_set_shuffled['labels'].replace(to_replace=[0, 1, 2], value=["class-0", "not-class-0", "not-class-0"])

x = training_set_shuffled.iloc[:, :-1]
y = training_set_shuffled['labels']

tree_shuffled = sklearn.tree.DecisionTreeClassifier()
tree_shuffled = tree_shuffled.fit(x, y)
tree_shuffled_pred = tree_shuffled.predict(test_set_shuffled.iloc[:, :-1])
# print(tree_shuffled_pred)

matrix = sklearn.metrics.confusion_matrix(test_set_shuffled['labels'], tree_shuffled_pred, labels=None, sample_weight=None, normalize=None)
print(matrix)
# 29/30 = 96.66% accuracy
print("Accuracy for preprocessed data is: ", 29/30)

# The data transformation was a little bit useful but just because of the
# higher percentage for the true positive & true negative values
# A comparison doesnt make sense because of the very minor differences

print("-----------------------------------------4a)-----------------------------------------")

data_house, target_house = sklearn.datasets.fetch_california_housing(return_X_y=True, as_frame=True)

print(data_house)
print(target_house)

print(data_house.isnull().values.any())
# If missing values, True else False

# Correlation with Pearson model
print("Correlation between the attributes:")
print(data_house.corr(method='pearson', min_periods=1))

plt.matshow(data_house.corr())
# plt.show()

print("-----------------------------------------4b)-----------------------------------------")

data_house['HOL'] = target_house
















