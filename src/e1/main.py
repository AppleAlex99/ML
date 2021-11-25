import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn import *
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import *

data, target = sklearn.datasets.load_wine(return_X_y=True, as_frame=True)

print("Data: \n", data)
print("Target: \n", target)

print("-----------------------------------------1a)-----------------------------------------")

print("Examples: ", len(data.index))

print("Attributes: ", len(data.columns))

print("Attributes names: ", data.columns)

print("Datatypes:\n", data.dtypes)

##################

print("Min: ", data["ash"].min())

print("Max: ", data["ash"].max())

print("Median: ", data["ash"].median())

print("Mean: ", data["ash"].mean())

print("Classes: ", len(target.unique()))

print("Examples associated to Class:\n", target.value_counts())

print("-----------------------------------------1b)-----------------------------------------")

ash = data["ash"]
n, bins, edges = plt.hist(ash, bins=10, ec="blue", alpha=0.7)
plt.xticks(bins)
plt.show()

plt.boxplot(ash)
plt.show()

# Most of the data is in between 2.25 and 2.50,
# the median is the red line and there are a few outlier

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
# with minmax one can scale the values between 0 and 1

# Split the dataframe
training_set_shuffled, test_set_shuffled = sklearn.model_selection.train_test_split(data_shuffled, test_size=0.15,
                                                                                    random_state=11)

# print(training_set_shuffled)
# print(test_set_shuffled)

print("-----------------------------------------3a)-----------------------------------------")
data_temp, target_temp = sklearn.datasets.load_wine(return_X_y=True, as_frame=True)

data_temp['labels'] = target_temp
data_temp['labels'] = data_temp['labels'].replace(to_replace=[0, 1, 2], value=["class-0", "not-class-0", "not-class-0"])

print(data_temp['labels'])

training_set, test_set = sklearn.model_selection.train_test_split(data_temp, test_size=0.15, random_state=11)
print(training_set)
print(test_set)

# decision tree
x_temp = training_set.iloc[:, :-1]
y_temp = training_set['labels']

tree = sklearn.tree.DecisionTreeClassifier()
tree = tree.fit(x_temp, y_temp)
tree_pred = tree.predict(test_set.iloc[:, :-1])
print(tree_pred)

# confusion matrix
matrix_temp = sklearn.metrics.confusion_matrix(test_set['labels'], tree_pred, labels=None, sample_weight=None,
                                               normalize=None)
print(matrix_temp)

# 26/27 = 96.29% accuracy
print("Accuracy for not-preprocessed data is: ", 26 / 27)

print("-----------------------------------------3b)-----------------------------------------")
training_set_shuffled['labels'] = training_set_shuffled['labels'].replace(to_replace=[0, 1, 2],
                                                                          value=["class-0", "not-class-0",
                                                                                 "not-class-0"])
test_set_shuffled['labels'] = test_set_shuffled['labels'].replace(to_replace=[0, 1, 2],
                                                                  value=["class-0", "not-class-0", "not-class-0"])

x = training_set_shuffled.iloc[:, :-1]
y = training_set_shuffled['labels']

tree_shuffled = sklearn.tree.DecisionTreeClassifier()
tree_shuffled = tree_shuffled.fit(x, y)
tree_shuffled_pred = tree_shuffled.predict(test_set_shuffled.iloc[:, :-1])
# print(tree_shuffled_pred)

matrix = sklearn.metrics.confusion_matrix(test_set_shuffled['labels'], tree_shuffled_pred, labels=None,
                                          sample_weight=None, normalize=None)
print(matrix)
# 29/30 = 96.66% accuracy
print("Accuracy for preprocessed data is: ", 29 / 30)

# The data transformation was a little bit useful but just because of the
# higher percentage for the true positive & true negative values
# A comparison doesnt make sense because of the very minor differences

print("-----------------------------------------4a)-----------------------------------------")

data_house, target_house = sklearn.datasets.fetch_california_housing(return_X_y=True, as_frame=True)

# print(data_house)
# print(target_house)

print("Missing data?:", data_house.isnull().values.any())
# If missing values, True else False,
# so no missing value in this dataframe

# Correlation with Pearson model
print("Correlation between the attributes:")
print("Correlation matrix:\n", data_house.corr())

plt.matshow(data_house.corr())
plt.show()

print("-----------------------------------------4b)-----------------------------------------")

data_house['HOL'] = 0
data_house.loc[(data_house['HouseAge'] > 25) & (data_house['AveBedrms'] > 3), 'HOL'] = 1
print(data_house['HOL'])

print("-----------------------------------------4c)-----------------------------------------")
training_set_house, test_set_house = sklearn.model_selection.train_test_split(data_house, test_size=0.20,
                                                                              random_state=11)
print("Training set house:\n", training_set_house)

print("-----------------------------------------4d)-----------------------------------------")
lin_reg = sklearn.linear_model.LinearRegression(fit_intercept=True, normalize='deprecated', copy_X=True, n_jobs=None,
                                                positive=False)

x_house = training_set_house.iloc[:, :-1]
y_house = training_set_house['HOL']
z_house = test_set_house.iloc[:, : -1]

lin_reg = lin_reg.fit(x_house, y_house)
lin_reg_pred = lin_reg.predict(z_house)
print("Linear regression predict test:\n", lin_reg_pred)

print("-----------------------------------------4e)-----------------------------------------")
mse_test = sklearn.metrics.mean_squared_error(test_set_house['HOL'], lin_reg_pred, sample_weight=None,
                                              multioutput='uniform_average', squared=True)
print("Mean Squared Error Test:")
print(mse_test)

rmse_test = sklearn.metrics.mean_squared_error(test_set_house['HOL'], lin_reg_pred, sample_weight=None,
                                               multioutput='uniform_average', squared=False)
print("Root Mean Squared Error Test:")
print(rmse_test)

mae_test = sklearn.metrics.mean_absolute_error(test_set_house['HOL'], lin_reg_pred, sample_weight=None,
                                               multioutput='uniform_average')
print("Mean Absolute Error Test:")
print(mae_test)

########## new predict with training_set_house

lin_reg_pred_train = lin_reg.predict(x_house)
print("Linear regression predict train:\n", lin_reg_pred_train)

mse_train = sklearn.metrics.mean_squared_error(training_set_house['HOL'], lin_reg_pred_train, sample_weight=None,
                                               multioutput='uniform_average', squared=True)
print("Mean Squared Error Train:")
print(mse_train)

rmse_train = sklearn.metrics.mean_squared_error(training_set_house['HOL'], lin_reg_pred_train, sample_weight=None,
                                                multioutput='uniform_average', squared=False)
print("Root Mean Squared Error Train:")
print(rmse_train)

mae_train = sklearn.metrics.mean_absolute_error(training_set_house['HOL'], lin_reg_pred_train, sample_weight=None,
                                                multioutput='uniform_average')
print("Mean Absolute Error Train:")
print(mae_train)

# The mae value is round about 0.0037. That describes the absolute value
# of the difference between the forecasted value and the actual value.
# In this specfic case, the mae is very low, so the forecast ist ok

# The model is overfitting

print("-----------------------------------------5a)-----------------------------------------")

data_house_e5, target_house_e5 = sklearn.datasets.fetch_california_housing(return_X_y=True, as_frame=True)
data_house_e5['Labels'] = target_house_e5
print(data_house_e5)

training_set_house_e5, test_set_house_e5 = sklearn.model_selection.train_test_split(data_house_e5, test_size=0.20,
                                                                                    random_state=11)

print("-----------------------------------------5b)-----------------------------------------")

lasso_reg = sklearn.linear_model.Lasso(alpha=1.0)
lasso_reg = lasso_reg.fit(training_set_house_e5.iloc[:, : -1], training_set_house_e5['Labels'])

print("-----------------------------------------5c)-----------------------------------------")

lasso_reg_pred = lasso_reg.predict(test_set_house_e5.iloc[:, : -1])
print("Lasso predict:\n", lasso_reg_pred)

mae_test_5 = sklearn.metrics.mean_absolute_error(test_set_house_e5['Labels'], lasso_reg_pred, sample_weight=None,
                                                 multioutput='uniform_average')
print("Mean Absolute Error:\n", mae_test_5)

print("-----------------------------------------5d)-----------------------------------------")

print("Coefficients:\n", lasso_reg.coef_)
# There are 5 values close to zero or equal to zero
# This information shows that some parameters have no or very little influence on targeting

print("-----------------------------------------5e)-----------------------------------------")

training_set_house_e5_rem = training_set_house_e5.drop(['AveRooms', 'AveBedrms', 'AveOccup', 'Latitude', 'Longitude'],
                                                       axis=1)
test_set_house_e5_rem = test_set_house_e5.drop(['AveRooms', 'AveBedrms', 'AveOccup', 'Latitude', 'Longitude'], axis=1)

lasso_reg.fit(training_set_house_e5_rem, training_set_house_e5['Labels'])
lasso_reg_pred_rem = lasso_reg.predict(test_set_house_e5_rem)
print('MAE with lasso regression and removed features:  ',
      sklearn.metrics.mean_absolute_error(test_set_house_e5['Labels'], lasso_reg_pred_rem))
# Feature removal did increase  the performance of the model just a little bit.
