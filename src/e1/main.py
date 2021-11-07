import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from keras.layers import Dense, Dropout
import keras
from sklearn import datasets, linear_model

data = sklearn.datasets.load_wine(return_X_y=True, as_frame=True)
wine_data = data[0]
print("Data: \n", wine_data)

wine_target = data[1]
print("Target: \n", wine_target)

########################## 1a) ############################

print("Number of rows: ", len(wine_data.index))

print("Attributes: ", len(wine_data.columns))

print("Attributes names: ", wine_data.columns)

print("Datatypes:\n", wine_data.dtypes)

##################

print("Min: ", wine_data["ash"].min())

print("Max: ", wine_data["ash"].max())

print("Median: ", wine_data["ash"].median())

print("Mean: ", wine_data["ash"].mean())

print("Classes: ", len(wine_target.unique()))

print("Examples associated to Class: \n", wine_target.value_counts())

########################## 1b) ############################

ash = wine_data["ash"]

plt.hist(ash, bins=10, edgecolor='black')
plt.show()

plt.boxplot(ash)
plt.show()
#Most of the data is in between 2.25 and 2.50

########################## 2a) ############################