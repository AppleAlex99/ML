import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from keras.layers import Dense, Dropout
import keras
from sklearn import datasets, linear_model


# test tensorflow
#tf_hello = tf.constant("Hello, Tensorflow!")
#print(tf_hello)

# test numpy
#print('Numpy version: ' + np.__version__)

# test scikit-learn
#iris = datasets.load_iris()
#digits = datasets.load_digits()
#print(digits.data)

# test matplotlib
#plt.plot([1, 2, 3, 4])
#plt.ylabel('some numbers')
#plt.show()

# test pandas
#print('Pandas version: ' + pd.__version__)

################################################################################################

df = sklearn.datasets.load_diabetes(return_X_y=True, as_frame=True)
data = df[0]
target = df[1]
age_bmi_col = data["age"] / data["bmi"]
data["age_bmi"] = age_bmi_col
print(data)
print(target)

reg = sklearn.linear_model.LinearRegression()
reg.fit(data, target)
reg_pred = reg.predict(data)

tf_model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation=None)
])

tf_model.compile(optimizer='adam',
                 loss='mse',
                 metrics=['mae'])

tf_model.fit(data, target, epochs=25, batch_size=32)

nn_pred = tf_model.predict(data)

print('MAE with linear model: {}\nMAE with neural network: {}'.format(
    sklearn.metrics.mean_absolute_error(target, reg_pred),
    sklearn.metrics.mean_absolute_error(target, nn_pred)))
