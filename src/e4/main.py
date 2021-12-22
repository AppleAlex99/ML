import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from keras.layers import Flatten, Dense, Dropout
from keras.models import Sequential
from sklearn import *
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

print("###################Task 1###################")
print("Part a")

print("###################Task 2###################")
print("Part a")

print("###################Task 3###################")

tf.random.set_seed(11)

print("Part a")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

# Visualize the data
f, ax = plt.subplots(1, 10, figsize=(20, 20))

for i in range(0, 10):
    sample = x_train[y_train == i][3]
    ax[i].imshow(sample, cmap='gray')
    ax[i].set_title("Label: {}".format(i), fontsize=16)

# plt.show()

# Scale the data
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)

print("Part b")

# Create Neural Network
model_b = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])
model_b.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

model_b.fit(x_train, y_train, epochs=5)

# The training accuracy is 0.9649
val_loss_b, val_acc_b = model_b.evaluate(x_test, y_test)
print("Accuracy for test set:", val_acc_b)
# The test accuracy is 0.9621

print("Part c")

# Improve the model
model_c = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
model_c.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_c.fit(x_train, y_train, epochs=5)

# The training accuracy is 0.9870

val_loss_c, val_acc_c = model_c.evaluate(x_test, y_test)
print("Accuracy for test set:", val_acc_c)

# The test accuracy is
# The accuracy of the deeper model is more accurat than the other one
# On the other hand it needs more data and is much more slower

print("###################Task 4###################")
(x_train_04, y_train_04), (x_test_04, y_test_04) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

# Scale the data
scaler = MinMaxScaler()
x_train_04 = scaler.fit_transform(x_train_04.reshape(-1, x_train_04.shape[-1])).reshape(x_train_04.shape)
x_test_04 = scaler.transform(x_test_04.reshape(-1, x_test_04.shape[-1])).reshape(x_test_04.shape)

print("Part a and b")
# Improve the model
model_a = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dropout(0.25),
    Dense(10, activation='softmax')
])
model_a.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_a.fit(x_train_04, y_train_04, epochs=5)
