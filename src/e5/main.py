import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

print("###################Task 1###################")
(x_train_01, y_train_01), (x_test_01, y_test_01) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

scaler = MinMaxScaler()
x_train_01 = scaler.fit_transform(x_train_01.reshape(-1, x_train_01.shape[-1])).reshape(x_train_01.shape)
x_test_01 = scaler.transform(x_test_01.reshape(-1, x_test_01.shape[-1])).reshape(x_test_01.shape)

img1 = x_train_01[2]
print(img1)
img2 = x_train_01[20]
print(img2)
img3 = x_train_01[200]
print(img3)
img4 = x_train_01[2000]
print(img4)
img5 = x_train_01[2000]
print(img5)