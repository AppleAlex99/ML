import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn import *
from sklearn.preprocessing import minmax_scale

print("###################Task 1###################")
print("Part a")


print("###################Task 2###################")
print("Part a")

print("###################Task 3###################")

tf.random.set_seed(11)

print("Part a")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

print("Data train:\n", x_train)
print("Target train:\n", y_train)
print("Data test:\n", x_test)
print("Target test:\n", y_test)

