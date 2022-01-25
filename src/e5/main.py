
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from keras.layers import Dense, Flatten
from keras.models import Sequential
from scipy import ndimage
from keras.layers import Reshape
from keras.layers import Conv2D

tf.random.set_seed(11)

print("###################Task 1###################")
print("Part a")
(x_train_01, y_train_01), (x_test_01, y_test_01) = tf.keras.datasets.fashion_mnist.load_data()

# Scale the data
x_train_01 = tf.keras.utils.normalize(x_train_01, axis=1)
x_test_01 = tf.keras.utils.normalize(x_test_01, axis=1)

print("Part b")
# Select random images from training set
img1 = x_train_01[2]
img2 = x_train_01[20]
img3 = x_train_01[200]
img4 = x_train_01[2000]
img5 = x_train_01[20000]

# Creating weigths
w1 = ([[-1, -1, -1],[2, 2, 2], [-1, -1, -1]])
w2 = ([[-1, 2, -1],[-1, 2, -1], [-1, 2, -1]])
w3 = ([[-1, -1, 2],[-1, 2, -1], [2, -1, -1]])

# Applying Weigths for image 1
wimg1_1 = ndimage.convolve(img1, w1)
wimg1_2 = ndimage.convolve(img1, w2)
wimg1_3 = ndimage.convolve(img1, w3)

# Weigths for image 2
wimg2_1 = ndimage.convolve(img2, w1)
wimg2_2 = ndimage.convolve(img2, w2)
wimg2_3 = ndimage.convolve(img2, w3)

# Weigths for image 3
wimg3_1 = ndimage.convolve(img3, w1)
wimg3_2 = ndimage.convolve(img3, w2)
wimg3_3 = ndimage.convolve(img3, w3)

# Weigths for image 4
wimg4_1 = ndimage.convolve(img4, w1)
wimg4_2 = ndimage.convolve(img4, w2)
wimg4_3 = ndimage.convolve(img4, w3)

# Weigths for image 5
wimg5_1 = ndimage.convolve(img5, w1)
wimg5_2 = ndimage.convolve(img5, w2)
wimg5_3 = ndimage.convolve(img5, w3)
#%%
print("Part c")
# Plot
image1 = x_train_01[2]# plot the sample
f, ax = plt.subplots(1, 4, figsize=(20, 20))
a1 = [image1, wimg1_1, wimg1_2, wimg1_3]
for i in range(0, 4):
    sample = a1[i]
    ax[i].imshow(sample)
plt.show()

image2 = x_train_01[20]# plot the sample
f2, ax2 = plt.subplots(1, 4, figsize=(20, 20))
a2 = [image2, wimg2_1, wimg2_2, wimg2_3]
for i in range(0, 4):
    sample = a2[i]
    ax2[i].imshow(sample)
plt.show()

image3 = x_train_01[200]# plot the sample
f3, ax3 = plt.subplots(1, 4, figsize=(20, 20))
a3 = [image3, wimg3_1, wimg3_2, wimg3_3]
for i in range(0, 4):
    sample = a3[i]
    ax3[i].imshow(sample)
plt.show()

image4 = x_train_01[2000]# plot the sample
f4, ax4 = plt.subplots(1, 4, figsize=(20, 20))
a4 = [image4, wimg4_1, wimg4_2, wimg4_3]
for i in range(0, 4):
    sample = a4[i]
    ax4[i].imshow(sample)
plt.show()

image5 = x_train_01[20000]# plot the sample
f5, ax5 = plt.subplots(1, 4, figsize=(20, 20))
a5 = [image5, wimg5_1, wimg5_2, wimg5_3]
for i in range(0, 4):
    sample = a5[i]
    ax5[i].imshow(sample)
plt.show()

# The first convolution layer is highlighting the horizontal lines of the images
# The second convolution layer is highlighting the vertical lines of the images
# The third convolution layer is highlighting...

print("###################Task 2###################")

print("Part a")
(x_train_02, y_train_02), (x_test_02, y_test_02) = tf.keras.datasets.fashion_mnist.load_data()

# Scale the data
x_train_02 = tf.keras.utils.normalize(x_train_02, axis=1)
x_test_02 = tf.keras.utils.normalize(x_test_02, axis=1)

# Create neural network with given layers
print("Part b")
model2b = Sequential([
    Reshape((28,28,1), input_shape=(28,28)),
    Conv2D(128, kernel_size=(3,3)),
    Conv2D(64, kernel_size=(3,3)),
    Flatten(),
    Dense(64),
    Dense(10, activation='softmax')
])

# Compile the cnn
model2b.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

# Fit the cnn
model2b.fit(x_train_02, y_train_02, epochs=5)

# Evaluate the nn on the training set
val_loss_2b, val_acc_2b = model2b.evaluate(x_train_02, y_train_02)
print("Accuracy for test set:", val_acc_2b)

# Print summary
print(model2b.summary())

# Explain output-shapes
# Conv2d: The output shape for the first convolutional layer describes shows the reduction of the 28x28 images to 26x26 with a kernel of 3x3
#         The third number shows the number of neurons in this conv layer
# Conv2d_1: It shows the next reduction step in another convolutional layer, so the output shape is 24x24 because of the second 3x3 kernel
#           The third number is also the number of neurons of this layer

from keras.layers import MaxPooling2D

print("Part c")
model2c = Sequential([
    Reshape((28,28,1)),
    Conv2D(128, kernel_size=(3,3)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3,3)),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(64),
    Dense(10, activation='softmax')
])

# Compile the nn
model2c.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

# Fit the nn
model2c.fit(x_train_02, y_train_02, epochs=5)

# Evaluate the nn on the training set
val_loss_2c, val_acc_2c = model2c.evaluate(x_test_02, y_test_02)
print("Accuracy for test set:", val_acc_2c)

# Print summary
print(model2c.summary())

# Compare performance with model b
# The performance of the model with maxpooling layer compiled much faster than the model without this kind of layer
# On addition the accuracy of this model is also a little bit better


from keras.datasets import cifar10

print("###################Task 3###################")
print("Part a")
(x_train_03, y_train_03), (x_test_03, y_test_03) = cifar10.load_data()

# Overview of the images
print('Train: X=%s, y=%s' % (x_train_03.shape, y_train_03.shape))
print('Test: X=%s, y=%s' % (x_test_03.shape, y_test_03.shape))
# plot first few images
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(x_train_03[i])
plt.show()

# Scale images
x_train_03 = tf.keras.utils.normalize(x_train_03, axis=1)
x_test_03 = tf.keras.utils.normalize(x_test_03, axis=1)

# Visualize scaled images
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(x_train_03[i])
plt.show()

print("Part b")
import tensorflow_hub as hub

# Load the model as keras layer and create model
keras_layer = hub.KerasLayer("https://tfhub.dev/deepmind/ganeval-cifar10-convnet/1")
model3 = Sequential()
model3.add(keras_layer)

# Show structure
model3.build(input_shape=(None,32,32,3))
model3.summary()

print("Part c")
# Predict the classes
model3.predict(x_test_03)




# Visualize the predicted probabilities

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

print("###################Task 5###################")
RANDOM_STATE = 11
print("Part a")
# Read csv
data, target = fetch_openml('eeg-eye-state', return_X_y=True, as_frame=True)

# Scale the data
data = tf.keras.utils.normalize(data, axis=1)
# print(data)

# Split the data into train test split
x_train_05, x_test_05, y_train_05, y_test_05 = train_test_split(data, target, test_size=0.2, random_state=RANDOM_STATE)
# print(x_train_05, y_train_05, x_test_05, y_test_05)

# Plot the data (just for myself)
plt.figure(figsize=(8, 5))
plt.plot(range(len(data)), data)
plt.show()

print("Part b")
sliding_windows_train = tf.keras.utils.timeseries_dataset_from_array(x_train_05, y_train_05, sequence_length=10)
print(sliding_windows_train)

sliding_windows_test = tf.keras.utils.timeseries_dataset_from_array(x_test_05, y_test_05, sequence_length=10)
print(sliding_windows_test)
# This function takes in a sequence of data-points gathered at equal intervals, along with time series parameters
# such as length of the sequences/windows, spacing between two sequence/windows, etc., to produce batches of timeseries
# inputs and targets.
# The structure is ((None, None, 14), (None)) for both arrays

from keras.layers import InputLayer, TimeDistributed, SimpleRNN, Dropout
print("Part c")
model5 = Sequential([
    InputLayer(input_shape=(None, None, 14)),
    TimeDistributed(Dense(128, activation='relu')),
    TimeDistributed(Dense(256, activation='relu')),
    SimpleRNN(128, return_sequences=True),
    SimpleRNN(128),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])
model5.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model5.fit(x_train_05, y_train_05, epochs=15)

