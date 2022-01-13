#%%

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense, Flatten
from keras.models import Sequential
from scipy import ndimage
from keras.layers import Reshape
from keras.layers import Conv2D

#%%

print("###################Task 1###################")
print("Part a")
(x_train_01, y_train_01), (x_test_01, y_test_01) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

# Scale the data
x_train_01 = tf.keras.utils.normalize(x_train_01, axis=1)
x_test_01 = tf.keras.utils.normalize(x_train_01, axis=1)

#%%

print("Part b")
# Select random images from training set
img1 = x_train_01[2]
img2 = x_train_01[20]
img3 = x_train_01[200]
img4 = x_train_01[2000]
img5 = x_train_01[2000]

#%%

# Creating weigths
w1 = ([[-1, -1, -1],[2, 2, 2], [-1, -1, -1]])
w2 = ([[-1, 2, -1],[-1, 2, -1], [-1, 2, -1]])
w3 = ([[-1, -1, 2],[-1, 2, -1], [2, -1, -1]])

#%%

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
    ax[i].imshow(sample, cmap='gray')
plt.show()

image2 = x_train_01[20]# plot the sample
f2, ax2 = plt.subplots(1, 4, figsize=(20, 20))
a2 = [image2, wimg2_1, wimg2_2, wimg2_3]
for i in range(0, 4):
    sample = a2[i]
    ax2[i].imshow(sample, cmap='gray')
plt.show()

image3 = x_train_01[200]# plot the sample
f3, ax3 = plt.subplots(1, 4, figsize=(20, 20))
a3 = [image3, wimg3_1, wimg3_2, wimg3_3]
for i in range(0, 4):
    sample = a3[i]
    ax3[i].imshow(sample, cmap='gray')
plt.show()

image4 = x_train_01[2000]# plot the sample
f4, ax4 = plt.subplots(1, 4, figsize=(20, 20))
a4 = [image4, wimg4_1, wimg4_2, wimg4_3]
for i in range(0, 4):
    sample = a4[i]
    ax4[i].imshow(sample, cmap='gray')
plt.show()

image5 = x_train_01[20000]# plot the sample
f5, ax5 = plt.subplots(1, 4, figsize=(20, 20))
a5 = [image5, wimg5_1, wimg5_2, wimg5_3]
for i in range(0, 4):
    sample = a5[i]
    ax5[i].imshow(sample, cmap='gray')
plt.show()

# The convolutions are highlighting the different gray scales

#%%

print("###################Task 2###################")

print("Part a")
(x_train_02, y_train_02), (x_test_02, y_test_02) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

# Scale the data
x_train_02 = tf.keras.utils.normalize(x_train_02, axis=1)
x_test_02 = tf.keras.utils.normalize(x_test_02, axis=1)

#%%
# Create neural network with given layers
print("Part b")
model2 = Sequential([
    Reshape((28,28,1)),
    Conv2D(128, kernel_size=(3,3)),
    Conv2D(64, kernel_size=(3,3)),
    Flatten(),
    Dense(64),
    Dense(10, activation='softmax')
])

model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
model2.build(input_shape=(10))

model2.summary()

model2.fit(x_train_02, y_train_02, epochs=5)
