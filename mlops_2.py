"""
================================
Recognizing hand-written digits
================================
This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.
"""

print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause
# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

# flatten the images
n_samples = len(digits.images)
print("Shape of original image : ",digits.images[0].shape)
print(" image taken with 16X16 with training data 70% and testing data 30%")


for gamma in np.arange(0.002, 0.020, 0.002):
    new_image = np.zeros((n_samples,16,16))
    for i in range(0,n_samples):
        new_image[i] = resize(digits.images[i],(16,16),anti_aliasing=True)
    data = new_image.reshape((n_samples, -1))
    clf = svm.SVC(gamma=gamma)
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.3, shuffle=False)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    print("Accuracy with 16x16 shape: " , metrics.accuracy_score(predicted , y_test))


