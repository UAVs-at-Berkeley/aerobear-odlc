from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import *
from skimage.io import imread
import numpy as np
from skimage import data_dir
import cv2
import matplotlib.pyplot as plt

"""
not really sure whats going on in this file,
I think I was testing a bunch of random stuff
"""

"""
there seems to be a few color settings for images, and certain operations
only seemed to work with certain types of images, but I could be wrong
Examples include BGR and HSV as well as just greyscale
"""
imgbgr = imread("maryland_test_50.png")
imghsv = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2HSV)

"""
#seems to be k-means for finding the most dominant/common color in image
#similar approach was used for color identification in text/shape
n_colors = 5
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
flags = cv2.KMEANS_RANDOM_CENTERS

_, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
_, counts = np.unique(labels, return_counts=True)
dominant = palette[np.argmax(counts)]
print(dominant)
"""

#Produces black and white images with shapes blacked out, varying amounts of noise
average = imgbgr.mean(axis=0).mean(axis=0)
print(average)
cv2.imshow('img', imgbgr)
u = np.array([125,125,105])
l = np.array([55,55,35])
lower = np.array([65,45,75])
upper = np.array([110,85,110]) 
maskhsv = cv2.inRange(imghsv, lower, upper)
maskbgr = cv2.inRange(imgbgr, l, u)
mask = maskhsv + maskbgr
cv2.imshow('bgr', maskbgr)
cv2.imshow('hsv', maskhsv)
cv2.imshow('mask', mask)
cv2.waitKey()

"""
#seems to be similar to an approach in skimageblob.py

image_gray = hsv2rgb(image)
image_gray = rgb2gray(image_gray)
image = cv2.imread('maryland_test_50.png')
original = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([33, 0, 238], dtype="uint8")
upper = np.array([135, 189, 255], dtype="uint8")
mask = cv2.inRange(image, lower, upper)
blobs_dog = blob_dog(image, max_sigma=40, threshold=.25)
blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

fig,ax = plt.subplots(1)
ax.imshow(image)
for blob in blobs_dog:
    y, x, r = blob
    c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
    ax.add_patch(c)
plt.tight_layout()
plt.show()
"""