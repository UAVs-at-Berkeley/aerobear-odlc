from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from skimage.io import imread
from skimage import data_dir
import cv2
import numpy as np
import matplotlib.pyplot as plt

#Read in image, currently set to "maryland_test_50.png change as needed"
image = imread("maryland_test_50.png")
grey = rgb2gray(image)

"""
#One method of masking for background color, unclear if this actually does anything
mask = image < (image.mean() + 80)
mask = mask > (image.mean() - 80)
image[mask] = 0
grey = rgb2gray(image)
"""

"""
#Another method of masking for background color, tighter bounds than the method below
lower = np.array([70,70,50])
upper = np.array([110,110,90])
mask = cv2.inRange(image, lower, upper)
img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
"""

"""
#Used for generating a mask of approximately the background color of maryland_test_50.png
u = np.array([125,125,105])
low = np.array([55,55,35])
maskbgr = cv2.inRange(image, low, u)
grey = rgb2gray(maskbgr)
"""


#use skimage difference of gaussian to detect blobs
#parameters around max_sigma = 10 and threshold = .18 seem to work best for maryland_test_50.png
#reconmend tuning max_sigma and thershold
#blobs_log is faster but less accurate
#blobs_doh is slower but more accurate theoretically, but in practice doesn't seem better
blobs_dog = blob_dog(grey, max_sigma=10, threshold=.18)
blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

#Circle each blob and plot
fig, ax = plt.subplots(1)
ax.imshow(image)
for blob in blobs_dog:
    y, x, r = blob
    c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
    ax.add_patch(c)
plt.tight_layout()
plt.show()
