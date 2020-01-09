import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter


def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# number_of_colors parameter tells cluster algorithm how many colors to search for 
# ideal img would be cropped to have two primary colors (color of shape and alphanumeric)
def get_colors(img, number_of_colors):
    # resizes image to speed up processing
    modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)
    counts = Counter(labels)
    center_colors = [list(map(int, lst)) for lst in clf.cluster_centers_]
    # ordered_colors contains the rgb values in order of most frequent to least
    ordered_colors = [center_colors[i] for i in counts.keys()]
    return ordered_colors