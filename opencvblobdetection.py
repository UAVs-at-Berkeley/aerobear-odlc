from matplotlib import pyplot as plt
import numpy as np
import cv2

# loading image
#img0 = cv2.imread('image.jpg',)
#img0 = cv2.imread('SanFrancisco.jpg',)
#img0 = cv2.imread('pic.png',)
#img0 = cv2.imread('windows.jpeg',)
img0 = cv2.imread('maryland_test_50.png',)

# converting to gray scale
gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

# remove noise
img = cv2.GaussianBlur(gray,(3,3),0)

# convolute with proper kernels
laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y
c = cv2.Canny(gray, 10, 20, 3, L2gradient=True)
plt.subplot(2,2,1),plt.imshow(c,cmap = 'gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()