import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('images/einstein.png', cv.IMREAD_GRAYSCALE)

k1 = np.array([[1], [2], [1]])
k2 = np.array([[1, 0, -1]])

temp_x = cv.filter2D(img, -1, k1)
sobel_x_filter = cv.filter2D(temp_x, -1, k2)

temp_y = cv.filter2D(img, -1, k2.T)
sobel_y_filter = cv.filter2D(temp_y, -1, k1.T)

gradient_magnitude = np.sqrt(sobel_x_filter**2 + sobel_y_filter**2)

plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1), plt.imshow(img, cmap='gray'), plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 4, 2), plt.imshow(sobel_x_filter, cmap='gray'), plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 4, 3), plt.imshow(sobel_y_filter, cmap='gray'), plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 4, 4), plt.imshow(gradient_magnitude, cmap='gray'), plt.title('Gradient Magnitude'), plt.xticks([]), plt.yticks([])

plt.show()
