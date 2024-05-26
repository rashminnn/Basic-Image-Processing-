import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


img = cv.imread('images/rice_gaussian_noise.png', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('images/rice_salt_pepper_noise.png' , cv.IMREAD_GRAYSCALE)

gaussian_blur = cv.GaussianBlur(img, (5, 5), 0)
median_blur = cv.medianBlur(img2, 5)

plt.subplots(1, 3, figsize=(15, 5))
plt.subplot(1, 3, 1),plt.imshow(img, cmap='gray'),plt.title("Original Image"),plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 2),plt.imshow(gaussian_blur, cmap='gray'),plt.title(f'Gaussian Blur'),plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 3),plt.imshow(median_blur, cmap='gray'),plt.title(f'Median Blur'),plt.xticks([]), plt.yticks([])
plt.show()

img_for_segmentation = median_blur
_, otsu_thresh = cv.threshold(img_for_segmentation, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
morph_open = cv.morphologyEx(otsu_thresh, cv.MORPH_OPEN, kernel, iterations=2)
morph_close = cv.morphologyEx(morph_open, cv.MORPH_CLOSE, kernel, iterations=2)

num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(morph_close, connectivity=8)

plt.subplot(1, 3, 1),plt.imshow(otsu_thresh, cmap='gray'),plt.title("Otsu's Threshold"),plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 2),plt.imshow(morph_close, cmap='binary'),plt.title("Morphological Operations"),plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 3),plt.imshow(labels, cmap='tab10', alpha = 0.5),plt.title(f'Connected Components: {num_labels - 1} objects'),plt.xticks([]), plt.yticks([])
plt.show()