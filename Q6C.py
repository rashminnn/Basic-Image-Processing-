import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Load image (ensure the path is correct)
img = cv.imread('images/einstein.png', cv.IMREAD_GRAYSCALE)

# Define the smaller filters based on the property
kernel_x = np.array([[-1, 0, 1]])
kernel_y = np.array([[1], [2], [1]])

# Apply horizontal and vertical filtering separately
sobel_x = cv.filter2D(img, cv.CV_64F, kernel_x)
sobel_y = cv.filter2D(img, cv.CV_64F, kernel_y.T)

# Combine to get the final Sobel results (equivalent to applying the full filter)
sobel_combined = np.sqrt(np.abs(sobel_x)**2 + np.abs(sobel_y)**2)
sobel_combined = cv.normalize(sobel_combined, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

# Display results
plt.figure(figsize=(12, 4))
plt.subplot(141), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(142), plt.imshow(sobel_x, cmap='gray'), plt.title('Sobel X (Vertical Edges)')
plt.subplot(143), plt.imshow(sobel_y, cmap='gray'), plt.title('Sobel Y (Horizontal Edges)')
plt.subplot(144), plt.imshow(sobel_combined, cmap='gray'), plt.title('Sobel Combined (Gradient Magnitude)')
plt.show()
