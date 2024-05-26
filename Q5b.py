import cv2
import numpy as np

# Load image
image = cv2.imread('images/rice_gaussian_noise.png', cv2.IMREAD_GRAYSCALE)

# Preprocessing (Gaussian blur)
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Thresholding (Otsu's method)
_, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Morphological operations (opening and closing)
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=2)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

# Connected component analysis
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closing)

# Count of rice grains (excluding background)
rice_count = num_labels - 1 

print(f"Number of rice grains in the image: {rice_count}")
