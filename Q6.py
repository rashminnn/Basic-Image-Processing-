import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('images/einstein.png', cv2.IMREAD_GRAYSCALE)

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Sobel X
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Sobel Y

sobelx_manual = cv2.filter2D(img, -1, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
sobely_manual = cv2.filter2D(img, -1, np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))

gradient_magnitude = cv2.magnitude(sobelx, sobely)

plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1), plt.imshow(img, cmap='gray'), plt.title('Original Image'), plt.axis('off')
plt.subplot(1, 4, 2), plt.imshow(sobelx, cmap='gray'), plt.title('Sobel X'), plt.axis('off')
plt.subplot(1, 4, 3), plt.imshow(sobely, cmap='gray'), plt.title('Sobel Y'), plt.axis('off')
plt.subplot(1, 4, 4), plt.imshow(gradient_magnitude, cmap='gray'), plt.title('Gradient Magnitude'), plt.axis('off')

plt.tight_layout()
plt.show()
