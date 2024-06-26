import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('images/shells.tif', cv.IMREAD_GRAYSCALE)

hist, bins = np.histogram(img.ravel(), 256, [0, 256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(cdf_normalized, color='b')
plt.hist(img.flatten(), 256, [0, 256], color='r', alpha=0.5)
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.title('Original Image Histogram')

equ = cv.equalizeHist(img)

hist, bins = np.histogram(equ.ravel(), 256, [0, 256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()


plt.subplot(1, 2, 2)
plt.plot(cdf_normalized, color='b')
plt.hist(equ.flatten(), 256, [0, 256], color='r', alpha=0.5)
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.title('Equalized Image Histogram')
plt.show()


cv.imshow('Original Image', img)
cv.imshow('Equalized Image', equ)
cv.waitKey(0)
cv.destroyAllWindows()
