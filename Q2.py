import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# read image

img_original = cv.imread('images/highlights_and_shadows.jpg', cv.IMREAD_COLOR)
assert not img_original is None

l, a, b = cv.split(cv.cvtColor(img_original, cv.COLOR_RGB2LAB))

# we applying gamma correction to Lighter plane so gamma < 1 to brighten image and decrease contrast
gamma = 0.6
table = np.array(
    [((i / 255) ** (gamma)) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
l_new = cv.LUT(l, table)
img_new = cv.merge((l_new, a, b))
img_new = cv.cvtColor(img_new, cv.COLOR_LAB2RGB)
cv.namedWindow('Original Image', cv.WINDOW_NORMAL)
cv.imshow('Original Image', img_original)
cv.waitKey(0)

cv.namedWindow('Transformed Image', cv.WINDOW_NORMAL)
cv.imshow('Transformed Image', img_new)
cv.waitKey(0)
cv.destroyAllWindows()

print(img_original.shape)
print(img_new.shape)

# color space 3 in both new and original image

color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr1 = cv.calcHist([img_original], [i], None, [256], [0, 256])
    plt.plot(histr1, color=col)
    plt.xlim([0, 256])
plt.show()

for i, col in enumerate(color):
    histr2 = cv.calcHist([img_new], [i], None, [256], [0, 256])
    plt.plot(histr2, color=col)
    plt.xlim([0, 256])
plt.show()
