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
img_new = cv.cvtColor(img_new, cv.COLOR_LAB2BGR)
img_original = cv.cvtColor(img_original, cv.COLOR_BGR2RGB)

fig, axarr = plt.subplots(3, 2)
axarr[0, 0].imshow(img_original)
axarr[0, 1].imshow(img_new)
color = ('b', 'g', 'r')
for i, c in enumerate(color):
    hist_org = cv.calcHist([img_original], [i], None, [256], [0, 256])
    axarr[1, 0].plot(hist_org, color=c)
    hist_gamma = cv.calcHist([img_new], [i], None, [256], [0, 256])
    axarr[1, 1].plot(hist_gamma, color=c)

axarr[2, 0].plot(table)
axarr[2, 0].set_xlim([0, 256])
plt.show()
