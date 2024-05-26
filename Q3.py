import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('images/spider.png', cv.IMREAD_COLOR)
img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

h, s, v = cv.split(img)

sigma = 70

a_val = np.linspace(0, 1, 11)
modified_images = []

for a in a_val:
    func = np.minimum(s + a * 128 * np.exp(-(s - 128) ** 2 /
                      (2 * sigma ** 2)), 255).astype(np.uint8)
    img_new = cv.merge((h, func, v))
    img_new = cv.cvtColor(img_new, cv.COLOR_HSV2BGR)
    modified_images.append(img_new)

fig, ax = plt.subplots(2, 5, figsize=(15, 8))

for i, ax in enumerate(ax.flat):
    ax.imshow(cv.cvtColor(modified_images[i], cv.COLOR_BGR2RGB))
    ax.set_title(f'a = {a_val[i]:.2f}')
    ax.axis('off')

plt.tight_layout()
plt.show()
