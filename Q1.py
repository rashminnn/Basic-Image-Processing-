import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# assumed the points (222,170) , (222,214) and end point is (255,239)

arr = np.array([(222, 170), (222, 214)])
t1 = np.linspace(0, arr[0, 1], arr[0, 0]+1).astype(np.uint8)
t2 = np.linspace(arr[0, 1], arr[1, 1], 0).astype(np.uint8)
t3 = np.linspace(arr[1, 1]+1, 239, 255-arr[1, 0]).astype(np.uint8)

trans = np.concatenate((t1, t2, t3), axis=0).astype(np.uint8)
print(len(trans))
plt.plot(trans)
plt.xlim([0, 255])
plt.ylim([0, 239])
plt.show()

img_original = cv.imread('images/margot.jpg', cv.IMREAD_GRAYSCALE)
assert img_original is not None
cv.namedWindow('Original Image', cv.WINDOW_NORMAL)
cv.imshow('Original Image', img_original)
cv.waitKey(0)

img_transformed = cv.LUT(img_original, trans)
cv.namedWindow('Transformed Image', cv.WINDOW_NORMAL)
cv.imshow('Transformed Image', img_transformed)
cv.waitKey(0)
cv.destroyAllWindows()
