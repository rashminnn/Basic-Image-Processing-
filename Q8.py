import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def segment_and_blur_background(image_path):
    image = cv.imread(image_path)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    height, width = image.shape[:2]
    margin = 10
    rect = (margin, margin, width - 2 * margin, height - 2 * margin)

    image_with_rect = image_rgb.copy()
    cv.rectangle(image_with_rect, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 2)

    cv.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    foreground = image * mask2[:, :, np.newaxis]
    blurred_background = cv.GaussianBlur(image, (21, 21), 0)

    enhanced_image = blurred_background * (1 - mask2[:, :, np.newaxis]) + foreground

    plt.figure(figsize=(20, 10))

    plt.subplot(151), plt.imshow(image_rgb), plt.title('Original Image'), plt.axis('off')
    plt.subplot(152), plt.imshow(image_with_rect), plt.title('Image with Rect'), plt.axis('off')
    plt.subplot(153), plt.imshow(mask2, cmap='gray'), plt.title('Segmentation Mask'), plt.axis('off')
    plt.subplot(154), plt.imshow(cv.cvtColor(foreground, cv.COLOR_BGR2RGB)), plt.title('Foreground'), plt.axis('off')
    plt.subplot(155), plt.imshow(cv.cvtColor(enhanced_image.astype(np.uint8), cv.COLOR_BGR2RGB)), plt.title('Enhanced Image with Blurred Background'), plt.axis('off')

    plt.show()


segment_and_blur_background('images/daisy.jpg')
