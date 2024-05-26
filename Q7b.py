import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def resize_image(image, target_height, target_width, interpolation=cv.INTER_LINEAR):
    return cv.resize(image, (target_width, target_height), interpolation=interpolation)

def normalized_ssd(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions for SSD calculation.")
    diff = (image1 - image2) ** 2
    ssd = np.sum(diff)
    normalized_ssd = ssd / float(image1.shape[0] * image1.shape[1])
    return normalized_ssd

original_large = cv.imread('images/a1q5images/im04.jpg', cv.IMREAD_GRAYSCALE)
zoomed_out_small = cv.imread('images/a1q5images/im04small.jpg', cv.IMREAD_GRAYSCALE)

assert original_large is not None and zoomed_out_small is not None

scale_factor = 4

target_height = original_large.shape[0]
target_width = original_large.shape[1]

nearest_neighbor_scaled = resize_image(zoomed_out_small, target_height, target_width, interpolation=cv.INTER_NEAREST)
bilinear_scaled = resize_image(zoomed_out_small, target_height, target_width, interpolation=cv.INTER_LINEAR)

ssd_nearest_neighbor = normalized_ssd(original_large, nearest_neighbor_scaled)
ssd_bilinear = normalized_ssd(original_large, bilinear_scaled)

print(f"Normalized SSD (Nearest Neighbor): {ssd_nearest_neighbor}")
print(f"Normalized SSD (Bilinear): {ssd_bilinear}")

plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1),plt.imshow(original_large, cmap='gray'),plt.title('Original Large Image')
plt.subplot(1, 4, 2),plt.imshow(zoomed_out_small, cmap='gray'),plt.title('Zoomed Out Small Image')
plt.subplot(1, 4, 3),plt.imshow(nearest_neighbor_scaled, cmap='gray'),plt.title('Nearest Neighbor Scaled')
plt.subplot(1, 4, 4),plt.imshow(bilinear_scaled, cmap='gray'),plt.title('Bilinear Scaled')
plt.show()
