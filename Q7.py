import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def nearest_neighbor_interpolation(image, scale):
    height, width = image.shape[:2]
    new_height, new_width = int(height * scale), int(width * scale)
    resized_image = np.zeros((new_height, new_width), dtype=image.dtype)

    for i in range(new_height):
        for j in range(new_width):
            x = min(int(i / scale), height - 1)
            y = min(int(j / scale), width - 1)
            resized_image[i, j] = image[x, y]

    return resized_image

def bilinear_interpolation(image, scale):
    height, width = image.shape[:2]
    new_height, new_width = int(height * scale), int(width * scale)
    resized_image = np.zeros((new_height, new_width), dtype=image.dtype)

    for i in range(new_height):
        for j in range(new_width):
            x = i / scale
            y = j / scale

            x1 = int(np.floor(x))
            x2 = min(x1 + 1, height - 1)
            y1 = int(np.floor(y))
            y2 = min(y1 + 1, width - 1)

            R1 = (x2 - x) * image[x1, y1] + (x - x1) * image[x2, y1]
            R2 = (x2 - x) * image[x1, y2] + (x - x1) * image[x2, y2]
            P = (y2 - y) * R1 + (y - y1) * R2

            resized_image[i, j] = P

    return resized_image

def normalized_ssd(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions for SSD calculation.")
    diff = (image1 - image2) ** 2
    ssd = np.sum(diff)
    normalized_ssd = ssd / float(image1.shape[0] * image1.shape[1])
    return normalized_ssd

# Load the images
original_large = cv.imread('images/a1q5images/im01.png', cv.IMREAD_GRAYSCALE)
zoomed_out_small = cv.imread('images/a1q5images/im01small.png', cv.IMREAD_GRAYSCALE)

# Define the scale factor
scale_factor = 4

# Scale up the zoomed-out images
nearest_neighbor_scaled = nearest_neighbor_interpolation(zoomed_out_small, scale_factor)
bilinear_scaled = bilinear_interpolation(zoomed_out_small, scale_factor)

# Compute the normalized SSD
ssd_nearest_neighbor = normalized_ssd(original_large, nearest_neighbor_scaled)
ssd_bilinear = normalized_ssd(original_large, bilinear_scaled)

print(f"Normalized SSD (Nearest Neighbor): {ssd_nearest_neighbor}")
print(f"Normalized SSD (Bilinear): {ssd_bilinear}")

# Display the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(original_large, cmap='gray')
plt.title('Original Large Image')

plt.subplot(1, 3, 2)
plt.imshow(nearest_neighbor_scaled, cmap='gray')
plt.title('Nearest Neighbor Scaled')

plt.subplot(1, 3, 3)
plt.imshow(bilinear_scaled, cmap='gray')
plt.title('Bilinear Scaled')

plt.show()
