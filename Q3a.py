import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def intensity_transformation(v, value_a, sigma):
    return np.minimum(v + value_a * 128 * np.exp(-(v - 128) ** 2 / (2 * sigma ** 2)), 255)

sigma = 70
a = np.arange(0, 1, 0.1)

transformation_results = []

for value_a in a:
    v = np.arange(0, 256)
    transformed_v = intensity_transformation(v, value_a, sigma)
    transformation_results.append(transformed_v)

plt.figure(figsize=(8, 6))
for i, value_a in enumerate(a):
    plt.plot(range(256), transformation_results[i], label=f'a={value_a}')

plt.title('Intensity Transformation')
plt.xlabel('Original Intensity')
plt.ylabel('Transformed Intensity')
plt.legend()
plt.grid(True)
plt.show()
