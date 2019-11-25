import numpy as np
from skimage import segmentation, color
from skimage.io import imread

import matplotlib.pyplot as plt


N_FEATURES = 500
# Segmentation X -> X'
img = imread("data/cat-2083492_1280.jpg")
img_segments = segmentation.slic(img, N_FEATURES, compactness=20)
superpixels = color.label2rgb(img_segments, img, kind="avg")

plt.subplot(131)
plt.imshow(img)

plt.subplot(132)
plt.imshow(img_segments)

plt.subplot(133)
plt.imshow(superpixels)

plt.show()

# Create a masking matrix (on/off)

mask = np.ones(img.shape)
unique_segments = np.unique(img_segments)
n_features = 50


print(f"""
    Image: {img.shape}
    Segments: {img_segments.shape}
    Superpixels: {superpixels.shape}
""")
