import numpy as np
from skimage import segmentation, color
from skimage.segmentation import mark_boundaries
from skimage.io import imread
from skimage.transform import resize as imresize
import matplotlib.pyplot as plt

from keras.applications import inception_v3 as inc_net
from keras.models import load_model


N_FEATURES = 10
samples_size = 100

# Segmentation X -> X'
img = imread("data/cat-2083492_1280.jpg")
img_original = imresize(img, (299, 299))
img_segments = segmentation.slic(img_original, N_FEATURES, compactness=20)
img_superpixels = color.label2rgb(img_segments, img_original, kind="avg")
unique_segments = np.unique(img_segments)

print(f"""
    Image: {img_original.shape}
    Segments: {img_segments.shape}
    Superpixels: {img_superpixels.shape}
""")

# Accept only if segments as number -> to support mul operations
assert np.issubdtype(np.int, unique_segments.dtype)
shape_org = img_original.shape

# number of vectors in feature vectors will be on/off ~ present/absent
n_features = len(unique_segments)
n_features_on = np.random.randint(0, n_features)
n_features_off = n_features - n_features_on

prob_features_on = n_features_on / n_features
prob_features_off = 1 - prob_features_on
prob_on_off = [prob_features_on, prob_features_off]

_z_comma = np.random.choice([True, False], size=(
    samples_size, n_features), p=prob_on_off)

samples_set = []
for i in range(samples_size):
    _on_features = unique_segments[_z_comma[i]]

    mask = np.isin(img_segments, _on_features).astype(int)
    mask_3d = mask.reshape(shape_org[0], shape_org[1], 1)

    z_original = img_original * mask_3d
    z_segments = img_segments * mask
    z_superpixels = img_superpixels * mask_3d

    assert z_original.shape == img_original.shape
    assert z_segments.shape == img_segments.shape
    assert z_superpixels.shape == img_superpixels.shape
    sample = (z_original, z_segments, z_superpixels)

    # list of tupples will be return (size * tuples)
    samples_set.append(sample)

rand_sample = np.random.randint(0, samples_size-1)
z_original, z_segments, z_superpixels = samples_set[rand_sample]

plt.subplot(231)
plt.imshow(img_original)

plt.subplot(232)
plt.imshow(img_segments)

plt.subplot(233)
plt.imshow(img_superpixels)

plt.subplot(234)
plt.imshow(z_original)

plt.subplot(235)
plt.imshow(z_segments)

plt.subplot(236)
plt.imshow(z_superpixels)

plt.show()


def _distance(x1, x2, kernel):
    pass

def weight(x1, x2, fn=None, *args):
    """Calculate weight of x1 and x2 based on distance function

    Arguments:
        x1 -- a vector of x1
        x2 -- a vector of x2
        distance_fn -- distance function. In default will be L2 distance function (pearson)
        *args -- arguments of distance function
    """
    pass


# def _predict(x, fn):
#     """Predict output of input x given model fn

#     Arguments:
#         x {[type]} -- [description]
#         fn {function} -- [description]
#     """
#     pass


# Test by download pre-train weight (.h5)
# https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20Image%20Classification%20Keras.ipynb
from keras.applications.imagenet_utils import decode_predictions
inet_model = inc_net.InceptionV3()


pred = inet_model.predict(img.reshape(1, 299, 299, 3))
print(decode_predictions(pred)[0])


