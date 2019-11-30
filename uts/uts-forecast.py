import seglearn as sl
import matplotlib.pyplot as plt
from random import seed, random, randrange
import numpy as np

# Random walk
# y(t) = B0 + X(t-1) + e(t)
# t=0
steps = 1000
np.random.seed(1)
random_walk = list()
random_walk.append(np.random.randint(-1, 2))
for i in range(1, steps):
    # B0 + e(t)
    movement = np.random.randint(-1, 2)
    y = random_walk[i-1] + movement
    random_walk.append(y)
plt.plot(random_walk)
plt.show()


# Given an TS
x_original = np.array(random_walk)

# create segments
width = 100
overlap_rate = 0.0
segmenter = sl.transform.SegmentX(width, overlap=overlap_rate)
segmenter.fit([x_original])

# If overlap_rate r = 0, then n_segments = [len(series) / (width * (1-r))] - 1
# If r = 0 ~ n_segments = len(series) / width
x_segments, _, _ = segmenter.transform([x_original])


print(x_original.shape, x_segments.shape)

# If plot all segments, it will become multiple time series with same width
# However, it is hard to see, and also because the index is overlap

# To plot multiple time series with different index
# (1) create an empty length with nan values at begining
# (2) continuously plot the value

plt.subplot(311)
plt.plot(x_original)
plt.subplot(312)
plt.plot(x_segments)

plt.subplot(313)
z = np.empty(len(x_original), dtype=np.float)
z.fill(np.nan)

_len = 0
_width = int(width * (1 - overlap_rate))
n_features = x_segments.shape[0]
for i in range(0, n_features):
    s = np.empty(_len + width)
    s.fill(np.nan)

    _len += _width

    s[-width:] = x_segments[i]
    plt.plot(s)

plt.show()


# z_segments -> z' -> z
mask = np.random.choice([True, False], size=n_features)
z_comma = np.random.choice([True, False], size=n_features)
z_segments = x_segments * mask.reshape(x_segments.shape[0], 1)
z_original = z_segments.ravel()

plt.subplot(311)
plt.plot(z_original)
plt.subplot(312)
plt.plot(z_segments)

_len = 0
_width = int(width * (1 - overlap_rate))
n_features = z_segments.shape[0]
for i in range(0, n_features):
    s = np.empty(_len + width)
    s.fill(np.nan)

    _len += _width

    s[-width:] = z_segments[i]
    plt.plot(s)

plt.show()
