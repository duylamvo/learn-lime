import seglearn as sl
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances


def viz_features(x_original, n_features=10, width=100, overlap_rate=0):
    # If plot all segments, it will become multiple time series with same width
    # However, it is hard to see, and also because the index is overlap

    # To plot multiple time series with different index
    # (1) create an empty length with nan values at begining
    # (2) continuously plot the value

    # x ---segment---> x'
    x_segments = segment(x_original, n_features, width, overlap_rate)
    print(x_original.shape, x_segments.shape)

    # visualization
    plt.subplot(311)
    plt.plot(x_original)
    plt.subplot(312)
    plt.plot(x_segments)

    plt.subplot(313)

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


def _distance(x1, x2):
    d = pairwise_distances(x1.reshape(1, -1),
                           x2.reshape(1, -1))
    return d


def _pi(x, z, gamma=0.1):
    # Apply gaussian radial basis function as kernel ~ RBF kernel
    # exp(-(||x-y||² / (2*sigma²)) ~ exp(-gamma*||x-y||²))
    #   gamma = 1/2.sigma²
    assert x.shape == z.shape

    d = _distance(x, z)
    pi = np.exp(-gamma*d)

    return np.asscalar(pi)


def _predict(x, fn=None):
    # Todo: get f(z) or model here
    return np.random.random()


def segment(x, n_features, steps=100, overlap_rate=0.):
    # Given an TS x as array
    # create segments
    segmenter = sl.transform.SegmentX(steps, overlap=overlap_rate)
    segmenter.fit([x])

    # Overlap_rate r, then n_segments = [len(series) / (width * (1-r))] - 1
    #  if r = 0 ~ n_segments = len(series) / width
    x_segments, _, _ = segmenter.transform([x])
    return x_segments


def neighbors(x, n_features=10, n_features_on=None, sample_size=100,
              predict_fn=None, *args):
    # Segmentation X -> X'
    x_segments = segment(x, n_features)

    if n_features_on is None:
        n_features_on = np.random.randint(0, n_features)

    prob_features_on = n_features_on / n_features
    prob_features_off = 1 - prob_features_on
    prob = [prob_features_on, prob_features_off]

    # Neighbors ---random-choose--> z'---convert-back---> z
    samples = np.random.choice([True, False],
                               size=(sample_size, n_features),
                               p=prob)

    samples_set = []
    for z_comma in samples:
        z_segments = x_segments * z_comma.reshape(x_segments.shape[0], 1)
        z_original = z_segments.ravel()

        # get f(z)
        f_z = None
        if predict_fn:
            f_z = predict_fn(z_original, *args)

        sample = (z_comma, z_original, f_z)
        samples_set.append(sample)

    return samples_set


def get_top_k(weights, top_k):
    # Select only top k features
    sorted_idx = np.argsort(weights)

    if top_k is None:
        top_k = len(sorted_idx)

    top_k_idx = sorted_idx[-top_k:]
    return top_k_idx


def test_main():
    N_FEATURES = 10
    SAMPLE_SIZE = 100
    TOP_K = 3
    X_STEPS = 100

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
    # plt.plot(random_walk)
    # plt.show()

    # get an instance x
    x_original = np.array(random_walk)
    x_segments = segment(x_original, N_FEATURES)

    viz_features(x_original, N_FEATURES, X_STEPS, overlap_rate=0)

    # Neighbors
    samples_set = neighbors(x_original,
                            n_features=N_FEATURES,
                            n_features_on=int(N_FEATURES/2),
                            sample_size=SAMPLE_SIZE,
                            predict_fn=_predict
                            )
    sample = samples_set[0]
    z_comma, z_original, f_z = sample

    # Define a local explain mode g(z') = w * z'
    linear_xai = Ridge()
    xai_z_commas = [i[0] for i in samples_set]
    xai_target = [i[2] for i in samples_set]

    sample_weight = [_pi(x_original, i[1], gamma=0.001) for i in samples_set]
    linear_xai.fit(xai_z_commas, xai_target, sample_weight)

    plt.plot(linear_xai.coef_, alpha=0.7,
             linestyle='none', marker='*', markersize=5)
    plt.show()

    top_k_idx = get_top_k(linear_xai.coef_, TOP_K)

    # Visualize xai explains with weights
    plt.plot(x_original)
    for i in top_k_idx:
        mask = np.empty(N_FEATURES)
        mask.fill(np.nan)
        mask[i] = 1
        xai_segments = x_segments * mask.reshape(N_FEATURES, 1)

        alpha = i / max(top_k_idx)
        print(i, alpha)
        plt.plot(xai_segments.ravel(),
                 color="red",
                 alpha=alpha)
    plt.show()


test_main()
