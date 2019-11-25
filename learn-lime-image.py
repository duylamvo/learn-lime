"""Learn lime to explain images."""
import numpy as np
import matplotlib.pyplot as plt

from keras.applications import inception_v3 as inc_net
from keras.applications.imagenet_utils import decode_predictions

from skimage import segmentation, color
from skimage.io import imread
from skimage.transform import resize as imresize
from sklearn.linear_model import Ridge
from skimage.measure import compare_ssim


def _distance(x1, x2):
    """Distance function of x1 and x2
    x1 and x2 must have same shape.
    x1: vector or matrix.
    x2: vector or matrix

    Return: a scalar value.
    """

    assert x1.shape == x2.shape
    _multichannel = True if len(x1.shape) == 3 else False
    _similarity = compare_ssim(x1, x2, multichannel=_multichannel)
    return _similarity


def _pi(x, z):
    """Return weight of instance z to x.
    x: instance x
    z: generated from z' which has same presentation to x.

    return: Pi of z to x ~ weight of z to x"""
    # because the function of distance is already return similarity of z, hence
    return _distance(x, z)


def _predict(x, fn):
    """Predict output of input x given model fn

    Arguments:
        x {[type]} -- image x
        fn {function} -- predict function
    """
    shape = (1, x.shape[0], x.shape[1], x.shape[2])
    img = x.reshape(shape)
    pred = fn(img)

    # Get top label
    _, label, prob = decode_predictions(pred)[0][0]

    return (label, prob)


def _mask_to_3d(mask2d):
    shape = mask2d.shape
    mask3d = mask2d.reshape(shape[0], shape[1], 1)
    return mask3d


def segment(x, n_features):
    """Segment instance x."""
    img_segments = segmentation.slic(x, n_features, compactness=20)
    return img_segments


def superpixel(x, x_segments):
    """Convert instance x to superpixels."""
    img_superpixels = color.label2rgb(x_segments, x, kind="avg")
    return img_superpixels


def neighboors(img_original,
               n_features=10,
               n_features_on=None,
               sample_size=100,
               predict_fn=_predict,
               *args):

    # Segmentation X -> X'
    img_segments = segment(img_original, n_features)

    # Vector of X':
    #   here is not a binary vector but a vector of unique feature in x
    unique_segments = np.unique(img_segments)

    # Accept only if segments as number -> to support mul operations
    assert np.issubdtype(np.int, unique_segments.dtype)

    # number of vectors in feature vectors will be on/off ~ present/absent
    n_features = len(unique_segments)
    if n_features_on is None:
        n_features_on = np.random.randint(0, n_features)

    prob_features_on = n_features_on / n_features
    prob_features_off = 1 - prob_features_on

    _z_comma = np.random.choice(a=[True, False],
                                size=(sample_size, n_features),
                                p=[prob_features_on, prob_features_off])

    samples_set = []
    # Create data set neighboors of X based on X'
    for i in range(sample_size):
        # z' is a binary vector of on/off features [True, False, ...]

        # Which feature currently available for sample z'
        _on_features = unique_segments[_z_comma[i]]

        # mask2d of z' ~ corresponding to pixels of instance x
        # remember, mask2d is for segments, mask3d is for RGB image
        mask2d = np.isin(img_segments, _on_features).astype(int)
        mask3d = _mask_to_3d(mask2d)

        # Get z as same original presentation of instance x
        z_original = img_original * mask3d

        # get f(z)
        label, prob_z = predict_fn(z_original, *args)
        # z_superpixels = img_superpixels * mask3d

        assert z_original.shape == img_original.shape

        # what we need for the model is only f(z) - g(z')
        # g(z') having input of X.
        sample = (_z_comma[i], prob_z, mask2d)

        # list of tupples will be return (size * tuples)
        samples_set.append(sample)
    return samples_set


def viz_explains(image_dict, idx=0, title=None):
    fig = plt.figure()
    fig.suptitle(title)
    idx = 330
    for k, v in image_dict.items():
        idx += 1
        ax = plt.subplot(idx)
        ax.set_title(k)
        ax.imshow(v)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    plt.show()


def main():
    N_FEATURES = 20
    SAMPLE_SIZE = 100
    TOP_K = 5

    img = imread("data/cat-2083492_1280.jpg")
    img_original = imresize(img, (299, 299))
    img_segments = segment(img_original, N_FEATURES)
    img_superpixels = superpixel(img_original, img_segments)

    print(f"""
        Image: {img_original.shape}
        Segments: {img_segments.shape}
        Superpixels: {img_superpixels.shape}
    """)

    # Test with Inception Net
    inet_model = inc_net.InceptionV3()
    fn = inet_model.predict
    samples_set = neighboors(img_original,
                             N_FEATURES,
                             int(N_FEATURES/2),
                             SAMPLE_SIZE,
                             _predict, fn)

    # Get one sample to visualize z_comma and z_original
    rand_sample = np.random.randint(0, 100)
    sample = samples_set[rand_sample]
    z_comma, prob_z, z_mask2d = sample

    # convert mask of segments to mask of image (RGB)
    z_mask3d = _mask_to_3d(z_mask2d)

    # get image of z in original, segments, and superpixel
    z_segments = img_segments * z_mask2d
    z_original = img_original * z_mask3d
    z_superpixels = img_superpixels * z_mask3d

    # Define a local explain mode g(z') = w * z'
    linear_xai = Ridge()

    # # get prob_y as y_hat. z_comma as input
    input_z_comma = [i[0] for i in samples_set]
    target = [i[1] for i in samples_set]

    linear_xai.fit(input_z_comma, target)
    plt.plot(linear_xai.coef_, alpha=0.7,
             linestyle='none', marker='*', markersize=5)
    plt.show()

    sorted_idx = np.argsort(linear_xai.coef_)

    top_k = TOP_K
    if top_k is None:
        top_k = len(sorted_idx)

    unique_segments = np.unique(img_segments)
    top_k_idx = sorted_idx[-top_k:]
    _segments_to_show = unique_segments[top_k_idx]
    xai_mask2d = np.isin(img_segments, _segments_to_show).astype(int)
    xai_mask3d = _mask_to_3d(xai_mask2d)
    xai_segments = img_segments * xai_mask2d
    xai_original = img_original * xai_mask3d
    xai_superpixels = img_superpixels * xai_mask3d

    # visualize how lime convert from x -> x'-> z'-> z -> explain
    viz_images = {
        "x_original": img_original,
        "x_segments": img_segments,
        "x_superpixel": img_superpixels,
        "z_original": z_original,
        "z_segments": z_segments,
        "z_superpixel": z_superpixels,
        "xai_original": xai_original,
        "xai_segments": xai_segments,
        "xai_superpixel": xai_superpixels,

    }
    viz_explains(viz_images,
                 idx=330,
                 title="Tiger Cat, p=0.45, model=InceptionNet")


def viz_lime_theory():
    from graphviz import Digraph

    dot = Digraph(comment='LIME')
    dot.node('X', 'Instance X')
    dot.node('X_segment', 'X - Segmentation')
    dot.node('X_comma', "X'")
    dot.node('Z_comma', "Sampling Neighbors Z'")
    dot.node('Z', "Z")
    dot.node('F', "f(Z)")
    dot.node('P', "π(x,z)")
    dot.node('G', "g(z') = W*Z'",
             color="turquoise", style="filled")
    dot.node('L', "Loss of f(z) ~ g(z') with weight π")
    dot.node('O', "Optimizer: argmin { loss }")
    dot.node('R', "W^ ~ Weights of features",
             color="turquoise", style="filled")

    dot.edge('X', 'X_segment')
    dot.edge('X_segment', 'X_comma')
    dot.edge('X_comma', 'Z_comma')
    dot.edge('Z_comma', 'Z')
    dot.edge('Z', 'F')
    dot.edge("Z", "P", "Similarity of x to z")
    dot.edge('Z_comma', 'G')
    dot.edges(["FL", "GL", "PL", "LO", "OR"])

    c = Digraph(name="sampling_z",
                node_attr={"shape": "box", "style": "dashed"})
    c.node('S', "Sampling Neighbors Z'")
    c.node('A', "Z'")
    c.node('B', "Z'")
    c.node('C', "Z'")

    c.edges(["SB", "SC", "SA"])
    dot.subgraph(c)

    return dot.render("test-ouput/round-table.gv", view=True)


if __name__ == '__main__':
    main()
    # viz_lime_theory()
    pass
