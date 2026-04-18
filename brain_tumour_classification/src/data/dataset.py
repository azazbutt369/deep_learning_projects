# src/data/dataset.py

import numpy as np

from .loader import load_images
from .preprocessing import shuffle_data, split_data, encode_labels


def prepare_dataset(data_dir, labels, image_size=150):
    """
    Full pipeline:
    - Loading images
    - Shuffle
    - Splitting
    - Encoding of labels
    """

    # load
    X, Y = load_images(data_dir, labels, image_size)

    # convert to numpy arrays
    X = np.array(X)
    Y = np.array(Y)

    # shuffle
    X, Y = shuffle_data(X, Y)

    # split
    X_train, X_test, Y_train, Y_test = split_data(X, Y)

    # encode labels
    Y_train = encode_labels(Y_train, labels)
    Y_test = encode_labels(Y_test, labels)

    return X_train, X_test, Y_train, Y_test
