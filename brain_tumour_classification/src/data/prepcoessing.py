import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def shuffle_data(X, Y, seed=101):
    return shuffle(X, Y, random_state=seed)


def split_data(X, Y, test_size=0.1, seed=2020):
    return train_test_split(X, Y, test_size=test_size, random_state=seed)


def encode_labels(Y, labels):
    """
    Encodes string labels to categorical (one-hot)
    EXACTLY as notebook logic
    """
    encoded = []
    for y in Y:
        encoded.append(labels.index(y))

    encoded = np.array(encoded)
    encoded = tf.keras.utils.to_categorical(encoded)

    return encoded
