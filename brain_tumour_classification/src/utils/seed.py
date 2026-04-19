# src/utils/seed.py

import os
import random
import numpy as np
import tensorflow as tf


def set_seed(seed=42, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if deterministic:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
