import os
import tensorflow as tf
import numpy as np
import random

def set_seeds(seed=123):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)