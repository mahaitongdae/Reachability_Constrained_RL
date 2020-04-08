import numpy as np


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)