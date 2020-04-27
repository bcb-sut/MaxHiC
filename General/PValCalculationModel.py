import tensorflow as tf
import numpy as np
from scipy.stats import nbinom


# returns logs of p-value
def nbinom_logsf(x, mu, dispersion):

    # for zero mus this will be assigned to -100
    res = -100.0 * np.ones(mu.shape, dtype=np.float64)

    p = dispersion / (dispersion + mu)

    res[mu > 0.0] = nbinom.logsf(x[mu > 0.0], dispersion, p[mu > 0.0])

    return res

