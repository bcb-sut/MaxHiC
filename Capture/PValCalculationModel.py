import tensorflow as tf
import numpy as np
from scipy.stats import nbinom


# returns log of p-value
def tf_nbinom_logsf(x, mu, dispersion):
    
    p = mu / (dispersion + mu)
    nbinom_objs = tf.contrib.distributions.NegativeBinomial(dispersion, probs=p)
    return nbinom_objs.log_survival_function(tf.cast(x, tf.float64))


def np_nbinom_logsf(x, mu, dispersion):

    p = dispersion / (dispersion + mu)
    return nbinom.logsf(x, dispersion, p)
