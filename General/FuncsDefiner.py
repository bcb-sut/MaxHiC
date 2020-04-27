import tensorflow as tf
import numpy as np


def tf_cis_var_func(cis_f_params, d):

    ln_d = tf.log(tf.cast(d, tf.float64))
    return tf.exp(
        cis_f_params[0] * tf.pow(ln_d, 3) +
        cis_f_params[1] * tf.pow(ln_d, 2) +
        cis_f_params[2] * ln_d +
        cis_f_params[3]
    )


def tf_cis_dist_func(cis_f_params, d):
    return tf_soft_max(tf_cis_var_func(cis_f_params, d), tf.exp(cis_f_params[4]))


def np_cis_var_func(cis_f_params, d):
    ln_d = np.log(d.astype(float))
    return np.exp(np.poly1d(cis_f_params[0:4])(ln_d))


def np_cis_dist_func(cis_f_params, d):
    return np_soft_max(np_cis_var_func(cis_f_params, d), np.exp(cis_f_params[4]))


def tf_soft_max(x1, x2, return_shares=False):

    x1f = tf.exp(tf.minimum(tf.constant(10.0, dtype=tf.float64) * x1, tf.constant(200.0, dtype=tf.float64)))
    x2f = tf.exp(tf.minimum(tf.constant(10.0, dtype=tf.float64) * x2, tf.constant(200.0, dtype=tf.float64)))
    sum_f = x1f + x2f

    sh_1 = (x1f / sum_f)
    sh_2 = (x2f / sum_f)

    if return_shares:
        return sh_1 * x1 + sh_2 * x2, sh_1, sh_2
    else:
        return sh_1 * x1 + sh_2 * x2


def np_soft_max(x1, x2):

    x1f = np.exp(np.minimum(10.0 * x1, 200.0))
    x2f = np.exp(np.minimum(10.0 * x2, 200.0))
    sum_f = x1f + x2f

    return (x1f / sum_f) * x1 + (x2f / sum_f) * x2


def tf_transformed_vis(vis_transformer_params, v):

    dist_s_power = vis_transformer_params[0]
    return tf_soft_max(tf.pow(v, dist_s_power), vis_transformer_params[1])


def np_transformed_vis(vis_transformer_params, v):

    dist_s_power = vis_transformer_params[0]
    return np_soft_max(np.power(v, dist_s_power), vis_transformer_params[1])


def tf_unsorted_segment_avg(values, indices, the_size):

    # calculating sum and count
    vals_sum = tf.unsorted_segment_sum(tf.cast(values, tf.float64), indices, the_size)

    vals_shape = tf.shape(values)

    vals_count = tf.unsorted_segment_sum(tf.ones(tf.expand_dims(vals_shape[0], 0), dtype=tf.float64), indices, the_size)

    the_avg = vals_sum / tf.maximum(tf.constant(1, dtype=tf.float64), vals_count)
    return the_avg


def tf_weighted_average(vals, weights):

    w_v_sum = tf.reduce_sum(vals * weights)
    w_f_sum = tf.reduce_sum(weights)
    w_avg = w_v_sum / w_f_sum

    return w_avg


def check_nan(x, title):

    x = tf.cast(x, tf.float64)

    new_x = tf.cond(
        tf.reduce_any(tf.is_nan(x)),
        lambda: tf.Print(x, [x], title + ' is nan, please report!'),
        lambda: x
    )
    return new_x


def check_inf(x, title):

    x = tf.cast(x, tf.float64)

    new_x = tf.cond(
        tf.reduce_any(tf.is_inf(x)),
        lambda: tf.Print(x, [x], title + ' is inf, please report!'),
        lambda: x
    )
    return new_x


def smax(arr):
    if np.size(arr) > 0:
        return np.amax(arr)
    else:
        return -1 * float('inf')
