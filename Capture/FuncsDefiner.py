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


def tf_soft_max(x1, x2, return_shares=False):

    max_val = tf.maximum(x1, x2)
    min_val = tf.minimum(x1, x2)

    max_val_factor = 1.0 / (1.0 + tf.exp(tf.constant(100.0, dtype=tf.float64) * (min_val - max_val)))
    min_val_factor = 1 - max_val_factor

    final_val = max_val_factor * max_val + min_val_factor * min_val

    if not return_shares:
        return final_val
    else:
        sh1 = tf.where(tf.greater_equal(x1, x2), max_val_factor, min_val_factor)
        sh2 = tf.where(tf.greater_equal(x1, x2), min_val_factor, max_val_factor)
        return final_val, sh1, sh2


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
    w_avg = w_v_sum / tf.maximum(w_f_sum, tf.constant(1.0, dtype=tf.float64))

    return w_avg


def print_val(x, title):

    x = tf.cast(x, tf.float64)
    new_x = tf.Print(x, [x], title + ' value')
    return new_x


def check_nan(x, title):

    x = tf.cast(x, tf.float64)

    new_x = tf.cond(
        tf.reduce_any(tf.is_nan(x)),
        lambda: tf.Print(x, [x], title + ' is nan'),
        lambda: x
    )
    return new_x


def check_inf(x, title):

    x = tf.cast(x, tf.float64)

    new_x = tf.cond(
        tf.reduce_any(tf.is_inf(x)),
        lambda: tf.Print(x, [x], title + ' is inf'),
        lambda: x
    )
    return new_x


def tf_transformed_vis(vis_transformer_params, v):
    dist_s_power = vis_transformer_params[0]
    return tf_soft_max(tf.pow(v, dist_s_power), vis_transformer_params[1])


def tf_cis_ints_expected_vals(ints, vis, cis_f_params,
                              b1_vis_transformer_params, b2_vis_transformer_params):

    st_i = tf.gather(vis, tf.squeeze(ints[:, 0]))
    st_j = tf.gather(vis, tf.squeeze(ints[:, 1]))

    dij = tf.squeeze(tf.abs(ints[:, 0] - ints[:, 1]))

    vi = tf_transformed_vis(b1_vis_transformer_params, st_i)
    vj = tf_transformed_vis(b2_vis_transformer_params, st_j)

    f_dij = tf_cis_dist_func(cis_f_params, dij)

    mu_ij = vi * vj * f_dij

    return mu_ij


def tf_trans_ints_expected_vals(ints, vis, trans_f_param,
                                b1_vis_transformer_params, b2_vis_transformer_params):

    st_i = tf.gather(vis, ints[:, 0])
    st_j = tf.gather(vis, ints[:, 1])

    vi = tf_transformed_vis(b1_vis_transformer_params, st_i)
    vj = tf_transformed_vis(b2_vis_transformer_params, st_j)

    f_dij = tf.exp(trans_f_param)

    mu_ij = vi * vj * f_dij

    return mu_ij


def np_cis_var_func(cis_f_params, d):
    ln_d = np.log(d.astype(float))
    return np.exp(np.poly1d(cis_f_params[0:4])(ln_d))


def np_cis_dist_func(cis_f_params, d):
    return np_soft_max(np_cis_var_func(cis_f_params, d), np.exp(cis_f_params[4]))


def np_soft_max(x1, x2):

    max_val = np.maximum(x1, x2)
    min_val = np.minimum(x1, x2)

    max_val_factor = 1.0 / (1.0 + np.exp(100.0 * (min_val - max_val)))
    min_val_factor = 1 - max_val_factor

    final_val = max_val_factor * max_val + min_val_factor * min_val

    return final_val


def np_transformed_vis(vis_transformer_params, v):
    dist_s_power = vis_transformer_params[0]
    return np_soft_max(np.power(v, dist_s_power), vis_transformer_params[1])


def np_cis_ints_expected_vals(ints, vis, cis_f_params,
                              b1_vis_transformer_params, b2_vis_transformer_params):

    st_i = vis[ints[:, 0]]
    st_j = vis[ints[:, 1]]

    dij = np.abs(ints[:, 0] - ints[:, 1])

    vi = np_transformed_vis(b1_vis_transformer_params, st_i)
    vj = np_transformed_vis(b2_vis_transformer_params, st_j)

    f_dij = np_cis_dist_func(cis_f_params, dij)

    mu_ij = vi * vj * f_dij

    return mu_ij


def np_trans_ints_expected_vals(ints, vis, trans_f_param,
                                b1_vis_transformer_params, b2_vis_transformer_params):

    st_i = vis[ints[:, 0]]
    st_j = vis[ints[:, 1]]

    vi = np_transformed_vis(b1_vis_transformer_params, st_i)
    vj = np_transformed_vis(b2_vis_transformer_params, st_j)

    f_dij = np.exp(trans_f_param)

    mu_ij = vi * vj * f_dij

    return mu_ij


def tf_ints_float_dist(ints):
    return tf.cast(tf.squeeze(tf.abs(ints[:, 0] - ints[:, 1])), tf.float64)


def tf_argsort_based_on_two_cols(v):

    c2_log = tf.log(1 + v[:, 1])
    max_c2 = tf.reduce_max(c2_log) + 1

    us_c1_u, _ = tf.unique(v[:, 0])
    c1_u = tf.sort(us_c1_u)
    c1_inds = tf.searchsorted(c1_u, v[:, 0], side='left')

    eq_num = tf.cast(c1_inds, tf.float64) * max_c2 + c2_log

    return tf.argsort(eq_num)


def get_b_type(int_type, bi):

    parts = int_type.split('_')
    the_type = int_type + '_' + parts[1][bi]
    return the_type


def part_assign(main_var, indices, new_vals):

    indices_mask = tf.cast(tf.sparse_to_dense(indices, tf.shape(main_var), np.float64(1.0), np.float64(0.0)), tf.float64)
    updated_val = tf.cast(tf.sparse_to_dense(indices, tf.shape(main_var), new_vals, np.float64(0.0)), tf.float64)

    new_complete_val = (1 - indices_mask) * main_var + updated_val

    return tf.assign(main_var, new_complete_val)


def smax(arr):
    if np.size(arr) == 0:
        return -1 * float('inf')
    else:
        return np.amax(arr)
