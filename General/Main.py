import sys
from time import time
import numpy as np
from Auxiliary import read_raw_data
from ModelRunner import run_ml
from Saver import save_results
import tensorflow as tf


def general_main(base_dir, save_dir,
                 p_val_limit, cores_num, remove_sig_from_vis, rounds_num, min_dist, max_dist, min_rc,
                 device, silent_mode, full_output_mode):

    s_time = time()

    run_code, chrs_names, chrs_bin_ranges, bin_size, cis_interactions, trans_interactions, bins_file_dir = read_raw_data(base_dir)
    cis_interactions = np.concatenate(tuple(cis_interactions), axis=0)

    if run_code != 0:
        sys.exit(-1)

    # filtering self interactions
    cis_self_ints = None
    if full_output_mode:
        cis_self_ints = cis_interactions[cis_interactions[:, 0] == cis_interactions[:, 1]]
    cis_interactions = cis_interactions[cis_interactions[:, 0] != cis_interactions[:, 1]]

    min_dist = int(np.ceil(min_dist / bin_size))
    if max_dist == -1:
        max_dist = 1 + np.amax(np.abs(cis_interactions[:, 0] - cis_interactions[:, 1]))
    else:
        max_dist = int(np.floor(max_dist / bin_size))

    with tf.device(device):
        vis, cis_f_params, cis_r, cis_vis_transformer_params, trans_f_param, trans_r, trans_vis_transformer_params, \
            pre_transform_norm_factor = \
            run_ml(chrs_bin_ranges, cis_interactions, trans_interactions, p_val_limit, cores_num, remove_sig_from_vis, rounds_num,
                   min_dist, max_dist, min_rc, silent_mode)

    # filtering interactions (the ones below min limit)
    save_results(cis_interactions, trans_interactions, cis_self_ints, vis,
                 cis_f_params, cis_vis_transformer_params, cis_r, trans_f_param, trans_vis_transformer_params, trans_r,
                 pre_transform_norm_factor, save_dir, full_output_mode, bins_file_dir)

    print('Analysis ended in %.2f secs.' % (time() - s_time,))
