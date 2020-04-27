import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import sys
from time import time
import numpy as np
from Auxiliary import read_raw_data
from ModelRunner import run_ml
from Saver import save_results
from BaitMasker import calculate_bait_coverage


def capture_main(base_dir, save_dir,
                 p_val_limit, cores_num, remove_sig_from_vis, rounds_num, min_dist, max_dist, min_rc,
                 device, silent_mode, full_output_mode, baits_dir, bait_ratio_lim, bait_len_lim, bait_overhangs):

    s_time = time()

    run_code, chrs_names, chrs_bin_ranges, bin_size, cis_interactions, trans_interactions,\
        bins_chrs, bins_starts, bins_ends, bins_file_dir = read_raw_data(base_dir)
    cis_interactions = np.concatenate(tuple(cis_interactions), axis=0)

    if run_code != 0:
        sys.exit(-1)

    min_dist = int(np.ceil(min_dist / bin_size))
    if max_dist == -1:
        max_dist = 1 + np.amax(np.abs(cis_interactions[:, 0] - cis_interactions[:, 1]))
    else:
        max_dist = int(np.floor(max_dist / bin_size))

    # filtering self interactions
    cis_self_ints = None
    if full_output_mode:
        cis_self_ints = cis_interactions[cis_interactions[:, 0] == cis_interactions[:, 1]]
    cis_interactions = cis_interactions[cis_interactions[:, 0] != cis_interactions[:, 1]]

    bins_bait_coverage = calculate_bait_coverage(bins_chrs, bins_starts, bins_ends, baits_dir, bait_overhangs)
    bins_bait_mask = np.greater_equal(bins_bait_coverage, max(bait_len_lim, bin_size * bait_ratio_lim))

    if not silent_mode:
        print('*********')
        print('The number of bins overlapping with baits: ', np.count_nonzero(bins_bait_coverage))
        print('The number of bins assumed as bait: ', np.sum(bins_bait_mask.astype(np.int32)))

    with tf.device(device):
        model_info_dict = run_ml(chrs_bin_ranges, cis_interactions, trans_interactions, p_val_limit, cores_num,
                                 remove_sig_from_vis, rounds_num, min_dist, max_dist, min_rc, silent_mode, bins_bait_mask)

    # filtering interactions (the ones below min limit)
    save_results(cis_interactions, trans_interactions, cis_self_ints, model_info_dict, save_dir, full_output_mode, bins_bait_mask, bins_file_dir)

    print('Analysis ended in %.2f secs.' % (time() - s_time,))
