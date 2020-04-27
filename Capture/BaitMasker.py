import numpy as np
from operator import itemgetter
import pandas as pd


def check_if_sorted_in_ascending_order(arr):

    # each element must be smaller than the previous one
    return np.all(arr[1:] >= arr[0:-1])


def get_chr_names_to_indices_dict(bins_chrs):

    cont_mask = np.concatenate((np.asarray([False]), bins_chrs[1:] == bins_chrs[:-1]), axis=0)
    unique_chr_names = bins_chrs[np.logical_not(cont_mask)]
    chr_indices = np.arange(len(unique_chr_names))

    return dict(zip(unique_chr_names, chr_indices))


def get_location_hash(chrs_names_to_indices_dict, chrs_names, starts, max_num_of_all=None, check_sorting=False):

    if max_num_of_all is None:
        max_num_of_all = np.amax(starts)

    # using chr index instead of name as they are not sorted in alphabetic order
    chr_indexer = np.vectorize(lambda x: chrs_names_to_indices_dict.get(x, len(chrs_names_to_indices_dict)))
    chrs_indices = chr_indexer(chrs_names)

    # finding the min power of 10 required to capture chr indices!
    p = 1
    while np.power(10, p) < len(chrs_names_to_indices_dict) + 1:
        p += 1

    n2s = lambda x: ('%0' + str(p) + 'd') % x
    v_n2s = np.vectorize(n2s)
    chrs_str_indices = v_n2s(chrs_indices)

    # as _ is smaller than all numbers
    chrs_str_indices = np.core.defchararray.add(chrs_str_indices.astype(str), '_')

    # finding the min power of 10 required to capture max!
    p = 1
    while np.power(10, p) < max_num_of_all:
        p += 1

    formatter_str = '%0' + str(p) + 'd'

    n2s = lambda x: formatter_str % x
    v_n2s = np.vectorize(n2s)
    starts_strs = v_n2s(starts)

    hash_codes = np.core.defchararray.add(chrs_str_indices, starts_strs)

    if check_sorting and not check_if_sorted_in_ascending_order(hash_codes):
        raise Exception('Problem: Hash codes are not sorted!!')

    return hash_codes


def calc_overlap(q1_chr, q1_start, q1_end, q2_chr, q2_start, q2_end):

    if q1_chr != q2_chr:
        return 0

    return max(0, min(q1_end, q2_end) - max(q1_start, q2_start))


def read_baits(baits_file_dir):

    # baits are in even indices
    baits_data = pd.read_csv(baits_file_dir, header=None, sep='\t').values

    return baits_data[:, 0].astype(str), baits_data[:, 1].astype(int), baits_data[:, 2].astype(int)


def calculate_bait_coverage(bins_chrs, bins_starts, bins_ends, baits_file_dir, bait_overhangs):

    chrs_dict = get_chr_names_to_indices_dict(bins_chrs)

    baits_chrs, baits_starts, baits_ends = read_baits(baits_file_dir)

    max_bp = max(np.amax(baits_ends), bins_ends[-1]) + 1

    bins_starts_hashes = get_location_hash(chrs_dict, bins_chrs, bins_starts, max_bp, check_sorting=True)
    bins_ends_hashes = get_location_hash(chrs_dict, bins_chrs, bins_ends, max_bp, check_sorting=True)

    # adding overhangs
    if bait_overhangs > 0:
        baits_starts = np.maximum(0, baits_starts - bait_overhangs)
        baits_ends = baits_ends + bait_overhangs

    baits_starts_hashes = get_location_hash(chrs_dict, baits_chrs, baits_starts, max_bp)
    baits_ends_hashes = get_location_hash(chrs_dict, baits_chrs, baits_ends, max_bp)

    bait_start_si = np.searchsorted(bins_starts_hashes, baits_starts_hashes, side='right') - 1
    bait_end_si = np.searchsorted(bins_ends_hashes, baits_ends_hashes, side='right')

    bins_bait_content = np.full((len(bins_chrs) + 2,), 0, dtype=np.int32)
    bins_ids = np.arange(1, len(bins_chrs) + 2)

    v_cal_overlap = np.vectorize(calc_overlap)

    bait_start_overlap = v_cal_overlap(bins_chrs[bait_start_si], bins_starts[bait_start_si], bins_ends[bait_start_si],
                                       baits_chrs, baits_starts, baits_ends)
    np.add.at(bins_bait_content, bins_ids[bait_start_si], bait_start_overlap)

    baits_with_different_end_mask = np.not_equal(bait_start_si, bait_end_si)
    bait_end_overlap = v_cal_overlap(
        bins_chrs[bait_end_si[baits_with_different_end_mask]],
        bins_starts[bait_end_si[baits_with_different_end_mask]],
        bins_ends[bait_end_si[baits_with_different_end_mask]],
        baits_chrs[baits_with_different_end_mask],
        baits_starts[baits_with_different_end_mask],
        baits_ends[baits_with_different_end_mask])
    np.add.at(bins_bait_content, bins_ids[bait_end_si[baits_with_different_end_mask]],
              bait_end_overlap)

    # baits with multiple bins coverage
    multiple_bins_coverage_ids = np.arange(len(baits_chrs))[np.greater(bait_end_si, bait_start_si + 1)].tolist()
    for i in multiple_bins_coverage_ids:
        bins_bait_content[bins_ids[bait_start_si[i]] + 1:bins_ids[bait_end_si[i]]] += \
            (bins_ends[bins_ids[bait_start_si[i]] + 1:bins_ids[bait_end_si[i]]] -
             bins_starts[bins_ids[bait_start_si[i]] + 1:bins_ids[bait_end_si[i]]])

    return bins_bait_content
