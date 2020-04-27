from os import listdir
import pandas as pd
import numpy as np
import threading


def read_bins_info(bins_file_dir):
    info = pd.read_csv(bins_file_dir, delimiter='\t', dtype=str, comment='#', header=None).values
    return info[:, 0], info[:, 1].astype(int), info[:, 2].astype(int)


# returns success code, chrs_names, chrs_bin_ranges, bin_size, list of cis interactions for each chromosome
def read_raw_data(base_dir):

    files_in_base_dir = listdir(base_dir)

    bins_file = None
    interactions_file = None

    for f in files_in_base_dir:
        if '.bed' in f:
            bins_file = base_dir + '/' + f
        elif '.matrix' in f:
            interactions_file = base_dir + '/' + f

    if bins_file is None:
        print('No .bed files for location of bins in the given directory')
        return -1, None, None, None, None, None, None, None

    if interactions_file is None:
        print('No .bed files for sparse interactions in the given directory')
        return -1, None, None, None, None, None, None, None

    chrs_names, chrs_bin_ranges, bin_size = get_chrs_names_bin_ranges_and_bin_size(bins_file)

    chrs_cis_ints, trans_ints = (read_separate_cis_interactions(interactions_file, chrs_bin_ranges))

    bins_chrs, bins_starts, bins_ends = read_bins_info(bins_file)

    return 0, chrs_names, chrs_bin_ranges, bin_size, chrs_cis_ints, trans_ints, bins_chrs, bins_starts, bins_ends, bins_file


def get_chrs_names_bin_ranges_and_bin_size(bins_location_dir):

    chrs_names = []
    chrs_bin_ranges = []
    bin_size = None

    last_chr_name = None
    last_chr_start_bin_index = None

    with open(bins_location_dir, 'r') as chr_bins:
        for line in chr_bins:

            if line.startswith('#') or (line == '\n'):
                continue

            line_parts = line.strip().split('\t')

            chr_name = line_parts[0]
            bin_index = int(line_parts[3])

            if (last_chr_name is None) or (last_chr_name != chr_name):

                # dumping info related to the prev chromosome
                if last_chr_name is not None:
                    chrs_names.append(last_chr_name)
                    chrs_bin_ranges.append([last_chr_start_bin_index, bin_index - 1])

                last_chr_name = chr_name
                last_chr_start_bin_index = bin_index
                bin_size = int(line_parts[2]) - int(line_parts[1])

    if last_chr_name is not None:
        chrs_names.append(last_chr_name)
        chrs_bin_ranges.append([last_chr_start_bin_index, bin_index])

    return chrs_names, chrs_bin_ranges, bin_size


def read_separate_cis_interactions(interactions_file_dir, chrs_bin_ranges):

    all_interactions = pd.read_csv(interactions_file_dir, delimiter='\t', dtype=np.int32, header=None).values
    chrs_start_bins = np.asarray([x[0] for x in chrs_bin_ranges])

    # searching each interaction's bins' chromosomes's index! and adding as column to the interactions file
    bin1_chr_id = np.searchsorted(chrs_start_bins, all_interactions[:, 0], side='right')
    bin2_chr_id = np.searchsorted(chrs_start_bins, all_interactions[:, 1], side='right')

    chrs_ints = [all_interactions[(bin1_chr_id == i + 1) & (bin2_chr_id == i + 1), :] for i in range(len(chrs_bin_ranges))]
    trans_ints = all_interactions[bin1_chr_id != bin2_chr_id, :]

    return chrs_ints, trans_ints


# returns controller
def execute_func_asynchronously(f, args):

    new_thread = threading.Thread(target=f, args=args)
    new_thread.start()
    return new_thread

