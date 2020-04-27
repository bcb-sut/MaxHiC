import pandas as pd
import numpy as np
from os import path, makedirs
from PValCalculationModel import nbinom_logsf
from Auxiliary import execute_func_asynchronously
from FuncsDefiner import np_transformed_vis, np_cis_dist_func, smax


def calculate_q_val_fdr(ints, pvals):

    num_of_tests = list(ints.shape)[0]

    # calculating q values
    sorted_p_vals = np.sort(pvals)

    # finding index of each pval in sorted list to find its rank!!!
    pvals_rank = np.searchsorted(sorted_p_vals, pvals, side='left') + 1

    adj_pvals = pvals * num_of_tests / pvals_rank

    return np.log(adj_pvals)


def save_one_info(data, save_dir, header):
    pd.DataFrame(data).to_csv(save_dir, sep='\t', header=header, index=False)


def save_results(cis_interactions, trans_interactions, cis_self_ints, vis, cis_f_params, cis_vis_transformer_params, cis_r,
                 trans_f_param, trans_vis_transformer_params, trans_r,
                 pre_transform_norm_factor,
                 save_dir, full_output_mode, bins_file_dir):
    
    bins_info = np.concatenate(
        (np.asarray([['', 0, 0]]), pd.read_csv(bins_file_dir, sep='\t', header=None).values[:, 0:3]), axis=0)

    if not path.exists(save_dir):
        makedirs(save_dir)

    params_save_dir = save_dir + '/ModelParameters'
    if not path.exists(params_save_dir):
        makedirs(params_save_dir)

        # if full output is requested,
    if full_output_mode:
        # calculating read sum of bins

        max_bin_id = max(smax(cis_interactions[:, 0:2]), smax(trans_interactions[:, 0:2]),
                         smax(cis_self_ints[:, 0:2]))
        bin_selfless_sum = np.full((max_bin_id + 1,), 0, dtype=int)
        bin_total_sum = np.full((max_bin_id + 1,), 0, dtype=int)

        np.add.at(bin_selfless_sum, cis_interactions[:, 0], cis_interactions[:, 2])
        np.add.at(bin_selfless_sum, cis_interactions[:, 1], cis_interactions[:, 2])
        np.add.at(bin_selfless_sum, trans_interactions[:, 0], trans_interactions[:, 2])
        np.add.at(bin_selfless_sum, trans_interactions[:, 1], trans_interactions[:, 2])

        bin_total_sum += bin_selfless_sum
        np.add.at(bin_total_sum, cis_self_ints[:, 0], cis_self_ints[:, 2])

    # SAVING TRANS INTERACTIONS ASYNCHRONOUSLY (AS IT TAKES MOST OF THE TIME)

    # calculating expected value
    vi_p = np_transformed_vis(trans_vis_transformer_params, vis[trans_interactions[:, 0]])
    vj_p = np_transformed_vis(trans_vis_transformer_params, vis[trans_interactions[:, 1]])

    visibilities_mul = np.multiply(vi_p, vj_p)

    exp_interactions = np.multiply(visibilities_mul, np.exp(trans_f_param))

    log_p_vals = nbinom_logsf(trans_interactions[:, 2], exp_interactions, trans_r)
    adj_q_val = calculate_q_val_fdr(trans_interactions, np.exp(log_p_vals))

    if not full_output_mode:
        interactions_info = np.column_stack((-1 * log_p_vals, -1 * adj_q_val, exp_interactions, vi_p, vj_p))
        full_ints = np.concatenate((trans_interactions, interactions_info), axis=1)

        trans_saver_handler = execute_func_asynchronously(save_one_info, (
            full_ints, save_dir + '/trans_interactions.txt', ['bin1ID', 'bin2ID', 'read_count', 'neg_ln_p_val', 'neg_ln_q_val',
                                                              'exp_read_count', 'b1_bias', 'b2_bias']))

    else:
        interactions_info = np.column_stack((
            trans_interactions[:, 0], bins_info[trans_interactions[:, 0], 0], bins_info[trans_interactions[:, 0], 1], bins_info[trans_interactions[:, 0], 2],
            trans_interactions[:, 1], bins_info[trans_interactions[:, 1], 0], bins_info[trans_interactions[:, 1], 1], bins_info[trans_interactions[:, 1], 2],
            trans_interactions[:, 2],
            exp_interactions, -1 * log_p_vals, -1 * adj_q_val, vi_p, vj_p,
                                             bin_total_sum[trans_interactions[:, 0]], bin_total_sum[trans_interactions[:, 1]],
                                             bin_selfless_sum[trans_interactions[:, 0]], bin_selfless_sum[trans_interactions[:, 1]]))

        trans_saver_handler = execute_func_asynchronously(save_one_info, (
            interactions_info, save_dir + '/trans_interactions.txt',
            ['bin1ID', 'bin1Chromosome', 'b1Start', 'b1End',
             'bin2ID', 'bin2Chromosome', 'b2Start', 'b2End',
             'read_count', 'exp_read_count', 'neg_ln_p_val', 'neg_ln_q_val', 'b1_bias', 'b2_bias',
             'b1_read_sum', 'b2_read_sum', 'b1_selfless_read_sum', 'b2_selfless_read_sum']))

    # SAVING CIS INTERACTIONS ASYNCHRONOUSLY
    dij = np.abs(cis_interactions[:, 0] - cis_interactions[:, 1])
    estimated_f_d = np_cis_dist_func(cis_f_params, dij)

    vi_p = np_transformed_vis(cis_vis_transformer_params, vis[cis_interactions[:, 0]])
    vj_p = np_transformed_vis(cis_vis_transformer_params, vis[cis_interactions[:, 1]])

    visibilities_mul = np.multiply(vi_p, vj_p)

    exp_interactions = np.multiply(visibilities_mul, estimated_f_d)

    log_p_vals = nbinom_logsf(cis_interactions[:, 2], exp_interactions, cis_r)
    adj_q_val = calculate_q_val_fdr(trans_interactions, np.exp(log_p_vals))

    if not full_output_mode:
        interactions_info = np.column_stack((-1 * log_p_vals, -1 * adj_q_val, exp_interactions, vi_p, vj_p))
        full_ints = np.concatenate((cis_interactions, interactions_info), axis=1)

        cis_saver_handler = execute_func_asynchronously(save_one_info, (full_ints, save_dir + '/cis_interactions.txt',
                                                                        ['bin1ID', 'bin2ID', 'observed_interactions', 'neg_ln_p_val', 'neg_ln_q_val',
                                                                         'exp_interactions', 'b1_bias', 'b2_bias']))

    else:
        interactions_info = np.column_stack((
            cis_interactions[:, 0], bins_info[cis_interactions[:, 0], 0], bins_info[cis_interactions[:, 0], 1], bins_info[cis_interactions[:, 0], 2],
            cis_interactions[:, 1], bins_info[cis_interactions[:, 1], 0], bins_info[cis_interactions[:, 1], 1], bins_info[cis_interactions[:, 1], 2],
            cis_interactions[:, 2],
            exp_interactions, -1 * log_p_vals, -1 * adj_q_val, vi_p, vj_p,
                                             bin_total_sum[cis_interactions[:, 0]],
                                             bin_total_sum[cis_interactions[:, 1]],
                                             bin_selfless_sum[cis_interactions[:, 0]],
                                             bin_selfless_sum[cis_interactions[:, 1]]))
        full_ints = np.concatenate((cis_interactions, interactions_info), axis=1)

        cis_saver_handler = execute_func_asynchronously(save_one_info, (
            full_ints, save_dir + '/cis_interactions.txt',
            ['bin1ID', 'bin1Chromosome', 'b1Start', 'b1End',
             'bin2ID', 'bin2Chromosome', 'b2Start', 'b2End',
             'read_count', 'exp_read_count', 'neg_ln_p_val', 'neg_ln_q_val', 'b1_bias', 'b2_bias',
             'b1_read_sum', 'b2_read_sum', 'b1_selfless_read_sum', 'b2_selfless_read_sum']))

    pd.DataFrame(vis).to_csv(params_save_dir + '/vis.txt', sep='\t', header=['v_i'], index=False)
    save_one_info(np.asarray([[pre_transform_norm_factor]]), params_save_dir + '/vis_transform_norm_factor.txt',
                  header=['norm_factor'])

    # SAVING CIS INFORMATION

    # saving parameters:
    save_one_info(np.asarray([cis_r]), params_save_dir + '/cis_r.txt', ['r'])
    save_one_info(np.asarray([cis_f_params]), params_save_dir + '/cis_distance_func_params.txt', header=['a3', 'a2', 'a1', 'a0', 'free_cis'])
    save_one_info(np.asarray([cis_vis_transformer_params]), params_save_dir + '/cis_hill_func_params.txt', header=['a', 'bv'])

    # SAVING TRANS INFORMATION

    # saving parameters:
    save_one_info(np.asarray([trans_r]), params_save_dir + '/trans_r.txt', ['r'])
    save_one_info(np.asarray([trans_f_param]), params_save_dir + '/trans_distance_func_param.txt', header=['free_trans'])
    save_one_info(np.asarray([trans_vis_transformer_params]), params_save_dir + '/trans_hill_func_params.txt', header=['a', 'bv'])

    cis_saver_handler.join()
    trans_saver_handler.join()
