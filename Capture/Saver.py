import pandas as pd
import numpy as np
from os import path, makedirs
from PValCalculationModel import np_nbinom_logsf
from Auxiliary import execute_func_asynchronously
from FuncsDefiner import \
    np_transformed_vis, np_cis_ints_expected_vals, np_trans_ints_expected_vals, get_b_type
from AllTFVariablesObjects import separate_order_ints


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
    
    
def save_results(cis_interactions, trans_interactions, cis_self_ints, model_info_dict, save_dir, full_output_mode, bait_mask,
                 bins_file_dir):

    bins_info = np.concatenate((np.asarray([['', 0, 0]]), pd.read_csv(bins_file_dir, sep='\t', header=None).values[:, 0:3]), axis=0)

    if not path.exists(save_dir):
        makedirs(save_dir)

    params_save_dir = save_dir + '/ModelParameters'
    if not path.exists(params_save_dir):
        makedirs(params_save_dir)

    saver_handlers = []

    # if full output is requested,
    if full_output_mode:

        # calculating read sum of bins

        max_bin_id = max(np.amax(cis_interactions[:, 0:2]), np.amax(trans_interactions[:, 0:2]), np.amax(cis_self_ints[:, 0:2]))
        bin_selfless_sum = np.full((max_bin_id + 1,), 0, dtype=int)
        bin_total_sum = np.full((max_bin_id + 1,), 0, dtype=int)

        np.add.at(bin_selfless_sum, cis_interactions[:, 0], cis_interactions[:, 2])
        np.add.at(bin_selfless_sum, cis_interactions[:, 1], cis_interactions[:, 2])
        np.add.at(bin_selfless_sum, trans_interactions[:, 0], trans_interactions[:, 2])
        np.add.at(bin_selfless_sum, trans_interactions[:, 1], trans_interactions[:, 2])

        bin_total_sum += bin_selfless_sum
        np.add.at(bin_total_sum, cis_self_ints[:, 0], cis_self_ints[:, 2])

    trans_ints_dict, cis_ints_dict = tuple([dict() for _ in range(2)])
    trans_ints_dict['t_bb'], trans_ints_dict['t_bo'], trans_ints_dict['t_oo'] = separate_order_ints(trans_interactions, bait_mask)
    cis_ints_dict['c_bb'], cis_ints_dict['c_bo'], cis_ints_dict['c_oo'] = separate_order_ints(cis_interactions, bait_mask)

    # SAVING TRANS INTERACTIONS ASYNCHRONOUSLY (AS IT TAKES MOST OF THE TIME)

    for itype in ['t_bb', 't_bo', 't_oo']:

        t_ints = trans_ints_dict[itype]

        # calculating expected value
        vi_p = np_transformed_vis(model_info_dict[get_b_type(itype, 0) + '_v_params'], model_info_dict['vis'][t_ints[:, 0]])
        vj_p = np_transformed_vis(model_info_dict[get_b_type(itype, 1) + '_v_params'], model_info_dict['vis'][t_ints[:, 1]])

        exp_interactions = np_trans_ints_expected_vals(t_ints, model_info_dict['vis'], model_info_dict[itype + '_dist_params'],
                                                       model_info_dict[get_b_type(itype, 0) + '_v_params'], model_info_dict[get_b_type(itype, 1) + '_v_params'])

        log_p_vals = np_nbinom_logsf(t_ints[:, 2], exp_interactions, model_info_dict[itype + '_r'])
        trans_adj_q_val = calculate_q_val_fdr(t_ints, np.exp(log_p_vals))

        if not full_output_mode:

            interactions_info = np.column_stack((-1 * log_p_vals, -1 * trans_adj_q_val, exp_interactions, vi_p, vj_p))
            full_ints = np.concatenate((t_ints, interactions_info), axis=1)

            saver_handlers.append(execute_func_asynchronously(save_one_info, (
                full_ints, save_dir + '/%s_trans_interactions.txt' % itype.split('_')[1],
                ['bin1ID', 'bin2ID', 'read_count', 'neg_ln_p_val', 'neg_ln_q_val', 'exp_read_count', 'b1_bias', 'b2_bias'])))

        else:
            interactions_info = np.column_stack((
                t_ints[:, 0], bins_info[t_ints[:, 0], 0], bins_info[t_ints[:, 0], 1], bins_info[t_ints[:, 0], 2],
                t_ints[:, 1], bins_info[t_ints[:, 1], 0], bins_info[t_ints[:, 1], 1], bins_info[t_ints[:, 1], 2],
                t_ints[:, 2],
                exp_interactions, -1 * log_p_vals, -1 * trans_adj_q_val, vi_p, vj_p,
                                                 bin_total_sum[t_ints[:, 0]], bin_total_sum[t_ints[:, 1]],
                                                 bin_selfless_sum[t_ints[:, 0]], bin_selfless_sum[t_ints[:, 1]]))

            saver_handlers.append(execute_func_asynchronously(save_one_info, (
                interactions_info, save_dir + '/%s_trans_interactions.txt' % itype.split('_')[1],
                ['bin1ID', 'bin1Chromosome', 'b1Start', 'b1End',
                 'bin2ID', 'bin2Chromosome', 'b2Start', 'b2End',
                 'read_count', 'exp_read_count', 'neg_ln_p_val', 'neg_ln_q_val', 'b1_bias', 'b2_bias',
                 'b1_read_sum', 'b2_read_sum', 'b1_selfless_read_sum', 'b2_selfless_read_sum'])))

    for itype in ['c_bb', 'c_bo', 'c_oo']:
        c_ints = cis_ints_dict[itype]

        # SAVING CIS INTERACTIONS ASYNCHRONOUSLY
        vi_p = np_transformed_vis(model_info_dict[get_b_type(itype, 0) + '_v_params'],
                                  model_info_dict['vis'][c_ints[:, 0]])
        vj_p = np_transformed_vis(model_info_dict[get_b_type(itype, 1) + '_v_params'],
                                  model_info_dict['vis'][c_ints[:, 1]])

        exp_interactions = np_cis_ints_expected_vals(c_ints, model_info_dict['vis'],
                                                     model_info_dict[itype + '_dist_params'],
                                                     model_info_dict[get_b_type(itype, 0) + '_v_params'],
                                                     model_info_dict[get_b_type(itype, 1) + '_v_params'])

        log_p_vals = np_nbinom_logsf(c_ints[:, 2], exp_interactions, model_info_dict[itype + '_r'])
        cis_adj_q_val = calculate_q_val_fdr(c_ints, np.exp(log_p_vals))

    if not full_output_mode:
        interactions_info = np.column_stack((-1 * log_p_vals, -1 * cis_adj_q_val, exp_interactions, vi_p, vj_p))
        full_ints = np.concatenate((c_ints, interactions_info), axis=1)

        saver_handlers.append(execute_func_asynchronously(save_one_info, (
        full_ints, save_dir + '/%s_cis_interactions.txt' % itype.split('_')[1],
        ['bin1ID', 'bin2ID', 'observed_interactions', 'neg_ln_p_val', 'neg_ln_q_val',
         'exp_interactions', 'b1_bias', 'b2_bias'])))

    else:

        interactions_info = np.column_stack((
            c_ints[:, 0], bins_info[c_ints[:, 0], 0], bins_info[c_ints[:, 0], 1], bins_info[c_ints[:, 0], 2],
            c_ints[:, 1], bins_info[c_ints[:, 1], 0], bins_info[c_ints[:, 1], 1], bins_info[c_ints[:, 1], 2],
            c_ints[:, 2],
            exp_interactions, -1 * log_p_vals, -1 * cis_adj_q_val, vi_p, vj_p,
            bin_total_sum[c_ints[:, 0]],
            bin_total_sum[c_ints[:, 1]],
            bin_selfless_sum[c_ints[:, 0]],
            bin_selfless_sum[c_ints[:, 1]]))

        saver_handlers.append(execute_func_asynchronously(save_one_info, (
            interactions_info, save_dir + '/%s_cis_interactions.txt' % itype.split('_')[1],
            ['bin1ID', 'bin1Chromosome', 'b1Start', 'b1End',
             'bin2ID', 'bin2Chromosome', 'b2Start', 'b2End',
             'read_count', 'exp_read_count', 'neg_ln_p_val', 'neg_ln_q_val', 'b1_bias', 'b2_bias',
             'b1_read_sum', 'b2_read_sum', 'b1_selfless_read_sum', 'b2_selfless_read_sum'])))

    save_one_info(np.asarray([[model_info_dict['s_norm']]]), params_save_dir + '/vis_transform_norm_factor.txt',
                  header=['norm_factor'])

    # SAVING CIS INFORMATION

    pd.DataFrame(model_info_dict['vis']).to_csv(params_save_dir + '/vis.txt', sep='\t', header=['v_i'], index=False)

    # saving parameters:
    for itype in ['c_bb', 'c_bo', 'c_oo']:
        save_one_info(np.asarray([model_info_dict[itype + '_r']]),
                      params_save_dir + '/%s_cis_r.txt' % itype, ['r'])
        save_one_info(np.asarray([model_info_dict[itype + '_dist_params']]),
                      params_save_dir + '/%s_cis_distance_func_params.txt' % itype, header=['a3', 'a2', 'a1', 'a0', 'free_cis'])

    for ctype in ['c', 't']:
        for itype in ['bb_b', 'bo_b', 'bo_o', 'oo_o']:
            save_one_info(np.asarray([model_info_dict['%s_%s_v_params' % (ctype, itype)]]),
                          params_save_dir + '/%s_%s_v_func_params.txt' % (ctype, itype), header=['b', 'b0'])

    # SAVING TRANS INFORMATION

    # saving parameters:
    for itype in ['t_bb', 't_bo', 't_oo']:
        save_one_info(np.asarray([model_info_dict[itype + '_r']]),
                      params_save_dir + '/%s_trans_r.txt' % itype, ['r'])
        save_one_info(np.asarray([model_info_dict[itype + '_dist_params']]),
                      params_save_dir + '/%s_trans_distance_func_params.txt' % itype, header=['free_trans'])

    for sh in saver_handlers:
        sh.join()
