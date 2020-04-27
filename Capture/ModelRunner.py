from __future__ import print_function
from time import time
import tensorflow as tf
from AllTFVariablesObjects import AllTFVariablesObjectsHolder
from ChrVisUpdating import VisUpdater
from CisSharedParamsUpdatingModel import \
    CisSharedParamsUpdater
from TransSharedParamsUpdatingModel import \
    TransSharedParamsUpdater
from TrainingInteractionsSeparatorModel import \
    TrainingInteractionsSeparator
from FuncsDefiner import part_assign


def run_ml(chrs_ranges, cis_interactions, trans_interactions, sig_p_val, cores_num, remove_sig_from_vis,
           rounds_num, min_dist, max_dist, min_read, silent_mode, bins_bait_mask, debug_mode=False):

    start_time = time()

    # making tf variables
    tf_var_objs_holder = AllTFVariablesObjectsHolder(chrs_ranges,
        cis_interactions, trans_interactions, cores_num, min_dist, max_dist, min_read, bins_bait_mask, silent_mode==False)

    # defining models
    cis_dist_params_updater, trans_dist_params_updater, vis_updater, ints_seperator, \
        training_interactions_seperator = tuple([dict() for _ in range(5)])

    for itype in ['bb', 'bo', 'oo']:
        cis_dist_params_updater[itype] = CisSharedParamsUpdater(tf_var_objs_holder, 'c_' + itype, silent_mode)
        trans_dist_params_updater[itype] = TransSharedParamsUpdater(tf_var_objs_holder, 't_' + itype, silent_mode)

    for itype in ['c_bb', 'c_bo', 'c_oo', 't_bb', 't_bo', 't_oo']:
        ints_seperator[itype] = TrainingInteractionsSeparator(tf_var_objs_holder, itype, sig_p_val, silent_mode)

    vis_updater = VisUpdater(tf_var_objs_holder, cores_num, remove_sig_from_vis, silent_mode)

    for itype in ['c_bb', 'c_bo', 'c_oo', 't_bb', 't_bo', 't_oo']:
        training_interactions_seperator[itype] = \
            TrainingInteractionsSeparator(tf_var_objs_holder, itype, sig_p_val, silent_mode)

    # fitting the model :D
    for big_step in range(rounds_num):

        if not silent_mode:
            print('>>>>>>>>>> Iteration ' + str(big_step + 1))

        m_time = time()

        if big_step > 0:

            # retraining vis, separating probable significant interactions
            if big_step > 2:
                vis_updater.run_model(assignment_stage=2)

            # separating interactions
            for itype in ['c_bb', 'c_bo', 'c_oo', 't_bb', 't_bo', 't_oo']:
                training_interactions_seperator[itype].separate_training_interactions()

        for itype in ['bb', 'bo', 'oo']:
            cis_dist_params_updater[itype].run_model()
            trans_dist_params_updater[itype].run_model()
            if not silent_mode and debug_mode:
                tf_var_objs_holder.print_cis_params('c_' + itype)
                tf_var_objs_holder.print_trans_params('t_' + itype)

        # resetting distance based norm factors based on newly calculated dist params.
        vis_updater.reset_dist_related_norm_params()
        vis_updater.run_model(assignment_stage=1)

        if not silent_mode:
            print('Iteration ' + str(big_step + 1) + ' ended at time ' + str(time() - m_time))

    for itype in ['bb', 'bo', 'oo']:
        cis_dist_params_updater[itype].run_model()
        trans_dist_params_updater[itype].run_model()
        if not silent_mode and debug_mode:
            tf_var_objs_holder.print_cis_params('c_' + itype)
            tf_var_objs_holder.print_trans_params('t_' + itype)

    # Resetting the invisible ones in vis to 0
    all_active_bins = tf.sort(tf.concat([tf_var_objs_holder.active_bins['b'], tf_var_objs_holder.active_bins['o']], axis=0))
    v_found_values = tf.gather(tf_var_objs_holder.vis, all_active_bins)
    new_vis_vals = tf.sparse_to_dense(all_active_bins, tf_var_objs_holder.vis.shape, v_found_values,
                                      default_value=tf.zeros([], tf.float64))
    _ = tf_var_objs_holder.sess.run(tf.assign(tf_var_objs_holder.vis, new_vis_vals))

    print('Fitting the model ended in ' + str(time() - start_time) + ' secs.')

    return tf_var_objs_holder.return_final_vals()
