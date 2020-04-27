import tensorflow as tf
from time import time
import numpy as np
import ChrVisUpdating, CisSharedParamsUpdatingModel, TransSharedParamsUpdatingModel, \
    TrainingInteractionsSeparatorModel, AllTFVariablesObjects
from Auxiliary import execute_func_asynchronously
from FuncsDefiner import smax


def run_ml(chrs_ranges, cis_interactions, trans_interactions, sig_p_val, cores_num, remove_sig_from_vis, rounds_num,
           min_dist, max_dist, min_rc, silent_mode, estimated_r=2.0, debug_mode=False):

    sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=cores_num,
                                            inter_op_parallelism_threads=cores_num))
    tf.set_random_seed(0)

    start_time = time()

    # making an array for all vis
    max_bin_id = max(smax(cis_interactions[:, 0:2]), smax(trans_interactions[:, 0:2]))
    vis = np.zeros((max_bin_id + 1,), dtype=np.float64)

    # finding active_bins
    chrs_active_bins = np.unique(np.concatenate((cis_interactions[:, 0:2], trans_interactions[:, 0:2]), axis=0))

    temp_cis_f_params = np.polyfit(np.log(np.abs(cis_interactions[:, 0] - cis_interactions[:, 1])),
                                   np.log(cis_interactions[:, 2]), 3)
    estimated_cis_f_params = np.asarray(list(tuple(temp_cis_f_params)) + [1.0], dtype=np.float64)
    estimated_cis_f_params[0] = min(-0.001, estimated_cis_f_params[0])

    # making tf variables
    vis = tf.get_variable('vis', initializer=vis, dtype=tf.float64)
    chrs_active_bins = tf.get_variable('active_bins', initializer=chrs_active_bins.astype(np.int32), dtype=tf.int32)

    # setting training interactions using a placeholder
    chrs_bin_ranges = tf.get_variable('chrs_ranges', initializer=np.asarray(chrs_ranges, dtype=np.int32), dtype=tf.int32)

    cis_training_interactions = tf.Variable([0], name='cis_training_interactions', dtype=tf.int32, validate_shape=False)
    cis_place = tf.placeholder(tf.int32, shape=cis_interactions.shape)
    cis_ints_assign = tf.assign(cis_training_interactions, cis_place, validate_shape=False)

    trans_training_interactions = tf.Variable([0], name='trans_training_interactions', dtype=tf.int32, validate_shape=False)
    trans_place = tf.placeholder(tf.int32, shape=trans_interactions.shape)
    trans_ints_assign = tf.assign(trans_training_interactions, trans_place, validate_shape=False)

    cis_significant_interactions = tf.get_variable('cis_sig_ints', shape=[1, 3], dtype=tf.int32, validate_shape=False)
    trans_significant_interactions = tf.get_variable('trans_sig_ints', shape=[1, 3], dtype=tf.int32, validate_shape=False)

    cis_r = tf.get_variable('cis_r', initializer=np.float64(estimated_r), dtype=tf.float64)
    cis_f_params = tf.get_variable('cis_f_params', initializer=estimated_cis_f_params.astype(np.float64), dtype=tf.float64)
    cis_vis_transformer_params = tf.get_variable('cis_vis_transformer_params',
                                                 initializer=np.asarray([1, 1], dtype=np.float64), dtype=tf.float64)

    trans_r = tf.get_variable('trans_r', initializer=np.float64(estimated_r), dtype=tf.float64)
    trans_f_param = tf.get_variable('trans_f_params', initializer=np.float64(1.0), dtype=tf.float64)
    trans_vis_transformer_params = tf.get_variable('trans_vis_transformer_params',
                                                   initializer=np.asarray([1, 1], dtype=np.float64), dtype=tf.float64)

    pre_transform_norm_factor = tf.get_variable('pre_transform_norm_factor', initializer=np.float64(1.0), dtype=tf.float64)

    sess.run(tf.variables_initializer([vis, chrs_active_bins, cis_r, cis_f_params, cis_vis_transformer_params,
                                       trans_r, trans_f_param, trans_vis_transformer_params, chrs_bin_ranges,
                                       pre_transform_norm_factor]))
    sess.run([cis_ints_assign, trans_ints_assign], feed_dict={cis_place: cis_interactions, trans_place: trans_interactions})

    # defining objects holder
    tf_var_objs_holder = AllTFVariablesObjects.AllTFVariablesObjectsHolder(
        vis, chrs_active_bins, chrs_bin_ranges,
        cis_training_interactions, trans_training_interactions, cis_significant_interactions, trans_significant_interactions,
        cis_r, cis_f_params, cis_vis_transformer_params,
        trans_r, trans_f_param, trans_vis_transformer_params,
        pre_transform_norm_factor,
        sess)

    # defining models
    cis_shared_params_updater = CisSharedParamsUpdatingModel.CisSharedParamsUpdater(tf_var_objs_holder,
                                                                                    min_dist, max_dist, min_rc, silent_mode)
    trans_shared_params_updater = TransSharedParamsUpdatingModel.TransSharedParamsUpdater(tf_var_objs_holder,
                                                                                          min_rc, silent_mode)

    vis_updater = ChrVisUpdating.VisUpdater(tf_var_objs_holder, cores_num, remove_sig_from_vis, silent_mode)

    training_interactions_separator = TrainingInteractionsSeparatorModel.TrainingInteractionsSeparator(tf_var_objs_holder,
                                                                                                       sig_p_val, silent_mode)

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
            training_interactions_separator.separate_training_interactions(cis_interactions, trans_interactions)

        # fixing the vis based on insignificant interactions only
        if big_step == 0:
            vis_updater.run_model(assignment_stage=0)
        else:
            vis_updater.run_model(assignment_stage=1)

        # updating shared parameters
        cis_shp_controller = execute_func_asynchronously(cis_shared_params_updater.run_model, ())
        trans_shp_controller = execute_func_asynchronously(trans_shared_params_updater.run_model, ())

        cis_shp_controller.join()
        trans_shp_controller.join()

        if (not silent_mode) and debug_mode:
            r_val, f_params_val, vis_transformer_params_vals = sess.run([cis_r, cis_f_params, cis_vis_transformer_params])
            print('Current refitted cis shared params are :' + '\t'.join([
                '%.4f' % r_val, *tuple(['%.4f' % x for x in f_params_val]), *tuple(['%.8f' % x for x in vis_transformer_params_vals])]))

        if (not silent_mode) and debug_mode:
            r_val, f_param_val, vis_transformer_params_vals = sess.run([trans_r, trans_f_param, trans_vis_transformer_params])
            print('Current refitted trans shared params are :' + '\t'.join([
                '%.4f' % r_val, '%.4f' % f_param_val, *tuple(['%.8f' % x for x in vis_transformer_params_vals])]))

        # resetting distance based norm factors based on newly calculated dist params.
        vis_updater.reset_dist_related_norm_params()

        if not silent_mode:
            print('Iteration ' + str(big_step + 1) + ' ended at time ' + str(time() - m_time))

    print('Fitting the model ended in ' + str(time() - start_time) + ' secs.')
    return sess.run([vis, cis_f_params, cis_r, cis_vis_transformer_params, trans_f_param, trans_r, trans_vis_transformer_params,
                     pre_transform_norm_factor])

