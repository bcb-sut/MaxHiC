import tensorflow as tf
import numpy as np
from time import time
# CARE: To eliminate the effect of scale (smaller f => higher vis => more smaller f!! => eternity!)
# The scale multiplier of f (last cis_f_params) is not included.
from FuncsDefiner import tf_transformed_vis, tf_cis_dist_func, tf_unsorted_segment_avg, \
    tf_cis_ints_expected_vals, tf_trans_ints_expected_vals, part_assign
from IntEquivalentCal import IntEquivalentCal


# one object of this class per chr (more efficient per thread) so multiple graphs of this type can run in parallel!
class VisUpdater:

    def __init__(self, objs_holder, cores_num, remove_sig_from_vis, silent_mode):

        self.sess = objs_holder.sess
        self.cores_num = cores_num
        self.remove_sig_from_vis = remove_sig_from_vis
        self.silent_mode = silent_mode

        self.ints_eq_converter = IntEquivalentCal(objs_holder)

        # universally known variables
        self.chrs_bin_ranges = objs_holder.chrs_bin_ranges
        self.vis = objs_holder.vis

        self.active_bin_ids = tf.sort(tf.concat([objs_holder.active_bins['b'], objs_holder.active_bins['o']], axis=0))
        _, self.max_bin_id = self.sess.run([self.active_bin_ids, tf.reduce_max(self.active_bin_ids)])

        self.cis_dist_params = objs_holder.dist_params['c_bb']
        self.cis_v_params = objs_holder.v_params['c_bb_b']

        self.trans_dist_params = objs_holder.dist_params['t_bb']
        self.trans_v_params = objs_holder.v_params['t_bb_b']

        self.cis_training_b1_id, self.cis_training_b2_id, self.cis_training_rc = self.organize_ints(
            objs_holder.insig_ints, 'c')
        self.trans_training_b1_id, self.trans_training_b2_id, self.trans_training_rc = self.organize_ints(
            objs_holder.insig_ints, 't')

        self.cis_sig_b1_id, self.cis_sig_b2_id, self.cis_sig_rc = self.organize_ints(objs_holder.sig_ints, 'c')
        self.trans_sig_b1_id, self.trans_sig_b2_id, self.trans_sig_rc = self.organize_ints(objs_holder.sig_ints, 't')

        self.max_tie_point = self.sess.run(tf.reduce_max(tf.abs(self.cis_training_b1_id - self.cis_training_b2_id)))

        # These need to be evaluated (3 different types of vis assignment!)
        # all normalized as geo_mean  = 1
        self.vis_as_pure_sum_assign = None
        self.vis_as_options_effected_assign = None
        self.vis_as_probable_sigs_effects_reduced_assign = None

        # NORMALIZATION FACTOR RELATED PARAMETERS

        # based on open options = the effect of significant interactions is not reduced
        self.based_on_open_options_norm_factors = tf.get_variable('bin_norm_factors', shape=[self.max_bin_id + 1], dtype=tf.float64)
        self.based_on_open_options_norm_factors_assign = None

        # cis f function's tie break with the free parameter
        self.cis_f_tie_point = tf.get_variable('cis_f_tie_point', shape=[], dtype=tf.float64)
        self.cis_f_tie_point_placeholder = tf.placeholder(tf.float64, shape=[])
        self.cis_f_tie_point_setter = tf.assign(self.cis_f_tie_point, self.cis_f_tie_point_placeholder)

        # These are added as open options for everybody (beside the available options before the tie
        # (except for the significant pair and the invisible bins!))
        self.normalized_cis_after_tie_ints_options = tf.get_variable('normalized_cis_after_tie_ints_options', shape=[], dtype=tf.float64)
        self.normalized_trans_ints_options = tf.get_variable('normalized_trans_options', shape=[], dtype=tf.float64)

        self.normalized_cis_after_tie_ints_options_assign = None
        self.normalized_trans_ints_options_assign = None

        # These are the weights of the options (normalized with respect to the cis' free parameter)
        self.normalized_before_tie_cis_weights = tf.get_variable('normalized_before_tie_cis_weights_assign',
                                                                 shape=[1], dtype=tf.float64, validate_shape=False)
        self.normalized_trans_weight = tf.get_variable('normalized_trans_weight', shape=[], dtype=tf.float64)

        self.normalized_before_tie_cis_weights_assign = None
        self.normalized_trans_weight_assign = None

        self.pre_transform_norm_factor = objs_holder.pre_transform_norm_factor
        self.pre_transform_norm_factor_assign_state1 = None
        self.pre_transform_norm_factor_assign_state2 = None
        self.pre_transform_norm_factor_assign_state3 = None

        self.define_dist_related_normalization_parameters_model()
        self.define_base_norm_factor_based_on_open_options_calculation_model()

        self.define_vis_as_sum_only_model()
        self.define_vis_as_normalized_with_open_options_model()
        self.define_recalculate_vis_without_probable_sig_interactions_model()

    # ********** for calculating interactions ********** #

    def organize_ints(self, ints_dict, gt):

        b1_id = tf.concat([
            ints_dict[gt + '_bb'][:, 0], ints_dict[gt + '_bb'][:, 1], ints_dict[gt + '_bo'][:, 0],
            ints_dict[gt + '_bo'][:, 1], ints_dict[gt + '_oo'][:, 0], ints_dict[gt + '_oo'][:, 1]
        ], axis=0)
        b2_id = tf.concat([
            ints_dict[gt + '_bb'][:, 1], ints_dict[gt + '_bb'][:, 0], ints_dict[gt + '_bo'][:, 1],
            ints_dict[gt + '_bo'][:, 0], ints_dict[gt + '_oo'][:, 1], ints_dict[gt + '_oo'][:, 0]
        ], axis=0)
        rc = tf.concat([
            tf.cast(ints_dict[gt + '_bb'][:, 2], tf.float64), tf.cast(ints_dict[gt + '_bb'][:, 0], tf.float64),
            self.ints_eq_converter.convert(ints_dict[gt + '_bo'], gt + '_bo', gt + '_bb'),
            self.ints_eq_converter.convert(ints_dict[gt + '_bo'], gt + '_bo', gt + '_bb'),
            self.ints_eq_converter.convert(ints_dict[gt + '_oo'], gt + '_oo', gt + '_bb'),
            self.ints_eq_converter.convert(ints_dict[gt + '_oo'], gt + '_oo', gt + '_bb')
        ], axis=0)

        return b1_id, b2_id, rc

        # ********** for resetting norm factors related to distance params ********** #

    def reset_cis_dist_func_tie_point(self):

        # OK what were we doing? :)
        cis_f_params = self.sess.run(self.cis_dist_params)
        tie_points = np.roots(cis_f_params[:4] - np.asarray([0, 0, 0, cis_f_params[4]]))
        tie_points = tie_points[np.logical_not(np.iscomplex(tie_points))]
        tie_points = tie_points[tie_points > 0]
        tie_points = [np.float64(np.real(x)) for x in tie_points]

        if len(tie_points) == 0:
            tie_points = [np.log(self.max_tie_point)]

        # picking the min
        the_tie_point = min(np.exp(np.float64(np.amin(tie_points))), np.float64(self.max_tie_point))
        if not self.silent_mode:
            print('new tie point %.2f' % the_tie_point)

        self.sess.run(self.cis_f_tie_point_setter, feed_dict={self.cis_f_tie_point_placeholder: the_tie_point})

    def define_dist_related_normalization_parameters_model(self):

        # counting the number of cis interactions before tie
        before_tie_mask = tf.less_equal(tf.cast(tf.abs(self.cis_training_b1_id - self.cis_training_b2_id), tf.float64), self.cis_f_tie_point)
        number_of_cis_ints_before_tie = tf.reduce_sum(tf.cast(before_tie_mask, tf.int32))
        number_of_cis_ints_after_tie = tf.size(before_tie_mask) - number_of_cis_ints_before_tie
        number_of_trans_ints = tf.size(self.trans_training_b1_id)
        number_of_cis_options_before_tie = tf.floor(self.cis_f_tie_point)

        # checking if tie is less than one (number of cis options before tie being 0)
        normalized_cis_after_tie_ints_options = tf.cond(
            tf.less(self.cis_f_tie_point, tf.constant(1, dtype=tf.float64)),
            lambda: (tf.cast(number_of_cis_ints_after_tie, tf.float64) / tf.cast(tf.size(self.active_bin_ids), tf.float64)),
            lambda: tf.cast(number_of_cis_options_before_tie, tf.float64) * tf.cast(number_of_cis_ints_after_tie, tf.float64) / \
            tf.cast(number_of_cis_ints_before_tie, tf.float64)
        )

        normalized_trans_ints_options = tf.cond(
            tf.less(self.cis_f_tie_point, tf.constant(1, dtype=tf.float64)),
            lambda: (tf.cast(number_of_trans_ints, tf.float64) / tf.cast(tf.size(self.active_bin_ids), tf.float64)),
            lambda: tf.cast(number_of_cis_options_before_tie, tf.float64) * tf.cast(number_of_trans_ints, tf.float64) / \
            tf.cast(number_of_cis_ints_before_tie, tf.float64)
        )

        # These are the weights of the options (normalized with respect to the cis' free parameter)
        distance_options_before_tie = tf.range(1, number_of_cis_options_before_tie + 1)
        log_dists = tf.log(tf.cast(distance_options_before_tie, tf.float64))

        # append a zero before everything to make it 1-indexed (distance as index)
        normalized_before_tie_cis_weights = \
            tf.concat([
                tf.zeros([1], dtype=tf.float64),
                tf.exp(
                    self.cis_dist_params[0] * tf.pow(log_dists, 3) + self.cis_dist_params[1] * tf.pow(log_dists, 2) +
                    self.cis_dist_params[2] * log_dists + self.cis_dist_params[3] - self.cis_dist_params[4])], 0)
        normalized_trans_weight = tf.exp(self.trans_dist_params - self.cis_dist_params[4])

        # assignments

        self.normalized_cis_after_tie_ints_options_assign = tf.assign(
            self.normalized_cis_after_tie_ints_options, normalized_cis_after_tie_ints_options)

        self.normalized_trans_ints_options_assign = tf.assign(
            self.normalized_trans_ints_options, normalized_trans_ints_options
        )

        self.normalized_before_tie_cis_weights_assign = tf.assign(
            self.normalized_before_tie_cis_weights, normalized_before_tie_cis_weights, validate_shape=False
        )

        self.normalized_trans_weight_assign = tf.assign(
            self.normalized_trans_weight, normalized_trans_weight
        )

    # ******************* for calculating distance related values ******************* #

    def calculate_option_weight_for_cis_ints(self, cis_b1, cis_b2):

        dists = tf.cast(tf.abs(cis_b1 - cis_b2), tf.float64)

        # the number of interactions
        ints_num = tf.size(cis_b1)

        # having a tensor of indices
        ints_indices = tf.range(0, ints_num)

        # separating before and after tie indices, and interactions
        before_tie_mask = tf.less_equal(dists, self.cis_f_tie_point)
        before_tie_mask.set_shape([None])

        before_tie_indices = tf.boolean_mask(ints_indices, before_tie_mask)
        after_tie_indices = tf.boolean_mask(ints_indices, tf.logical_not(before_tie_mask))

        # finding weights of options before tie
        before_tie_dists = tf.boolean_mask(dists, before_tie_mask)
        before_tie_ints_weights = tf.gather(self.normalized_before_tie_cis_weights, before_tie_dists)

        after_tie_ints_num = tf.shape(after_tie_indices)[0]

        all_weights = \
            tf.sparse_to_dense(before_tie_indices, tf.expand_dims(ints_num, 0), before_tie_ints_weights) + \
            tf.sparse_to_dense(after_tie_indices, tf.expand_dims(ints_num, 0), tf.ones(tf.expand_dims(after_tie_ints_num, 0), dtype=tf.float64))

        return all_weights

    def calculate_dist_func(self, dij):
        return tf_cis_dist_func(self.cis_dist_params, dij)

    # ********* for resetting based on open options only norm factors ********** #

    def chr_bins_base_norm_body(self, chr_index, prev_bin_ids, prev_base_norm_factors):

        chr_start_bin = self.chrs_bin_ranges[chr_index, 0]
        chr_end_bin = self.chrs_bin_ranges[chr_index, 1]

        # separating binIDs related to this chr
        chr_bin_ids = tf.boolean_mask(self.active_bin_ids, tf.logical_and(
            tf.greater_equal(self.active_bin_ids, chr_start_bin), tf.less_equal(self.active_bin_ids, chr_end_bin)
        ))

        # making a bitmap for visible bins of chr
        visible_bins_mask = tf.sparse_to_dense(chr_bin_ids - chr_start_bin, tf.expand_dims(chr_end_bin + 1 - chr_start_bin, 0),
                                               np.float32(1.0), np.float32(0.0))

        # calculating options with convolution, the filter is the weights for cis before tie options
        option_weight_filter = tf.cast(tf.concat([
            tf.reverse(self.normalized_before_tie_cis_weights[1:], axis=[0]), self.normalized_before_tie_cis_weights], 0), tf.float32)

        bins_cis_options = tf.nn.conv2d(
            tf.reshape(visible_bins_mask, [1, 1, -1, 1]),
            tf.reshape(option_weight_filter, [1, -1, 1, 1]),
            strides=[1, 1, 1, 1],
            padding="SAME")

        # extracting the options related to visible bins
        # adding trans and after tie cis options

        visible_bins_options = tf.cast(tf.gather(tf.reshape(bins_cis_options, [-1]), chr_bin_ids - chr_start_bin), tf.float64) + \
            self.normalized_trans_weight * self.normalized_trans_ints_options + self.normalized_cis_after_tie_ints_options

        new_bin_ids = tf.concat([prev_bin_ids, chr_bin_ids], 0)
        new_base_norm_factors = tf.concat([prev_base_norm_factors, visible_bins_options], 0)

        new_bin_ids.set_shape([None])
        new_base_norm_factors.set_shape([None])

        return chr_index + 1, new_bin_ids, new_base_norm_factors

    def chr_bins_base_norm_cond(self, chr_index, prev_bin_ids, prev_base_norm_factors):

        chrs_ranges_shape = tf.shape(self.chrs_bin_ranges)
        chrs_num = chrs_ranges_shape[0]
        return tf.less(chr_index, chrs_num)

    def define_base_norm_factor_based_on_open_options_calculation_model(self):

        _, bin_ids, base_norm_factors = tf.while_loop(self.chr_bins_base_norm_cond, self.chr_bins_base_norm_body,
                                                      [tf.constant(0, dtype=tf.int32),
                                                       tf.zeros([0], dtype=tf.int32),
                                                       tf.zeros([0], dtype=tf.float64)],
                                                      shape_invariants=[
                                                          tf.TensorShape([]),
                                                          tf.TensorShape([None]),
                                                          tf.TensorShape([None])
                                                      ], parallel_iterations=self.cores_num)

        expanded_norm_factors = tf.sparse_to_dense(bin_ids, [self.max_bin_id + 1], base_norm_factors, np.float64(0.0))
        self.based_on_open_options_norm_factors_assign = tf.assign(self.based_on_open_options_norm_factors, expanded_norm_factors)

    # ** Auxiliary functions for aggregating the effect of significant interactions ** #

    def calculate_sum_of_insignificant_interactions_min_one_for_visible_bins(self, cis_b1, cis_rc, trans_b1, trans_rc):

        # based on multi nomial distribution's likelihood optimization! each vi ~ sum of observed reads of each bin
        #  :D no need for division! F will support it...
        bins_cis_sum = tf.unsorted_segment_sum(cis_rc, cis_b1, self.max_bin_id + 1)
        bins_trans_sum = tf.unsorted_segment_sum(trans_rc, trans_b1, self.max_bin_id + 1)
        bins_sum = bins_cis_sum + bins_trans_sum

        visible_bins_ints_sum = tf.maximum(tf.ones([], tf.float64), tf.gather(bins_sum, self.active_bin_ids))

        return tf.sparse_to_dense(self.active_bin_ids, [self.max_bin_id + 1], visible_bins_ints_sum, default_value=tf.zeros([], dtype=tf.float64))

    def calculate_sum_of_significant_interactions_expectations(self, cis_b1, cis_b2, trans_b1, trans_b2):

        # calculating dist function for cis ints
        cis_exp = tf_cis_ints_expected_vals(tf.stack([cis_b1, cis_b2], axis=1), self.vis, self.cis_dist_params, 
                                            self.cis_v_params, self.cis_v_params)

        trans_exp = tf_trans_ints_expected_vals(tf.stack([trans_b1, trans_b2], axis=1), self.vis, self.trans_dist_params,
                                                self.trans_v_params, self.trans_v_params)

        # based on multi nomial distribution's likelihood optimization! each vi ~ sum of observed reads of each bin
        #  :D no need for division! F will support it...
        bins_cis_sum = tf.unsorted_segment_sum(cis_exp, cis_b1, self.max_bin_id + 1)
        bins_trans_sum = tf.unsorted_segment_sum(trans_exp, trans_b1, self.max_bin_id + 1)
        bins_sum = bins_cis_sum + bins_trans_sum

        return bins_sum

    def calculate_aggregated_significant_interactions_reduction_effect_from_norm_factors(self, 
                                                                                         cis_b1, cis_b2, trans_b1):

        # separating cis interactions before tie
        cis_num = tf.size(cis_b1)
        trans_num = tf.size(trans_b1)

        #trans_shape = tf.Print(trans_shape, [trans_shape], "trans shape")

        cis_ints_weights = tf.cond(
            tf.equal(cis_num, tf.constant(0, dtype=tf.int32, shape=[])),
            lambda: tf.zeros([self.max_bin_id + 1], dtype=tf.float64),
            lambda: self.calculate_option_weight_for_cis_ints(cis_b1, cis_b2)
        )

        cis_effects = tf.cond(
            tf.less_equal(cis_num, tf.constant(1, dtype=tf.int32, shape=[])),
            lambda: tf.zeros([self.max_bin_id + 1], dtype=tf.float64),
            lambda: tf.unsorted_segment_sum(cis_ints_weights, cis_b1, self.max_bin_id + 1
            )
        )

        trans_effects = tf.cond(
            tf.less_equal(trans_num, tf.constant(1, dtype=tf.int32, shape=[])),
            lambda: tf.zeros([self.max_bin_id + 1], dtype=tf.float64),
            lambda: tf.unsorted_segment_sum(
                self.normalized_trans_weight * tf.ones(tf.expand_dims(trans_num, 0), dtype=tf.float64),
                trans_b1, self.max_bin_id + 1)
        )

        agg_effect_of_sigs = cis_effects + trans_effects

        return agg_effect_of_sigs

    # ***************** Vis assignment and reassignment models ***************** #

    def calculate_geo_mean(self, cal_vis):

        # extracting visible bins
        visible_bins_vis = tf.gather(cal_vis, self.active_bin_ids)

        # calculating geo mean
        geo_mean = tf.exp(tf.reduce_mean(tf.log(visible_bins_vis)))

        return geo_mean

    def define_vis_as_sum_only_model(self):

        bins_ints_sum = self.calculate_sum_of_insignificant_interactions_min_one_for_visible_bins(
            self.cis_training_b1_id, self.cis_training_rc, self.trans_training_b1_id, self.trans_training_rc)
        geo_mean = self.calculate_geo_mean(bins_ints_sum)
        self.pre_transform_norm_factor_assign_state1 = tf.assign(self.pre_transform_norm_factor, 1.0 / geo_mean)
        self.vis_as_pure_sum_assign = part_assign(self.vis, self.active_bin_ids, 
                                                  tf.gather(bins_ints_sum / geo_mean, self.active_bin_ids))

    def define_vis_as_normalized_with_open_options_model(self):

        bins_insig_ints_sum = self.calculate_sum_of_insignificant_interactions_min_one_for_visible_bins(
            self.cis_training_b1_id, self.cis_training_rc, self.trans_training_b1_id, self.trans_training_rc)

        if self.remove_sig_from_vis:
            bins_sig_ints_exp_sum = self.calculate_sum_of_significant_interactions_expectations(
                self.cis_sig_b1_id, self.cis_sig_b2_id, self.trans_sig_b1_id, self.trans_sig_b2_id)
        else:
            bins_sig_ints_exp_sum = self.calculate_sum_of_insignificant_interactions_min_one_for_visible_bins(
                self.cis_sig_b1_id, self.cis_sig_rc, self.trans_sig_b1_id, self.trans_sig_rc)

        bins_ints_sum = bins_sig_ints_exp_sum + bins_insig_ints_sum
        bins_norm_factors = self.based_on_open_options_norm_factors

        calculated_vis = bins_ints_sum / tf.maximum(tf.ones([], dtype=tf.float64), bins_norm_factors)
        geo_mean = self.calculate_geo_mean(calculated_vis)

        self.pre_transform_norm_factor_assign_state2 = tf.assign(self.pre_transform_norm_factor, 1.0 / geo_mean)
        self.vis_as_options_effected_assign = part_assign(self.vis, self.active_bin_ids,
                                                  tf.gather(calculated_vis / geo_mean, self.active_bin_ids))

    def define_recalculate_vis_without_probable_sig_interactions_model(self):

        cis_f_dij = self.calculate_dist_func(tf.abs(self.cis_training_b1_id - self.cis_training_b2_id))
        cis_x_ij = self.cis_training_rc
        cis_xij_prime = cis_x_ij / cis_f_dij

        trans_x_ij = self.trans_training_rc
        trans_xij_prime = trans_x_ij / tf.exp(self.trans_dist_params)

        # calculating the current vis (not geo mean normalized)
        bins_insig_ints_sum = self.calculate_sum_of_insignificant_interactions_min_one_for_visible_bins(
            self.cis_training_b1_id, self.cis_training_rc, self.trans_training_b1_id, self.trans_training_rc)
        bins_sig_exp_sum = self.calculate_sum_of_significant_interactions_expectations(
            self.cis_sig_b1_id, self.cis_sig_b2_id, self.trans_sig_b1_id, self.trans_sig_b2_id)
        bins_ints_sum = bins_sig_exp_sum + bins_insig_ints_sum

        # calculating norm factors for each bin
        #bins_norm_factors = self.based_on_open_options_norm_factors - \
        #    self.calculate_aggregated_significant_interactions_reduction_effect_from_norm_factors(self.cis_sig_ints, self.trans_sig_ints)
        bins_norm_factors = self.based_on_open_options_norm_factors

        current_vis = bins_ints_sum / tf.maximum(tf.ones([], dtype=tf.float64), bins_norm_factors)

        # calculating vis after assuming each interaction significant (reducing the effect of interaction from both ends)
        # the interaction itself would be reduced from sum, and the option weight would be reduced from the norm factor
        cis_exp_replacement = tf_cis_ints_expected_vals(tf.stack([self.cis_training_b1_id, self.cis_training_b2_id], axis=1),
                                                        current_vis, self.cis_dist_params,
                                                        self.cis_v_params, self.cis_v_params)

        trans_exp_replacement = tf_trans_ints_expected_vals(tf.stack([self.trans_training_b1_id, self.trans_training_b2_id], axis=1),
                                                            current_vis, self.trans_dist_params,
                                                            self.trans_v_params, self.trans_v_params)

        #cis_option_weights = self.calculate_option_weight_for_cis_ints(self.cis_training_ints)
        b1_vis_after_eliminating_each_cis_interaction = \
            tf.maximum(tf.ones([], tf.float64), (tf.gather(bins_ints_sum, self.cis_training_b1_id) - cis_x_ij +
                                                 cis_exp_replacement)) / \
            (tf.gather(bins_norm_factors, self.cis_training_b1_id))  # - cis_option_weights)
        b2_vis_after_eliminating_each_cis_interaction = \
            tf.maximum(tf.ones([], tf.float64), (tf.gather(bins_ints_sum, self.cis_training_b2_id) - cis_x_ij + 
                                                 cis_exp_replacement)) / \
            (tf.gather(bins_norm_factors, self.cis_training_b2_id))  # - cis_option_weights)

        b1_vis_after_eliminating_each_trans_interaction = \
            tf.maximum(tf.ones([], tf.float64), (tf.gather(bins_ints_sum, self.trans_training_b1_id) - trans_x_ij + 
                                                 trans_exp_replacement)) / \
            (tf.gather(bins_norm_factors, self.trans_training_b1_id))  # - self.normalized_trans_weight)

        b2_vis_after_eliminating_each_trans_interaction = \
            tf.maximum(tf.ones([], tf.float64), (tf.gather(bins_ints_sum, self.trans_training_b2_id) - trans_x_ij + 
                                                 trans_exp_replacement)) / \
            (tf.gather(bins_norm_factors, self.trans_training_b2_id))  # - self.normalized_trans_weight)

        # calculating logs odds of interactions to choose the ones with odds above the avg
        cis_ints_log_odds = tf.log(cis_xij_prime /
                                   (tf_transformed_vis(self.cis_v_params, b1_vis_after_eliminating_each_cis_interaction) *
                                    tf_transformed_vis(self.cis_v_params, b2_vis_after_eliminating_each_cis_interaction)))
        trans_ints_log_odds = tf.log(trans_xij_prime /
                                     (tf_transformed_vis(self.trans_v_params, b1_vis_after_eliminating_each_trans_interaction) *
                                      tf_transformed_vis(self.trans_v_params, b2_vis_after_eliminating_each_trans_interaction)))

        # calculating avg of logs odds for each bin, for cis and trans interactions
        bin_avg_cis_log_odds = tf_unsorted_segment_avg(cis_ints_log_odds, self.cis_training_b1_id,
            self.max_bin_id + 1)

        bin_avg_trans_log_odds = tf_unsorted_segment_avg(trans_ints_log_odds, self.trans_training_b1_id,
            self.max_bin_id + 1
        )

        # separating interactions greater than avg of both ends and causing in increment of vis
        # (otherwise not eliminating them is an easier condition for being considered significant)
        # + the ones above the avg expectation of their distance!
        cis_p_sig_mask = \
            tf.logical_and(
                tf.greater(cis_x_ij, cis_f_dij),
                tf.logical_and(
                    tf.logical_and(
                        tf.less_equal(b1_vis_after_eliminating_each_cis_interaction, tf.gather(current_vis, self.cis_training_b1_id)),
                        tf.less_equal(b2_vis_after_eliminating_each_cis_interaction, tf.gather(current_vis, self.cis_training_b2_id))
                    ),
                    tf.logical_and(
                        tf.greater_equal(cis_ints_log_odds, tf.gather(bin_avg_cis_log_odds, self.cis_training_b1_id)),
                        tf.greater_equal(cis_ints_log_odds, tf.gather(bin_avg_cis_log_odds, self.cis_training_b2_id))
                    )
                )
            )

        trans_p_sig_mask = \
            tf.logical_and(
                tf.greater(trans_x_ij, tf.exp(self.trans_dist_params)),
                tf.logical_and(
                    tf.logical_and(
                        tf.less_equal(b1_vis_after_eliminating_each_trans_interaction, tf.gather(current_vis, self.trans_training_b1_id)),
                        tf.less_equal(b2_vis_after_eliminating_each_trans_interaction, tf.gather(current_vis, self.trans_training_b2_id))
                    ),
                    tf.logical_and(
                        tf.greater_equal(trans_ints_log_odds, tf.gather(bin_avg_trans_log_odds, self.trans_training_b1_id)),
                        tf.greater_equal(trans_ints_log_odds, tf.gather(bin_avg_trans_log_odds, self.trans_training_b2_id))
                    )
                )
            )

        cis_p_sig_mask.set_shape([None])
        trans_p_sig_mask.set_shape([None])

        # eliminating the effect of significant ints
        def mask_int(b1, b2, rc, mask):
            return tf.boolean_mask(b1, mask), tf.boolean_mask(b2, mask), tf.boolean_mask(rc, mask)
            
        cis_new_sig_b1, cis_new_sig_b2, cis_new_sig_rc = mask_int(
            self.cis_training_b1_id, self.cis_training_b2_id, self.cis_training_rc, cis_p_sig_mask)
        trans_new_sig_b1, trans_new_sig_b2, trans_new_sig_rc = mask_int(
            self.trans_training_b1_id, self.trans_training_b2_id, self.trans_training_rc, trans_p_sig_mask)

        cis_new_insig_b1, cis_new_insig_b2, cis_new_insig_rc = mask_int(
            self.cis_training_b1_id, self.cis_training_b2_id, self.cis_training_rc, tf.logical_not(cis_p_sig_mask))
        trans_new_insig_b1, trans_new_insig_b2, trans_new_insig_rc = mask_int(
            self.trans_training_b1_id, self.trans_training_b2_id, self.trans_training_rc, tf.logical_not(trans_p_sig_mask))

        #calculated_vis = new_bins_insig_sum / tf.maximum(tf.ones([], tf.float64), new_bins_norm_factors)
        new_bins_ints_sum = \
            self.calculate_sum_of_insignificant_interactions_min_one_for_visible_bins(
                cis_new_insig_b1, cis_new_insig_rc, trans_new_insig_b1, trans_new_insig_rc) + \
            self.calculate_sum_of_significant_interactions_expectations(
                cis_new_sig_b1, cis_new_sig_b2, trans_new_sig_b1, trans_new_sig_b2) + \
            self.calculate_sum_of_significant_interactions_expectations(
                self.cis_sig_b1_id, self.cis_sig_b2_id, self.trans_sig_b1_id, self.trans_sig_b2_id
            )

        calculated_vis = new_bins_ints_sum / tf.maximum(tf.ones([], tf.float64), bins_norm_factors)

        nan_check = tf.reduce_any(tf.is_nan(calculated_vis))
        greater_check = tf.reduce_any(tf.greater(calculated_vis, current_vis))

        if not self.silent_mode:
            calculated_vis = tf.cond(
                tf.logical_or(nan_check, greater_check),
                lambda: tf.Print(calculated_vis, [nan_check, greater_check], ' vis debug info'),
                lambda: calculated_vis
            )

        geo_mean = self.calculate_geo_mean(calculated_vis)

        self.pre_transform_norm_factor_assign_state3 = tf.assign(self.pre_transform_norm_factor, 1.0 / geo_mean)

        # we have calculated norm factors before
        self.vis_as_probable_sigs_effects_reduced_assign = part_assign(self.vis, self.active_bin_ids,
                                                                       tf.gather(calculated_vis / geo_mean, self.active_bin_ids))

    # ***************************** Base model runners ***************************** #

    def reset_dist_related_norm_params(self):

        self.reset_cis_dist_func_tie_point()
        self.sess.run([
            self.normalized_cis_after_tie_ints_options_assign,
            self.normalized_trans_ints_options_assign,
            self.normalized_before_tie_cis_weights_assign,
            self.normalized_trans_weight_assign,
        ])

        # resetting base norm factors based on new function parameters
        self.sess.run(self.based_on_open_options_norm_factors_assign)

    def run_model(self, assignment_stage=0):

        s_time = time()

        if assignment_stage == 0:
            self.sess.run([self.vis_as_pure_sum_assign, self.pre_transform_norm_factor_assign_state1])
        elif (assignment_stage == 1) or (not self.remove_sig_from_vis):
            self.sess.run([self.vis_as_options_effected_assign, self.pre_transform_norm_factor_assign_state2])
        elif assignment_stage == 2:
            self.sess.run([self.vis_as_probable_sigs_effects_reduced_assign, self.pre_transform_norm_factor_assign_state3])

        f_time = time()

        if not self.silent_mode:
            print('Updating vis ended in ' + str(f_time - s_time) + ' secs.')

