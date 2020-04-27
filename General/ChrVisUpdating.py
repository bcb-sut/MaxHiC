import tensorflow as tf
import numpy as np
from time import time
# CARE: To eliminate the effect of scale (smaller f => higher vis => more smaller f!! => eternity!)
# The scale multiplier of f (last cis_f_params) is not included.
from FuncsDefiner import tf_transformed_vis, tf_cis_dist_func, tf_unsorted_segment_avg


# one object of this class per chr (more efficient per thread) so multiple graphs of this type can run in parallel!
class VisUpdater:

    def __init__(self, objs_holder, cores_num, remove_sig_from_vis, silent_mode):

        self.sess = objs_holder.sess
        self.cores_num = cores_num
        self.remove_sig_from_vis = remove_sig_from_vis
        self.silent_mode = silent_mode

        # universally known variables
        self.chrs_bin_ranges = objs_holder.chrs_bin_ranges
        self.vis = objs_holder.vis

        self.active_bin_ids = objs_holder.active_bins
        self.max_bin_id = np.int32(self.sess.run(tf.reduce_max(self.active_bin_ids)))

        self.cis_f_params = objs_holder.cis_f_params
        self.cis_vis_transformer_params = objs_holder.cis_vis_transformer_params

        self.trans_f_param = objs_holder.trans_f_param
        self.trans_vis_transformer_params = objs_holder.trans_vis_transformer_params

        self.cis_training_ints = objs_holder.cis_training_interactions
        self.trans_training_ints = objs_holder.trans_training_interactions

        self.cis_sig_ints = objs_holder.cis_significant_interactions
        self.trans_sig_ints = objs_holder.trans_significant_interactions

        self.max_tie_point = self.sess.run(tf.reduce_max(tf.abs(self.cis_training_ints[:, 0] - self.cis_training_ints[:, 1])))

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

    # ********** for resetting norm factors related to distance params ********** #

    def reset_cis_dist_func_tie_point(self):

        # OK what were we doing? :)
        cis_f_params = self.sess.run(self.cis_f_params)
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
        before_tie_mask = tf.less_equal(tf.cast(tf.abs(self.cis_training_ints[:, 0] - self.cis_training_ints[:, 1]), tf.float64), self.cis_f_tie_point)
        before_tie_mask.set_shape([None])

        cis_ints_before_tie = tf.boolean_mask(self.cis_training_ints, before_tie_mask)
        cis_ints_after_tie = tf.boolean_mask(self.cis_training_ints, tf.logical_not(before_tie_mask))

        before_tie_shape = tf.shape(cis_ints_before_tie)
        number_of_cis_ints_before_tie = before_tie_shape[0]

        after_tie_shape = tf.shape(cis_ints_after_tie)
        number_of_cis_ints_after_tie = after_tie_shape[0]

        trans_shape = tf.shape(self.trans_training_ints)
        number_of_trans_ints = trans_shape[0]

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
                    self.cis_f_params[0] * tf.pow(log_dists, 3) + self.cis_f_params[1] * tf.pow(log_dists, 2) +
                    self.cis_f_params[2] * log_dists + self.cis_f_params[3] - self.cis_f_params[4])], 0)
        normalized_trans_weight = tf.exp(self.trans_f_param - self.cis_f_params[4])

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

    def calculate_option_weight_for_cis_ints(self, cis_ints):

        # the number of interactions
        ints_shape = tf.shape(cis_ints)
        ints_num = ints_shape[0]

        # having a tensor of indices
        ints_indices = tf.range(0, ints_num)

        # separating before and after tie indices, and interactions
        before_tie_mask = tf.less_equal(tf.cast(tf.abs(tf.squeeze(cis_ints[:, 0] - cis_ints[:, 1])), tf.float64), self.cis_f_tie_point)
        before_tie_mask.set_shape([None])

        before_tie_indices = tf.boolean_mask(ints_indices, before_tie_mask)
        after_tie_indices = tf.boolean_mask(ints_indices, tf.logical_not(before_tie_mask))

        # finding weights of options before tie
        before_tie_ints = tf.boolean_mask(cis_ints, before_tie_mask)
        before_tie_ints_weights = tf.gather(self.normalized_before_tie_cis_weights,
                                            tf.squeeze(tf.abs(before_tie_ints[:, 0] - before_tie_ints[:, 1])))

        after_tie_shape = tf.shape(after_tie_indices)
        after_tie_ints_num = after_tie_shape[0]

        all_weights = \
            tf.sparse_to_dense(before_tie_indices, tf.expand_dims(ints_num, 0), before_tie_ints_weights) + \
            tf.sparse_to_dense(after_tie_indices, tf.expand_dims(ints_num, 0), tf.ones(tf.expand_dims(after_tie_ints_num, 0), dtype=tf.float64))

        return all_weights

    def calculate_dist_func(self, dij):
        return tf_cis_dist_func(self.cis_f_params, dij)

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
                                                      ], parallel_iterations=min(self.cores_num, 6))

        expanded_norm_factors = tf.sparse_to_dense(bin_ids, [self.max_bin_id + 1], base_norm_factors, np.float64(0.0))
        self.based_on_open_options_norm_factors_assign = tf.assign(self.based_on_open_options_norm_factors, expanded_norm_factors)

    # ** Auxiliary functions for aggregating the effect of significant interactions ** #

    def calculate_sum_of_insignificant_interactions_min_one_for_visible_bins(self, cis_ints, trans_ints):

        # based on multi nomial distribution's likelihood optimization! each vi ~ sum of observed reads of each bin
        #  :D no need for division! F will support it...
        bins_cis_sum = tf.unsorted_segment_sum(
            tf.concat([tf.squeeze(cis_ints[:, 2]), tf.squeeze(cis_ints[:, 2])], 0),
            tf.concat([tf.squeeze(cis_ints[:, 0]), tf.squeeze(cis_ints[:, 1])], 0),
            self.max_bin_id + 1)

        bins_trans_sum = tf.unsorted_segment_sum(
            tf.concat([tf.squeeze(trans_ints[:, 2]), tf.squeeze(trans_ints[:, 2])], 0),
            tf.concat([tf.squeeze(trans_ints[:, 0]), tf.squeeze(trans_ints[:, 1])], 0),
            self.max_bin_id + 1)
        
        bins_sum = bins_cis_sum + bins_trans_sum

        visible_bins_ints_sum = tf.maximum(tf.ones([], tf.float64),
                                           tf.cast(tf.gather(bins_sum, self.active_bin_ids), tf.float64))

        return tf.sparse_to_dense(self.active_bin_ids, [self.max_bin_id + 1], visible_bins_ints_sum, default_value=tf.zeros([], dtype=tf.float64))

    def calculate_sum_of_significant_interactions_expectations(self, cis_ints, trans_ints):

        # calculating dist function for cis ints
        cis_ints_dist_exp = self.calculate_dist_func(tf.abs(tf.squeeze(cis_ints[:, 0] - cis_ints[:, 1])))
        cis_exp = tf.minimum(tf.cast(tf.squeeze(cis_ints[:, 2]), tf.float64), cis_ints_dist_exp)

        trans_exp = tf.minimum(tf.cast(tf.squeeze(trans_ints[:, 2]), tf.float64), tf.exp(self.trans_f_param))

        # based on multi nomial distribution's likelihood optimization! each vi ~ sum of observed reads of each bin
        #  :D no need for division! F will support it...
        bins_cis_sum = tf.unsorted_segment_sum(
            tf.concat([cis_exp, cis_exp], 0),
            tf.concat([tf.squeeze(cis_ints[:, 0]), tf.squeeze(cis_ints[:, 1])], 0),
            self.max_bin_id + 1)

        bins_trans_sum = tf.unsorted_segment_sum(
            tf.concat([trans_exp, trans_exp], 0),
            tf.concat([tf.squeeze(trans_ints[:, 0]), tf.squeeze(trans_ints[:, 1])], 0),
            self.max_bin_id + 1)

        bins_sum = bins_cis_sum + bins_trans_sum

        return tf.cast(bins_sum, tf.float64)

    def calculate_aggregated_significant_interactions_reduction_effect_from_norm_factors(self, sig_cis, sig_trans):

        # separating cis interactions before tie
        trans_shape = tf.shape(sig_trans)
        cis_shape = tf.shape(sig_cis)

        #trans_shape = tf.Print(trans_shape, [trans_shape], "trans shape")

        cis_ints_weights = tf.cond(
            tf.equal(cis_shape[0], tf.constant(0, dtype=tf.int32, shape=[])),
            lambda: tf.zeros([self.max_bin_id + 1], dtype=tf.float64),
            lambda: self.calculate_option_weight_for_cis_ints(sig_cis)
        )

        cis_effects = tf.cond(
            tf.less_equal(cis_shape[0], tf.constant(1, dtype=tf.int32, shape=[])),
            lambda: tf.zeros([self.max_bin_id + 1], dtype=tf.float64),
            lambda: tf.unsorted_segment_sum(
                tf.concat([cis_ints_weights, cis_ints_weights], 0),
                tf.concat([tf.squeeze(sig_cis[:, 0]), tf.squeeze(sig_cis[:, 1])], 0), self.max_bin_id + 1
            )
        )

        trans_effects = tf.cond(
            tf.less_equal(trans_shape[0], tf.constant(1, dtype=tf.int32, shape=[])),
            lambda: tf.zeros([self.max_bin_id + 1], dtype=tf.float64),
            lambda: tf.unsorted_segment_sum(
                self.normalized_trans_weight * tf.ones(tf.expand_dims(2 * trans_shape[0], 0), dtype=tf.float64),
                tf.concat([tf.squeeze(sig_trans[:, 0]), tf.squeeze(sig_trans[:, 1])], 0), self.max_bin_id + 1)
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
            self.cis_training_ints, self.trans_training_ints)
        geo_mean = self.calculate_geo_mean(bins_ints_sum)
        self.pre_transform_norm_factor_assign_state1 = tf.assign(self.pre_transform_norm_factor, 1.0 / geo_mean)
        self.vis_as_pure_sum_assign = tf.assign(self.vis, bins_ints_sum / geo_mean)

    def define_vis_as_normalized_with_open_options_model(self):

        bins_insig_ints_sum = self.calculate_sum_of_insignificant_interactions_min_one_for_visible_bins(
            self.cis_training_ints, self.trans_training_ints)

        if self.remove_sig_from_vis:
            bins_sig_ints_exp_sum = self.calculate_sum_of_significant_interactions_expectations(self.cis_sig_ints, self.trans_sig_ints)
        else:
            bins_sig_ints_exp_sum = self.calculate_sum_of_insignificant_interactions_min_one_for_visible_bins(
                self.cis_sig_ints, self.trans_sig_ints)

        bins_ints_sum = bins_sig_ints_exp_sum + bins_insig_ints_sum
        bins_norm_factors = self.based_on_open_options_norm_factors

        #sig_reduced_norm_factors = self.based_on_open_options_norm_factors - \
        #    self.calculate_aggregated_significant_interactions_reduction_effect_from_norm_factors(
        #                               self.cis_sig_ints, self.trans_sig_ints)

        calculated_vis = bins_ints_sum / tf.maximum(tf.ones([], dtype=tf.float64), bins_norm_factors)
        geo_mean = self.calculate_geo_mean(calculated_vis)

        self.pre_transform_norm_factor_assign_state2 = tf.assign(self.pre_transform_norm_factor, 1.0 / geo_mean)

        self.vis_as_options_effected_assign = tf.assign(self.vis, calculated_vis / geo_mean)

    def define_recalculate_vis_without_probable_sig_interactions_model(self):

        cis_bin1_id = tf.squeeze(self.cis_training_ints[:, 0])
        cis_bin2_id = tf.squeeze(self.cis_training_ints[:, 1])

        cis_f_dij = self.calculate_dist_func(tf.abs(cis_bin1_id - cis_bin2_id))
        cis_x_ij = tf.cast(tf.squeeze(self.cis_training_ints[:, 2]), tf.float64)
        cis_xij_prime = cis_x_ij / cis_f_dij

        trans_bin1_id = tf.squeeze(self.trans_training_ints[:, 0])
        trans_bin2_id = tf.squeeze(self.trans_training_ints[:, 1])
        
        trans_x_ij = tf.cast(tf.squeeze(self.trans_training_ints[:, 2]), tf.float64)
        trans_xij_prime = trans_x_ij / tf.exp(self.trans_f_param)

        # calculating the current vis (not geo mean normalized)
        bins_insig_ints_sum = self.calculate_sum_of_insignificant_interactions_min_one_for_visible_bins(
            self.cis_training_ints, self.trans_training_ints)
        bins_sig_exp_sum = self.calculate_sum_of_significant_interactions_expectations(self.cis_sig_ints, self.trans_sig_ints)
        bins_ints_sum = bins_sig_exp_sum + bins_insig_ints_sum

        # calculating norm factors for each bin
        #bins_norm_factors = self.based_on_open_options_norm_factors - \
        #    self.calculate_aggregated_significant_interactions_reduction_effect_from_norm_factors(self.cis_sig_ints, self.trans_sig_ints)
        bins_norm_factors = self.based_on_open_options_norm_factors

        current_vis = bins_ints_sum / tf.maximum(tf.ones([], dtype=tf.float64), bins_norm_factors)

        # calculating vis after assuming each interaction significant (reducing the effect of interaction from both ends)
        # the interaction itself would be reduced from sum, and the option weight would be reduced from the norm factor
        cis_exp_replacement = tf.minimum(cis_x_ij, cis_f_dij)
        trans_exp_replacement = tf.minimum(trans_x_ij, tf.exp(self.trans_f_param))

        #cis_option_weights = self.calculate_option_weight_for_cis_ints(self.cis_training_ints)
        b1_vis_after_eliminating_each_cis_interaction = \
            tf.maximum(tf.ones([], tf.float64), (tf.gather(bins_ints_sum, cis_bin1_id) - cis_x_ij + cis_exp_replacement)) / \
            (tf.gather(bins_norm_factors, cis_bin1_id))  # - cis_option_weights)
        b2_vis_after_eliminating_each_cis_interaction = \
            tf.maximum(tf.ones([], tf.float64), (tf.gather(bins_ints_sum, cis_bin2_id) - cis_x_ij + cis_exp_replacement)) / \
            (tf.gather(bins_norm_factors, cis_bin2_id))  # - cis_option_weights)

        b1_vis_after_eliminating_each_trans_interaction = \
            tf.maximum(tf.ones([], tf.float64), (tf.gather(bins_ints_sum, trans_bin1_id) - trans_x_ij + trans_exp_replacement)) / \
            (tf.gather(bins_norm_factors, trans_bin1_id))  # - self.normalized_trans_weight)
        b2_vis_after_eliminating_each_trans_interaction = \
            tf.maximum(tf.ones([], tf.float64), (tf.gather(bins_ints_sum, trans_bin2_id) - trans_x_ij + trans_exp_replacement)) / \
            (tf.gather(bins_norm_factors, trans_bin2_id))  # - self.normalized_trans_weight)

        # calculating logs odds of interactions to choose the ones with odds above the avg
        cis_ints_log_odds = tf.log(cis_xij_prime /
                                   (tf_transformed_vis(self.cis_vis_transformer_params, b1_vis_after_eliminating_each_cis_interaction) *
                                    tf_transformed_vis(self.cis_vis_transformer_params, b2_vis_after_eliminating_each_cis_interaction)))
        trans_ints_log_odds = tf.log(trans_xij_prime /
                                     (tf_transformed_vis(self.trans_vis_transformer_params, b1_vis_after_eliminating_each_trans_interaction) *
                                      tf_transformed_vis(self.trans_vis_transformer_params, b2_vis_after_eliminating_each_trans_interaction)))

        # calculating avg of logs odds for each bin, for cis and trans interactions
        bin_avg_cis_log_odds = tf_unsorted_segment_avg(
            tf.concat([cis_ints_log_odds, cis_ints_log_odds], 0),
            tf.concat([cis_bin1_id, cis_bin2_id], 0),
            self.max_bin_id + 1
        )

        bin_avg_trans_log_odds = tf_unsorted_segment_avg(
            tf.concat([trans_ints_log_odds, trans_ints_log_odds], 0),
            tf.concat([trans_bin1_id, trans_bin2_id], 0),
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
                        tf.less_equal(b1_vis_after_eliminating_each_cis_interaction, tf.gather(current_vis, cis_bin1_id)),
                        tf.less_equal(b2_vis_after_eliminating_each_cis_interaction, tf.gather(current_vis, cis_bin2_id))
                    ),
                    tf.logical_and(
                        tf.greater_equal(cis_ints_log_odds, tf.gather(bin_avg_cis_log_odds, cis_bin1_id)),
                        tf.greater_equal(cis_ints_log_odds, tf.gather(bin_avg_cis_log_odds, cis_bin2_id))
                    )
                )
            )

        trans_p_sig_mask = \
            tf.logical_and(
                tf.greater(trans_x_ij, tf.exp(self.trans_f_param)),
                tf.logical_and(
                    tf.logical_and(
                        tf.less_equal(b1_vis_after_eliminating_each_trans_interaction, tf.gather(current_vis, trans_bin1_id)),
                        tf.less_equal(b2_vis_after_eliminating_each_trans_interaction, tf.gather(current_vis, trans_bin2_id))
                    ),
                    tf.logical_and(
                        tf.greater_equal(trans_ints_log_odds, tf.gather(bin_avg_trans_log_odds, trans_bin1_id)),
                        tf.greater_equal(trans_ints_log_odds, tf.gather(bin_avg_trans_log_odds, trans_bin2_id))
                    )
                )
            )

        cis_p_sig_mask.set_shape([None])
        trans_p_sig_mask.set_shape([None])

        # eliminating the effect of significant ints
        cis_new_sig_ints = tf.boolean_mask(self.cis_training_ints, cis_p_sig_mask)
        trans_new_sig_ints = tf.boolean_mask(self.trans_training_ints, trans_p_sig_mask)

        cis_insig_ints = tf.boolean_mask(self.cis_training_ints, tf.logical_not(cis_p_sig_mask))
        trans_insig_ints = tf.boolean_mask(self.trans_training_ints, tf.logical_not(trans_p_sig_mask))

        #new_bins_insig_sum = self.calculate_sum_of_insignificant_interactions_min_one_for_visible_bins(
        #    cis_insig_ints, trans_insig_ints)
        #new_bins_norm_factors = bins_norm_factors - \
        #    self.calculate_aggregated_significant_interactions_reduction_effect_from_norm_factors(
        #        tf.boolean_mask(self.cis_training_ints, cis_p_sig_mask),
        #        tf.boolean_mask(self.trans_training_ints, trans_p_sig_mask)
        #    )

        #calculated_vis = new_bins_insig_sum / tf.maximum(tf.ones([], tf.float64), new_bins_norm_factors)
        new_bins_ints_sum = \
            self.calculate_sum_of_insignificant_interactions_min_one_for_visible_bins(cis_insig_ints, trans_insig_ints) + \
            self.calculate_sum_of_significant_interactions_expectations(cis_new_sig_ints, trans_new_sig_ints) + \
            self.calculate_sum_of_significant_interactions_expectations(self.cis_sig_ints, self.trans_sig_ints)

        calculated_vis = new_bins_ints_sum / tf.maximum(tf.ones([], tf.float64), bins_norm_factors)

        nan_check = tf.reduce_any(tf.is_nan(calculated_vis))
        greater_check = tf.reduce_any(tf.greater(calculated_vis, current_vis))

        '''
        calculated_vis = tf.cond(
            tf.logical_or(nan_check, greater_check),
            lambda: tf.Print(calculated_vis, [nan_check, greater_check], ' vis debug info'),
            lambda: calculated_vis
        )
        '''

        geo_mean = self.calculate_geo_mean(calculated_vis)

        self.pre_transform_norm_factor_assign_state3 = tf.assign(self.pre_transform_norm_factor, 1.0 / geo_mean)

        # we have calculated norm factors before
        self.vis_as_probable_sigs_effects_reduced_assign = tf.assign(self.vis, calculated_vis / geo_mean)

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

