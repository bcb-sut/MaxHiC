import numpy as np
import tensorflow as tf
from CisPValCalculationModel import CisPValCalculator
from TransPValCalculationModel import TransPValCalculator


class TrainingInteractionsSeparator:

    def __init__(self, objs_holder, pval_limit, silent_mode):

        self.silent_mode = silent_mode

        self.limit = np.log(pval_limit)

        self.cis_training_ints_var = objs_holder.cis_training_interactions
        self.trans_training_ints_var = objs_holder.trans_training_interactions

        self.cis_sig_ints_var = objs_holder.cis_significant_interactions
        self.trans_sig_ints_var = objs_holder.trans_significant_interactions

        self.sess = objs_holder.sess

        self.cis_pval_calculator = CisPValCalculator(objs_holder)
        self.trans_pval_calculator = TransPValCalculator(objs_holder)

    def separate_training_interactions(self, cis_interactions, trans_interactions):

        cis_log_pvals = self.cis_pval_calculator.run_model(cis_interactions)
        trans_log_pvals = self.trans_pval_calculator.run_model(trans_interactions)

        # significant masks
        cis_sig_masks = (cis_log_pvals <= self.limit)
        trans_sig_masks = (trans_log_pvals <= self.limit)

        cis_sig_ints = cis_interactions[cis_sig_masks, :]
        trans_sig_ints = trans_interactions[trans_sig_masks, :]

        cis_insig_ints = cis_interactions[np.logical_not(cis_sig_masks), :]
        trans_insig_ints = trans_interactions[np.logical_not(trans_sig_masks), :]

        cis_sig_ph = tf.placeholder(tf.int32, shape=cis_sig_ints.shape)
        trans_sig_ph = tf.placeholder(tf.int32, shape=trans_sig_ints.shape)
        cis_insig_ph = tf.placeholder(tf.int32, shape=cis_insig_ints.shape)
        trans_insig_ph = tf.placeholder(tf.int32, shape=trans_insig_ints.shape)

        self.sess.run([
            tf.assign(self.cis_training_ints_var, cis_insig_ph, validate_shape=False),
            tf.assign(self.trans_training_ints_var, trans_insig_ph, validate_shape=False),
            tf.assign(self.cis_sig_ints_var, cis_sig_ph, validate_shape=False),
            tf.assign(self.trans_sig_ints_var, trans_sig_ph, validate_shape=False),
        ],
            feed_dict={
                cis_insig_ph: cis_insig_ints,
                trans_insig_ph: trans_insig_ints,
                cis_sig_ph: cis_sig_ints,
                trans_sig_ph: trans_sig_ints
            })

        # finding the percent of unreals for debugging
        if not self.silent_mode:
            print('Percent of cis training interactions: %.2f' % (100.0 * cis_insig_ints.size / cis_interactions.size,))
            print('Percent of trans training interactions: %.2f' % (100.0 * trans_insig_ints.size / trans_interactions.size,))
