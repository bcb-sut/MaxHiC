import numpy as np
import tensorflow as tf
from CisPValCalculationModel import CisPValCalculator
from TransPValCalculationModel import TransPValCalculator


class TrainingInteractionsSeparator:

    def __init__(self, objs_holder, itype, pval_limit, silent_mode):
        
        self.itype = itype
        
        self.ints = objs_holder.tf_ints[itype]
        self.on_mask = objs_holder.tf_ints_on_mask[itype]

        # ranges also need to be modified
        self.sig_end = objs_holder.ints_sig_end[itype]
        self.training_end = objs_holder.ints_training_end[itype]

        self.silent_mode = silent_mode
        self.limit = np.log(pval_limit)
        self.sess = objs_holder.sess

        if itype.startswith('c'):
            self.pval_calculator = CisPValCalculator(objs_holder, itype)
        else:
            self.pval_calculator = TransPValCalculator(objs_holder, itype)

        # ASSIGNMENTS

        self.ints_assign = None
        self.on_mask_assign = None

        # ranges also need to be modified
        self.sig_end_assign = None
        self.training_end_assign = None

        self.on_sig_percentage = None
        self.off_sig_percentage = None
        
        self.define_separation_model()

    def define_separation_model(self):

        log_pvals = self.pval_calculator.run_model(self.ints)

        # significant masks
        sig_masks = tf.less_equal(log_pvals, self.limit)

        # now finding new ranges! SIG-ON, SIG-OFF, INSIG-ON, NOISE-ON, INSIG-OFF
        all_indices = tf.range(tf.shape(self.ints)[0])

        sig_on_mask = tf.logical_and(sig_masks, self.on_mask)
        sig_off_mask = tf.logical_and(sig_masks, tf.logical_not(self.on_mask))
        insig_on_mask = tf.logical_and(tf.logical_not(sig_masks), self.on_mask)
        insig_off_mask = tf.logical_and(tf.logical_not(sig_masks), tf.logical_not(self.on_mask))

        sig_on_mask.set_shape([None])
        sig_off_mask.set_shape([None])
        insig_on_mask.set_shape([None])
        insig_off_mask.set_shape([None])

        sig_on = tf.boolean_mask(all_indices, sig_on_mask)
        sig_off = tf.boolean_mask(all_indices, sig_off_mask)
        insig_on = tf.boolean_mask(all_indices, insig_on_mask)
        insig_off = tf.boolean_mask(all_indices, insig_off_mask)

        new_order = tf.concat([sig_on, sig_off, insig_on, insig_off], axis=0)
        
        # assignments
        self.ints_assign = tf.assign(self.ints, tf.gather(self.ints, new_order))
        self.on_mask_assign = tf.assign(self.on_mask, tf.gather(self.on_mask, new_order))
        
        self.sig_end_assign = tf.assign(self.sig_end, tf.shape(sig_on)[0] + tf.shape(sig_off)[0])
        self.training_end_assign = tf.assign(self.training_end, 
                                             tf.shape(sig_on)[0] + tf.shape(sig_off)[0] + tf.shape(insig_on)[0])

        self.on_sig_percentage = tf.cast(tf.shape(sig_on)[0], tf.float64) / \
                                 tf.cast(tf.maximum(1, tf.shape(sig_on)[0] + tf.shape(insig_on)[0]), tf.float64)
        self.off_sig_percentage = tf.cast(tf.shape(sig_off)[0], tf.float64) / \
                                  tf.cast(tf.maximum(1, tf.shape(sig_off)[0] + tf.shape(insig_off)[0]), tf.float64)

    def separate_training_interactions(self):

        # IN RANGE INTS
        _, _, _, _, sig_on_per, sig_off_per = self.sess.run([
            self.ints_assign, self.on_mask_assign, self.sig_end_assign, self.training_end_assign,
            self.on_sig_percentage, self.off_sig_percentage])

        # finding the percent of unreals for debugging
        if not self.silent_mode:
            print('~~~~~~~')
            print('Percent of sig interactions for %s:' % self.itype)
            print('On: %.4f' % sig_on_per)
            print('Off: %.4f' % sig_off_per)
            print('~~~~~~~')
