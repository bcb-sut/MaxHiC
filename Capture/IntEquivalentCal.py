import tensorflow as tf
from FuncsDefiner import get_b_type, tf_cis_ints_expected_vals, tf_trans_ints_expected_vals


class IntEquivalentCal:

    def __init__(self, objs_holder):

        self.dist_params = objs_holder.dist_params
        self.v_params = objs_holder.v_params

        self.vis = objs_holder.vis

    def convert(self, ints, src_type, dest_type):

        if 'c_' in src_type:
            src_exp = tf_cis_ints_expected_vals(ints, self.vis, self.dist_params[src_type],
                                                self.v_params[get_b_type(src_type, 0)], self.v_params[get_b_type(src_type, 1)])
            dest_exp = tf_cis_ints_expected_vals(ints, self.vis, self.dist_params[dest_type],
                                                 self.v_params[get_b_type(dest_type, 0)], self.v_params[get_b_type(dest_type, 1)])
        else:
            src_exp = tf_trans_ints_expected_vals(ints, self.vis, self.dist_params[src_type],
                                                  self.v_params[get_b_type(src_type, 0)], self.v_params[get_b_type(src_type, 1)])
            dest_exp = tf_trans_ints_expected_vals(ints, self.vis, self.dist_params[dest_type],
                                                   self.v_params[get_b_type(dest_type, 0)], self.v_params[get_b_type(dest_type, 1)])

        return tf.cast(ints[:, 2], tf.float64) * dest_exp / src_exp

