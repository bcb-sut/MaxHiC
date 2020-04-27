import tensorflow as tf
from FuncsDefiner import tf_cis_ints_expected_vals, get_b_type


class CisLogLCalculator:

    def __init__(self, objs_holder, itype):

        self.sess = objs_holder.sess

        # universally known variables
        self.vis = objs_holder.vis
        self.r = objs_holder.r[itype]
        self.dist_params = objs_holder.dist_params[itype]
        self.b1_v_params = objs_holder.v_params[get_b_type(itype, 0)]
        self.b2_v_params = objs_holder.v_params[get_b_type(itype, 1)]
        self.training_ints = objs_holder.training_ints[itype]

        # Need to be fed into the model
        self.regularization_lambda = tf.placeholder(tf.float64, ())

        # These need to be evaluated
        self.ll = None

        self.define_model()

    def tf_calculate_ll(self, b1_v_params, b2_v_params, dist_params, r, regularization_lambda):

        if b1_v_params is None:
            b1_v_params = self.b1_v_params
            
        if b2_v_params is None:
            b2_v_params = self.b2_v_params

        xij = tf.cast(tf.squeeze(self.training_ints[:, 2]), tf.float64)
        mu_ij = tf_cis_ints_expected_vals(self.training_ints, self.vis, dist_params, b1_v_params, b2_v_params)

        p_ij = mu_ij / (r + mu_ij)

        non_zero_samples_ll = tf.reduce_mean(tf.lgamma(xij + r - 1) + xij * tf.log(p_ij) + r * tf.log(1 - p_ij))
        ll = -1.0 * tf.lgamma(r - 1) + non_zero_samples_ll - regularization_lambda * tf.pow(r, 2)

        return ll

    def define_model(self):

        self.ll = self.tf_calculate_ll(self.b1_v_params, self.b2_v_params, self.dist_params, self.r, self.regularization_lambda)

    def run_model(self, regularization_lambda=0):
        return self.sess.run(self.ll, feed_dict={self.regularization_lambda: regularization_lambda})
