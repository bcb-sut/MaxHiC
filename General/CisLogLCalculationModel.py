import tensorflow as tf
from FuncsDefiner import tf_transformed_vis, tf_cis_dist_func, check_nan, check_inf


class CisLogLCalculator:

    def __init__(self, objs_holder):

        self.sess = objs_holder.sess

        # universally known variables
        self.vis = objs_holder.vis
        self.r = objs_holder.cis_r
        self.f_params = objs_holder.cis_f_params
        self.vis_transformer_params = objs_holder.cis_vis_transformer_params

        self.obs_interactions = objs_holder.cis_training_interactions

        # Need to be fed into the model
        self.regularization_lambda = tf.placeholder(tf.float64, ())

        # These need to be evaluated
        self.ll = None

        self.define_model()

    def tf_calculate_ll(self, vis_transformer_params, f_params, r, regularization_lambda):

        # nonzero samples
        vi = tf.gather(self.vis, tf.squeeze(self.obs_interactions[:, 0]))
        vj = tf.gather(self.vis, tf.squeeze(self.obs_interactions[:, 1]))

        dij = tf.squeeze(tf.abs(self.obs_interactions[:, 0] - self.obs_interactions[:, 1]))

        vi_p = tf_transformed_vis(vis_transformer_params, vi)
        vj_p = tf_transformed_vis(vis_transformer_params, vj)

        xij = tf.cast(tf.squeeze(self.obs_interactions[:, 2]), tf.float64)
        f_dij = tf_cis_dist_func(f_params, dij)

        mu_ij = vi_p * vj_p * f_dij

        p_ij = mu_ij / (r + mu_ij)

        non_zero_samples_ll = tf.reduce_mean(tf.lgamma(xij + r - 1) + xij * tf.log(p_ij) + r * tf.log(1 - p_ij))

        non_zero_samples_ll = check_inf(non_zero_samples_ll, 'nz_ll')

        ll = -1.0 * tf.lgamma(r - 1) + non_zero_samples_ll - regularization_lambda * tf.pow(r, 2)

        ll = check_inf(ll, 'll')

        return ll

    def define_model(self):

        self.ll = self.tf_calculate_ll(self.vis_transformer_params, self.f_params, self.r, self.regularization_lambda)

    def run_model(self, regularization_lambda=0):
        return self.sess.run(self.ll, feed_dict={self.regularization_lambda: regularization_lambda})
