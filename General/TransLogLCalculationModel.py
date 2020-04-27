import tensorflow as tf
from FuncsDefiner import tf_transformed_vis


class TransLogLCalculator:
    
    def __init__(self, objs_holder):

        self.sess = objs_holder.sess

        # universally known variables
        self.vis = objs_holder.vis
        self.r = objs_holder.trans_r
        self.f_param = objs_holder.trans_f_param
        self.vis_transformer_params = objs_holder.trans_vis_transformer_params

        self.obs_interactions = objs_holder.trans_training_interactions

        # Need to be fed into the model
        self.regularization_lambda = tf.placeholder(tf.float64, ())

        # These need to be evaluated
        self.ll = None

        self.define_model()

    def tf_calculate_ll(self, vis_transformer_params, f_param, r, regularization_lambda):

        # nonzero samples
        vi = tf.gather(self.vis, tf.squeeze(self.obs_interactions[:, 0]))
        vj = tf.gather(self.vis, tf.squeeze(self.obs_interactions[:, 1]))

        vi_p = tf_transformed_vis(vis_transformer_params, vi)
        vj_p = tf_transformed_vis(vis_transformer_params, vj)

        xij = tf.cast(tf.squeeze(self.obs_interactions[:, 2]), tf.float64)

        f_dij = tf.exp(f_param)
        mu_ij = vi_p * vj_p * f_dij
        p_ij = mu_ij / (r + mu_ij)

        non_zero_samples_ll = tf.reduce_mean(tf.lgamma(xij + r - 1) + xij * tf.log(p_ij) + r * tf.log(1 - p_ij))

        ll = -1.0 * tf.lgamma(r - 1) + non_zero_samples_ll - regularization_lambda * tf.pow(r, 2)

        return ll

    def define_model(self):

        self.ll = self.tf_calculate_ll(self.vis_transformer_params, self.f_param, self.r, self.regularization_lambda)

    def run_model(self, regularization_lambda=0):
        return self.sess.run(self.ll, feed_dict={self.regularization_lambda: regularization_lambda})
