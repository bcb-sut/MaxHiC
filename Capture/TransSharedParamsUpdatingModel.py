import tensorflow as tf
import numpy as np
from time import time
from TransLogLCalculationModel import TransLogLCalculator
from FuncsDefiner import tf_soft_max, tf_weighted_average, check_nan, get_b_type


class TransSharedParamsUpdater:

    def __init__(self, objs_holder, itype, silent_mode, alpha=np.float64(0.1), beta1=0.9, beta2=0.999, eps=1e-8,
                 mini_batch_size=100000, abs_max_iters=1000, max_iters=50.0, min_iters=50, acc_diff_limit=1e-4, caution_rounds=10,
                 init_regularization_factor=0.001, overflowed_r=200.0):

        self.silent_mode = silent_mode
        self.equal_v_params = (itype[2] == itype[3])

        vtpv = objs_holder.sess.run(objs_holder.v_params[get_b_type(itype, 1)])
        self.n_vis_t_params = list(vtpv.shape)[0]

        self.overflowed_r = overflowed_r

        self.last_alpha = None

        self.min_iters = min_iters
        self.mini_batch_size = mini_batch_size
        self.acc_diff_limit = acc_diff_limit
        self.caution_rounds = caution_rounds
        self.set_alpha_init = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.given_max_iters = max_iters
        if type(max_iters) == float:
            self.factor_max_iter = True
        else:
            self.factor_max_iter = False
        self.init_regularization_factor = init_regularization_factor
        self.abs_max_iters = abs_max_iters

        self.sess = objs_holder.sess
        self.ll_calculator = TransLogLCalculator(objs_holder, itype)

        # universally known variables
        self.vis = objs_holder.vis
        self.r = objs_holder.r[itype]
        self.dist_param = objs_holder.dist_params[itype]
        self.b1_v_params = objs_holder.v_params[get_b_type(itype, 0)]
        self.b2_v_params = objs_holder.v_params[get_b_type(itype, 1)]
        
        self.training_ints = objs_holder.training_ints[itype]

        # These need to be defined later
        self.max_iters = None
        self.total_samples_num = None

        self.non_zero_batch_size = None
        self.non_zero_samples_num = None

        # These need to be evaluated
        self.req_step = None
        self.r_assign = None
        self.f_assign = None
        self.b1_v_params_assign = None
        self.b2_v_params_assign = None

        self.max_ll = None
        self.extra_took_steps = None

        self.define_model()

    def normal_loop_body(self, r, dist_param, b1_v_params, b2_v_params,
                         m_t, v_t, prev_ll, the_number_of_rounds_under_acc_diff, prev_delta,
                         stable_r, stable_dist_param, stable_b1_v_params, stable_b2_v_params,
                         stable_m_t, stable_v_t, stable_prev_ll,
                         stable_the_number_of_rounds_under_acc_diff, stable_prev_delta, step, alpha, max_ll, extra_took_steps):

        # sub-sampling for mini batch
        nzero_mb = tf.cond(tf.equal(self.non_zero_batch_size, self.non_zero_samples_num),
                           lambda: self.training_ints,
                           lambda: tf.gather(self.training_ints,
                                             tf.random_uniform([self.non_zero_batch_size], maxval=self.non_zero_samples_num, dtype=tf.int32)))

        # calculating derivations of nonzero batch
        si = tf.gather(self.vis, tf.squeeze(nzero_mb[:, 0]))
        sj = tf.gather(self.vis, tf.squeeze(nzero_mb[:, 1]))

        # %%
        tsi = tf.pow(si, b1_v_params[0])
        tsj = tf.pow(sj, b2_v_params[0])

        tvi, tsi_share, bv_i_share = tf_soft_max(tsi, b1_v_params[1], True)
        tvj, tsj_share, bv_j_share = tf_soft_max(tsj, b2_v_params[1], True)

        mu_ij = tvi * tvj * tf.exp(dist_param)
        xij = tf.cast(nzero_mb[:, 2], tf.float64)

        # MAIN PARTS DERIVATION

        common_der = (xij - mu_ij) / (r + mu_ij)

        d_r = r * (-2.0 * self.init_regularization_factor * r + (tf.log(r) - tf.digamma(r - 1)) +
                   tf.reduce_mean(tf.digamma(xij + r - 1) - common_der - tf.log(r + mu_ij)))

        # MU PARTS

        # VIS PARAMS

        if self.equal_v_params:
            b1_v_params_der = tf.stack([
                2.0 * tf.sqrt(b1_v_params[0]) * r * tf_weighted_average(
                    common_der * ((tsi_share * tsi * tf.log(si)) + (tsj_share * tsj * tf.log(sj)) /
                                  tf.maximum(tf.constant(1.0, dtype=tf.float64), (tsi_share + tsj_share))), tsi_share + tsj_share
                ),
                2.0 * tf.sqrt(b1_v_params[1]) * r * tf_weighted_average(
                    common_der, bv_i_share + bv_j_share
                )
            ])
            b2_v_params_der = b1_v_params_der
        else:
            b1_v_params_der = tf.stack([
                2.0 * tf.sqrt(b1_v_params[0]) * r * tf_weighted_average(
                    common_der * (tsi * tf.log(si)), tsi_share
                ),
                2.0 * tf.sqrt(b1_v_params[1]) * r * tf_weighted_average(
                    common_der, bv_i_share
                )
            ])
            b2_v_params_der = tf.stack([
                2.0 * tf.sqrt(b2_v_params[0]) * r * tf_weighted_average(
                    common_der * (tsj * tf.log(sj)), tsj_share
                ),
                2.0 * tf.sqrt(b2_v_params[1]) * r * tf_weighted_average(
                    common_der, bv_j_share
                )
            ])

        free_f_drev = r * tf.reduce_mean(common_der)

        # %%
        non_zero_g_t = tf.concat([tf.expand_dims(d_r, 0), tf.expand_dims(free_f_drev, 0),
                                  b1_v_params_der, b2_v_params_der], axis=0)

        # ****************************************
        g_t = non_zero_g_t

        # updating values:
        b1 = tf.constant(self.beta1, dtype=tf.float64)
        b2 = tf.constant(self.beta2, dtype=tf.float64)

        r_t = tf.cast(step + 1, tf.float64)
        r_m = b1 * m_t + (1 - b1) * g_t
        r_v = b2 * v_t + (1 - b2) * tf.pow(g_t, 2)

        a_t = alpha * (tf.sqrt(1 - tf.pow(b2, r_t))) / (1 - tf.pow(b1, r_t))
        delta_vals = a_t * r_m / (tf.sqrt(r_v) + self.eps)

        r_p = tf.log(r) + delta_vals[0]
        new_r = tf.exp(r_p)
        new_dist_param = tf.exp(tf.log(dist_param) + delta_vals[1])

        new_b1_v_params = tf.pow(tf.sqrt(b1_v_params) + delta_vals[2:2 + 2], 2)
        new_b2_v_params = tf.pow(tf.sqrt(b2_v_params) + delta_vals[4:4 + 2], 2)

        new_the_number_of_rounds_under_acc_diff = tf.cond(
            tf.less(tf.reduce_max(tf.abs(delta_vals)), self.acc_diff_limit),
            lambda: the_number_of_rounds_under_acc_diff + 1,
            lambda: tf.constant(0))

        # checking flips
        new_alpha = tf.cond(
            tf.less(tf.reduce_max(delta_vals * prev_delta), tf.constant(0.0, dtype=tf.float64)),
            lambda: 0.1 * alpha,
            lambda: alpha
        )

        new_step = step + 1

        return new_r, new_dist_param, new_b1_v_params, new_b2_v_params, \
               r_m, r_v, prev_ll, new_the_number_of_rounds_under_acc_diff, delta_vals,\
               stable_r, stable_dist_param, stable_b1_v_params, stable_b2_v_params, \
               stable_m_t, stable_v_t, stable_prev_ll,\
               stable_the_number_of_rounds_under_acc_diff, stable_prev_delta, new_step, new_alpha, max_ll, extra_took_steps

    def recheck_model_body(self, r, dist_param, b1_v_params, b2_v_params, m_t, v_t, prev_ll,
                           the_number_of_rounds_under_acc_diff, prev_delta,
                           stable_r, stable_dist_param, stable_b1_v_params, stable_b2_v_params,
                           stable_m_t, stable_v_t, stable_prev_ll,
                           stable_the_number_of_rounds_under_acc_diff, stable_prev_delta, step, alpha, max_ll, extra_took_steps):

        # recalculating ll
        new_ll = self.ll_calculator.tf_calculate_ll(b1_v_params, b2_v_params, dist_param, r, self.init_regularization_factor)
        new_max_ll = tf.maximum(new_ll, max_ll)

        # checking ll
        return tf.cond(
            tf.greater_equal(new_ll, prev_ll),
            lambda: self.normal_loop_body(r, dist_param, b1_v_params, b2_v_params, m_t, v_t, new_ll,
                                          the_number_of_rounds_under_acc_diff, prev_delta,
                                          r, dist_param, b1_v_params, b2_v_params, m_t, v_t, prev_ll,
                                          the_number_of_rounds_under_acc_diff, prev_delta, step, alpha, new_max_ll, extra_took_steps),
            lambda: self.normal_loop_body(stable_r, stable_dist_param, stable_b1_v_params, stable_b2_v_params,
                                          stable_m_t, stable_v_t,
                                          prev_ll, stable_the_number_of_rounds_under_acc_diff, stable_prev_delta,
                                          stable_r, stable_dist_param, stable_b1_v_params, stable_b2_v_params,
                                          stable_m_t, stable_v_t, stable_prev_ll,
                                          stable_the_number_of_rounds_under_acc_diff, stable_prev_delta, step - tf.constant(100, dtype=tf.int32),
                                          alpha * tf.constant(0.3, dtype=tf.float64), new_max_ll, extra_took_steps + tf.constant(100, dtype=tf.int32))
        )

    def general_model_body(self, r, dist_param, b1_v_params, b2_v_params,
                           m_t, v_t, prev_ll, the_number_of_rounds_under_acc_diff, prev_delta,
                           stable_r, stable_dist_param, stable_b1_v_params, stable_b2_v_params,
                           stable_m_t, stable_v_t, stable_prev_ll,
                           stable_the_number_of_rounds_under_acc_diff, stable_prev_delta,
                           step, alpha, max_ll, extra_took_steps):

        # checking out the turn!
        new_r, new_dist_param, new_b1_v_params, new_b2_v_params, new_m_t, new_v_t, new_prev_ll, \
        new_the_number_of_rounds_under_acc_diff, new_prev_delta, \
        new_stable_r, new_stable_dist_param, new_stable_b1_v_params, new_stable_b2_v_params, \
        new_stable_m_t, new_stable_v_t, new_stable_prev_ll, \
        new_stable_the_number_of_rounds_under_acc_diff, new_stable_prev_delta, new_step, new_alpha, new_max_ll, new_extra_took_steps = \
            tf.cond(
                tf.equal(step % 100, tf.constant(0, dtype=tf.int32)),
                lambda: self.recheck_model_body(r, dist_param, b1_v_params, b2_v_params,
                                                m_t, v_t, prev_ll, the_number_of_rounds_under_acc_diff, prev_delta,
                                                stable_r, stable_dist_param, stable_b1_v_params, stable_b2_v_params,
                                                stable_m_t, stable_v_t, stable_prev_ll,
                                                stable_the_number_of_rounds_under_acc_diff, stable_prev_delta, step, alpha, max_ll, extra_took_steps),
                lambda: self.normal_loop_body(r, dist_param, b1_v_params, b2_v_params,
                                              m_t, v_t, prev_ll, the_number_of_rounds_under_acc_diff, prev_delta,
                                              stable_r, stable_dist_param, stable_b1_v_params, stable_b2_v_params,
                                              stable_m_t, stable_v_t, stable_prev_ll,
                                              stable_the_number_of_rounds_under_acc_diff, stable_prev_delta, step, alpha, max_ll, extra_took_steps)
            )

        # checking nan s!
        new_r = check_nan(new_r, 'r')
        new_dist_param = check_nan(new_dist_param, 'trans_dist_params')
        new_b1_v_params = check_nan(new_b1_v_params, 'trans vis params')
        new_b2_v_params = check_nan(new_b2_v_params, 'trans vis params')
        new_m_t = check_nan(new_m_t, 'm')
        new_v_t = check_nan(new_v_t, 'v')

        new_r.set_shape(r.get_shape())
        new_dist_param.set_shape(dist_param.get_shape())
        new_b1_v_params.set_shape(b1_v_params.get_shape())
        new_b2_v_params.set_shape(b2_v_params.get_shape())
        new_m_t.set_shape(m_t.get_shape())
        new_v_t.set_shape(v_t.get_shape())
        new_prev_ll.set_shape(prev_ll.get_shape())
        new_the_number_of_rounds_under_acc_diff.set_shape(the_number_of_rounds_under_acc_diff.get_shape())
        new_prev_delta.set_shape(prev_delta.get_shape())
        new_stable_r.set_shape(stable_r.get_shape())
        new_stable_dist_param.set_shape(stable_dist_param.get_shape())
        new_stable_b1_v_params.set_shape(stable_b1_v_params.get_shape())
        new_stable_b2_v_params.set_shape(stable_b2_v_params.get_shape())
        new_stable_m_t.set_shape(stable_m_t.get_shape())
        new_stable_v_t.set_shape(stable_v_t.get_shape())
        new_stable_prev_ll.set_shape(stable_prev_ll.get_shape())
        new_stable_the_number_of_rounds_under_acc_diff.set_shape(stable_the_number_of_rounds_under_acc_diff.get_shape())
        new_stable_prev_delta.set_shape(stable_prev_delta.get_shape())
        new_step.set_shape(step.get_shape())
        new_alpha.set_shape(alpha.get_shape())
        new_max_ll.set_shape(max_ll.get_shape())
        new_extra_took_steps.set_shape(extra_took_steps.get_shape())

        return new_r, new_dist_param, new_b1_v_params, new_b2_v_params, new_m_t, new_v_t, new_prev_ll, \
            new_the_number_of_rounds_under_acc_diff, new_prev_delta, \
            new_stable_r, new_stable_dist_param, new_stable_b1_v_params, new_stable_b2_v_params, \
            new_stable_m_t, new_stable_v_t, new_stable_prev_ll, new_stable_the_number_of_rounds_under_acc_diff, new_stable_prev_delta, new_step, new_alpha, new_max_ll, new_extra_took_steps

    def model_cond(self, r, dist_param, b1_v_params, b2_v_params, m_t, v_t,
                   prev_ll, the_number_of_rounds_under_acc_diff, prev_delta,
                   stable_r, stable_dist_param, stable_b1_v_params, stable_b2_v_params,
                   stable_m_t, stable_v_t, stable_prev_ll,
                   stable_the_number_of_rounds_under_acc_diff, stable_prev_delta, step, alpha, max_ll, extra_took_steps):

        def has_nan(x):
            return tf.reduce_any(tf.is_nan(x))

        has_nan_params = tf.logical_or(
            has_nan(r),
            tf.logical_or(
                tf.logical_or(has_nan(dist_param), tf.logical_or(has_nan(b1_v_params), has_nan(b2_v_params))),
                tf.logical_or(has_nan(m_t), has_nan(v_t))
            )
        )

        return \
            tf.logical_and(
                tf.logical_not(has_nan_params),
                tf.logical_or(
                    tf.less(step, self.min_iters),
                    tf.logical_and(
                        tf.logical_and(
                            tf.less(the_number_of_rounds_under_acc_diff, self.caution_rounds),
                            tf.less(step, self.max_iters)),
                        tf.logical_or(
                            tf.less(prev_ll, stable_prev_ll),
                            tf.greater(prev_ll - stable_prev_ll, tf.constant(0.000001, dtype=tf.float64))
                        )))
            )

    def define_model(self):

        self.non_zero_samples_num = tf.shape(self.training_ints)[0]
        self.total_samples_num = self.non_zero_samples_num

        mini_batch_size = tf.minimum(self.total_samples_num, self.mini_batch_size)

        if self.factor_max_iter:
            self.max_iters = tf.maximum(self.abs_max_iters, tf.cast(tf.ceil(
                tf.cast(self.given_max_iters, tf.float64) *
                (tf.cast(self.total_samples_num, tf.float64) / tf.cast(mini_batch_size, tf.float64))), tf.int32))
        else:
            self.max_iters = tf.constant(self.given_max_iters)

        self.non_zero_batch_size = tf.cast(mini_batch_size, tf.int32)

        n_total_p = 2 + 2 * self.n_vis_t_params

        new_r, new_dist_param, new_b1_v_params, new_b2_v_params, \
        _, _, _, _, _, _, _, _, _, _, _, _, _, _, req_step, last_alpha, \
            self.max_ll, self.extra_took_steps = \
            tf.while_loop(self.model_cond, self.general_model_body,
                          [
                              self.r,
                              self.dist_param,
                              self.b1_v_params,
                              self.b2_v_params,
                              tf.zeros([n_total_p], dtype=tf.float64),
                              tf.zeros([n_total_p], dtype=tf.float64),
                              tf.constant(-10000.0, dtype=tf.float64),
                              tf.constant(0, dtype=tf.int32),
                              tf.zeros([n_total_p], dtype=tf.float64),
                              self.r,
                              self.dist_param,
                              self.b1_v_params,
                              self.b2_v_params,
                              tf.zeros([n_total_p], dtype=tf.float64),
                              tf.zeros([n_total_p], dtype=tf.float64),
                              tf.constant(-20000.0, dtype=tf.float64),
                              tf.constant(0, dtype=tf.int32),
                              tf.zeros([n_total_p], dtype=tf.float64),
                              tf.constant(0, dtype=tf.int32),
                              self.set_alpha_init,
                              tf.constant(-20000.0, dtype=tf.float64),
                              tf.constant(0, dtype=tf.int32)
                          ], shape_invariants=[
                              tf.TensorShape([]),
                              tf.TensorShape([]),
                              tf.TensorShape([self.n_vis_t_params]),
                              tf.TensorShape([self.n_vis_t_params]),
                              tf.TensorShape([n_total_p]),
                              tf.TensorShape([n_total_p]),
                              tf.TensorShape([]),
                              tf.TensorShape([]),
                              tf.TensorShape([n_total_p]),
                              tf.TensorShape([]),
                              tf.TensorShape([]),
                              tf.TensorShape([self.n_vis_t_params]),
                              tf.TensorShape([self.n_vis_t_params]),
                              tf.TensorShape([n_total_p]),
                              tf.TensorShape([n_total_p]),
                              tf.TensorShape([]),
                              tf.TensorShape([]),
                              tf.TensorShape([n_total_p]),
                              tf.TensorShape([]),
                              tf.TensorShape([]),
                              tf.TensorShape([]),
                              tf.TensorShape([])
                          ],
                          parallel_iterations=1)

        self.r_assign = self.r.assign(new_r)
        self.f_assign = self.dist_param.assign(new_dist_param)
        self.req_step = req_step
        self.last_alpha = last_alpha

        if self.equal_v_params:
            self.b1_v_params_assign = self.b1_v_params.assign(new_b1_v_params)
            self.b2_v_params_assign = tf.zeros([], dtype=tf.int32)
        else:
            self.b1_v_params_assign = self.b1_v_params.assign(new_b1_v_params)
            self.b2_v_params_assign = self.b2_v_params.assign(new_b2_v_params)

    def run_model(self):

        if self.sess.run(tf.size(self.training_ints)) == 0:
            return

        s_time = time()

        init_ll = self.ll_calculator.run_model(self.init_regularization_factor)

        # before run, resetting back the distance function parameter!
        # running one loop
        req_step, _, _, _, _, max_iters, last_alpha, r_new, max_ll, extra_took_steps = \
            self.sess.run([self.req_step, self.r_assign, self.f_assign,
                           self.b1_v_params_assign, self.b2_v_params_assign, self.max_iters,
                           self.last_alpha, self.r, self.max_ll, self.extra_took_steps])

        final_ll = self.ll_calculator.run_model(self.init_regularization_factor)

        steps_str = str(req_step)
        if req_step == max_iters:
            steps_str += ' (max)'

        f_time = time()

        if not self.silent_mode:
            print('Calculating trans shared params ended in %s (+ %d extra burned steps) and in %.2f secs (ll %.4f -> %.4f)\nlast_alpha: %g, max_ll: %.4f' %
                  (steps_str, extra_took_steps, f_time - s_time, init_ll, final_ll, last_alpha, max_ll))

        return None
