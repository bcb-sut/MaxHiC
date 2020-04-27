import tensorflow as tf
import numpy as np
from time import time
from CisLogLCalculationModel import CisLogLCalculator
from FuncsDefiner import tf_soft_max, tf_cis_var_func, tf_weighted_average, check_nan, check_inf


class CisSharedParamsUpdater:

    def __init__(self, objs_holder, min_dist, max_dist, min_rc, silent_mode,
                 alpha=np.float64(0.1), beta1=0.9, beta2=0.999, eps=1e-8,
                 mini_batch_size=100000, abs_max_iters=1000, max_iters=50.0, min_iters=10, acc_diff_limit=1e-4, caution_rounds=10,
                 init_regularization_factor=0.001, overflowed_r=200.0):

        self.silent_mode = silent_mode
        fpv, vtpv = objs_holder.sess.run([objs_holder.cis_f_params, objs_holder.cis_vis_transformer_params])
        self.n_f_params = list(fpv.shape)[0]
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
        self.ll_calculator = CisLogLCalculator(objs_holder)

        # universally known variables
        self.vis = objs_holder.vis
        self.r = objs_holder.cis_r
        self.f_params = objs_holder.cis_f_params
        self.vis_transformer_params = objs_holder.cis_vis_transformer_params

        active_mask = tf.logical_and(
            objs_holder.cis_training_interactions[:, 2] >= min_rc,
            tf.logical_and(
                tf.abs(objs_holder.cis_training_interactions[:, 0] - objs_holder.cis_training_interactions[:, 1]) <= max_dist,
                tf.abs(objs_holder.cis_training_interactions[:, 0] - objs_holder.cis_training_interactions[:, 1]) >= min_dist
            )
        )
        active_mask.set_shape([None])
        self.insig_interactions = tf.boolean_mask(objs_holder.cis_training_interactions, active_mask)

        # These need to be defined later
        self.max_iters = None
        self.total_samples_num = None

        self.non_zero_batch_size = None
        self.non_zero_samples_num = None

        # These need to be evaluated
        self.req_step = None
        self.r_assign = None
        self.f_assign = None
        self.hill_assign = None

        self.max_ll = None
        self.extra_took_steps = None

        self.define_model()

    def normal_loop_body(self, r, f_params, vis_transformer_params, m_t, v_t, prev_ll, the_number_of_rounds_under_acc_diff, prev_delta,
                         stable_r, stable_f_params, stable_vis_transformer_params, stable_m_t, stable_v_t, stable_prev_ll,
                         stable_the_number_of_rounds_under_acc_diff, stable_prev_delta, step, alpha, max_ll, extra_took_steps):

        # sub-sampling for mini batch
        nzero_mb = tf.cond(tf.equal(self.non_zero_batch_size, self.non_zero_samples_num),
                           lambda: self.insig_interactions,
                           lambda: tf.gather(self.insig_interactions,
                                             tf.random_uniform([self.non_zero_batch_size], maxval=self.non_zero_samples_num, dtype=tf.int32)))

        # calculating derivations of nonzero batch
        si = tf.gather(self.vis, tf.squeeze(nzero_mb[:, 0]))
        sj = tf.gather(self.vis, tf.squeeze(nzero_mb[:, 1]))

        xij = tf.cast(nzero_mb[:, 2], tf.float64)

        d_ij = tf.cast(tf.abs(nzero_mb[:, 0] - nzero_mb[:, 1]), tf.float64)
        ln_dij = tf.log(d_ij)

        # %%
        bv = vis_transformer_params[1]
        spower = vis_transformer_params[0]

        tsi = tf.pow(si, spower)
        tsj = tf.pow(sj, spower)

        tvi, tsi_share, bv_i_share = tf_soft_max(tsi, bv, True)
        tvj, tsj_share, bv_j_share = tf_soft_max(tsj, bv, True)

        f_var_ij = tf_cis_var_func(f_params, d_ij)
        f_free = tf.exp(f_params[4])

        f_d_ij, var_f_share, free_f_share = tf_soft_max(f_var_ij, f_free, True)
        mu_ij = tvi * tvj * f_d_ij

        common_der = (xij - mu_ij) / (r + mu_ij)

        r_der = r * (-2.0 * self.init_regularization_factor * r + (tf.log(r) - tf.digamma(r - 1)) +
                     tf.reduce_mean(tf.digamma(xij + r - 1) - common_der - tf.log(r + mu_ij)))

        # %%
        vis_transformer_params_der = tf.stack([
            2.0 * tf.sqrt(spower) * r * tf_weighted_average(
                common_der * ((tsi_share * tsi * tf.log(si)) + (tsj_share * tsj * tf.log(sj)) / (tsi_share + tsj_share)), tsi_share + tsj_share
            ),
            2.0 * tf.sqrt(bv) * r * tf_weighted_average(
                common_der, bv_i_share + bv_j_share
            )
        ])

        f_var_common_der = common_der * f_var_ij / f_d_ij

        f_var_common_der = check_nan(f_var_common_der, 'f_var_common_der')
        ln_dij = check_nan(ln_dij, 'ln_dij')
        var_f_share = check_nan(var_f_share, 'var_f_share')
        var_f_share = check_inf(var_f_share, 'var_f_share')
        f_params = check_nan(f_params, 'f_params')

        f_drev = tf.stack([
            f_params[0] * r * tf_weighted_average(f_var_common_der * tf.pow(ln_dij, 3), var_f_share),
            r * tf_weighted_average(f_var_common_der * tf.pow(ln_dij, 2), var_f_share),
            r * tf_weighted_average(f_var_common_der * ln_dij, var_f_share),
            r * tf_weighted_average(f_var_common_der, var_f_share)
        ])

        f_drev = check_nan(f_drev, 'f_drev')

        free_f_drev = r * tf_weighted_average((common_der / f_d_ij), free_f_share)

        # %%
        non_zero_g_t = tf.concat([tf.expand_dims(r_der, 0), f_drev, tf.expand_dims(free_f_drev, 0), vis_transformer_params_der], axis=0)

        # ****************************************
        g_t = non_zero_g_t

        g_t = check_nan(g_t, 'g_t')

        # updating values:
        b1 = tf.constant(self.beta1, dtype=tf.float64)
        b2 = tf.constant(self.beta2, dtype=tf.float64)

        r_t = tf.cast(step + 1, tf.float64)
        r_m = b1 * m_t + (1 - b1) * g_t
        r_v = b2 * v_t + (1 - b2) * tf.pow(g_t, 2)

        a_t = alpha * (tf.sqrt(1 - tf.pow(b2, r_t))) / (1 - tf.pow(b1, r_t))
        delta_vals = a_t * r_m / (tf.sqrt(r_v) + self.eps)

        delta_vals = check_nan(delta_vals, 'delta_vals')

        r_p = tf.log(r) + delta_vals[0]
        new_r = tf.exp(r_p)

        a3_p = tf.log(-1.0 * f_params[0]) + delta_vals[1]
        new_f_params = tf.stack([-1.0 * tf.exp(a3_p), f_params[1] + delta_vals[2], f_params[2] + delta_vals[3], f_params[3] + delta_vals[4],
                                 f_params[4] + delta_vals[5]])

        # %%
        new_vis_transformer_params = tf.pow(tf.sqrt(vis_transformer_params) + delta_vals[6:6 + self.n_vis_t_params], 2)

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

        new_step = step + tf.constant(1, dtype=tf.int32)

        return new_r, new_f_params, new_vis_transformer_params, r_m, r_v, prev_ll, \
            new_the_number_of_rounds_under_acc_diff, delta_vals, \
            stable_r, stable_f_params, stable_vis_transformer_params, stable_m_t, stable_v_t, stable_prev_ll,\
            stable_the_number_of_rounds_under_acc_diff, stable_prev_delta, new_step, new_alpha, max_ll, extra_took_steps

    def recheck_model_body(self, r, f_params, vis_transformer_params, m_t, v_t, prev_ll, the_number_of_rounds_under_acc_diff, prev_delta,
                           stable_r, stable_f_params, stable_vis_transformer_params, stable_m_t, stable_v_t, stable_prev_ll,
                           stable_the_number_of_rounds_under_acc_diff, stable_prev_delta, step, alpha, max_ll, extra_took_steps):

        # recalculating ll
        new_ll = self.ll_calculator.tf_calculate_ll(vis_transformer_params, f_params, r, self.init_regularization_factor)

        new_max_ll = tf.maximum(new_ll, max_ll)

        # checking ll
        return tf.cond(
            tf.greater_equal(new_ll, prev_ll),
            lambda: self.normal_loop_body(r, f_params, vis_transformer_params, m_t, v_t, new_ll,
                                          the_number_of_rounds_under_acc_diff, prev_delta,
                                          r, f_params, vis_transformer_params, m_t, v_t, prev_ll,
                                          the_number_of_rounds_under_acc_diff, prev_delta, step, alpha, new_max_ll, extra_took_steps),
            lambda: self.normal_loop_body(stable_r, stable_f_params, stable_vis_transformer_params, stable_m_t, stable_v_t,
                                          prev_ll, stable_the_number_of_rounds_under_acc_diff, stable_prev_delta,
                                          stable_r, stable_f_params, stable_vis_transformer_params, stable_m_t, stable_v_t, stable_prev_ll,
                                          stable_the_number_of_rounds_under_acc_diff, stable_prev_delta, step - tf.constant(100, dtype=tf.int32),
                                          alpha * tf.constant(0.3, dtype=tf.float64), new_max_ll, extra_took_steps + tf.constant(100, dtype=tf.int32))
        )

    def general_model_body(self, r, f_params, vis_transformer_params, m_t, v_t, prev_ll, the_number_of_rounds_under_acc_diff, prev_delta,
                           stable_r, stable_f_params, stable_vis_transformer_params, stable_m_t, stable_v_t, stable_prev_ll,
                           stable_the_number_of_rounds_under_acc_diff, stable_prev_delta,
                           step, alpha, max_ll, extra_took_steps):

        # checking out the turn!
        new_r, new_f_params, new_vis_transformer_params, new_m_t, new_v_t, new_prev_ll, \
        new_the_number_of_rounds_under_acc_diff, new_prev_delta, \
        new_stable_r, new_stable_f_params, new_stable_vis_transformer_params, new_stable_m_t, new_stable_v_t, new_stable_prev_ll, \
        new_stable_the_number_of_rounds_under_acc_diff, new_stable_prev_delta, new_step, new_alpha, new_max_ll, new_extra_took_steps = \
            tf.cond(
                tf.equal(step % 100, tf.constant(0, dtype=tf.int32)),
                lambda: self.recheck_model_body(r, f_params, vis_transformer_params, m_t, v_t, prev_ll, the_number_of_rounds_under_acc_diff, prev_delta,
                                        stable_r, stable_f_params, stable_vis_transformer_params, stable_m_t, stable_v_t, stable_prev_ll,
                                        stable_the_number_of_rounds_under_acc_diff, stable_prev_delta, step, alpha, max_ll, extra_took_steps),
                lambda: self.normal_loop_body(r, f_params, vis_transformer_params, m_t, v_t, prev_ll, the_number_of_rounds_under_acc_diff, prev_delta,
                                      stable_r, stable_f_params, stable_vis_transformer_params, stable_m_t, stable_v_t, stable_prev_ll,
                                      stable_the_number_of_rounds_under_acc_diff, stable_prev_delta, step, alpha, max_ll, extra_took_steps)
            )

        # checking nan s!
        new_r = check_nan(new_r, 'r')
        new_f_params = check_nan(new_f_params, 'cis_f_params')
        new_vis_transformer_params = check_nan(new_vis_transformer_params, 'cis vis params')
        new_m_t = check_nan(new_m_t, 'm')
        new_v_t = check_nan(new_v_t, 'v')

        new_r.set_shape(r.get_shape())
        new_f_params.set_shape(f_params.get_shape())
        new_vis_transformer_params.set_shape(vis_transformer_params.get_shape())
        new_m_t.set_shape(m_t.get_shape())
        new_v_t.set_shape(v_t.get_shape())
        new_prev_ll.set_shape(prev_ll.get_shape())
        new_the_number_of_rounds_under_acc_diff.set_shape(the_number_of_rounds_under_acc_diff.get_shape())
        new_prev_delta.set_shape(prev_delta.get_shape())
        new_stable_r.set_shape(stable_r.get_shape())
        new_stable_f_params.set_shape(stable_f_params.get_shape())
        new_stable_vis_transformer_params.set_shape(stable_vis_transformer_params.get_shape())
        new_stable_m_t.set_shape(stable_m_t.get_shape())
        new_stable_v_t.set_shape(stable_v_t.get_shape())
        new_stable_prev_ll.set_shape(stable_prev_ll.get_shape())
        new_stable_the_number_of_rounds_under_acc_diff.set_shape(stable_the_number_of_rounds_under_acc_diff.get_shape())
        new_stable_prev_delta.set_shape(stable_prev_delta.get_shape())
        new_step.set_shape(step.get_shape())
        new_alpha.set_shape(alpha.get_shape())
        new_max_ll.set_shape(max_ll.get_shape())
        new_extra_took_steps.set_shape(extra_took_steps.get_shape())

        return new_r, new_f_params, new_vis_transformer_params, new_m_t, new_v_t, new_prev_ll, \
            new_the_number_of_rounds_under_acc_diff, new_prev_delta, \
            new_stable_r, new_stable_f_params, new_stable_vis_transformer_params, new_stable_m_t, new_stable_v_t, new_stable_prev_ll, \
            new_stable_the_number_of_rounds_under_acc_diff, new_stable_prev_delta, new_step, new_alpha, new_max_ll, new_extra_took_steps

    def model_cond(self, r, f_params, vis_transformer_params, m_t, v_t, prev_ll, the_number_of_rounds_under_acc_diff, prev_delta,
                   stable_r, stable_f_params, stable_vis_transformer_params, stable_m_t, stable_v_t, stable_prev_ll,
                   stable_the_number_of_rounds_under_acc_diff, stable_prev_delta,
                   step, alpha, max_ll, extra_took_steps):

        def has_nan(x):
            return tf.reduce_any(tf.is_nan(x))

        has_nan_params = tf.logical_or(
            has_nan(r),
            tf.logical_or(
                tf.logical_or(has_nan(f_params), has_nan(vis_transformer_params)),
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

        self.non_zero_samples_num = tf.shape(self.insig_interactions)[0]
        self.total_samples_num = self.non_zero_samples_num

        mini_batch_size = tf.minimum(self.total_samples_num, self.mini_batch_size)

        if self.factor_max_iter:
            self.max_iters = tf.maximum(self.abs_max_iters, tf.cast(tf.ceil(
                tf.cast(self.given_max_iters, tf.float64) *
                (tf.cast(self.total_samples_num, tf.float64) / tf.cast(mini_batch_size, tf.float64))), tf.int32))
        else:
            self.max_iters = tf.constant(self.given_max_iters)

        self.non_zero_batch_size = tf.cast(mini_batch_size, tf.int32)

        n_total_p = 1 + self.n_vis_t_params + self.n_f_params

        new_r, new_f_params, new_vis_transformer_params, _, _, _, _, _, _, _, _, _, _, _, _, _, req_step, last_alpha, \
            self.max_ll, self.extra_took_steps = \
            tf.while_loop(self.model_cond, self.general_model_body,
                          [
                              self.r,
                              self.f_params,
                              self.vis_transformer_params,
                              tf.zeros([n_total_p], dtype=tf.float64),
                              tf.zeros([n_total_p], dtype=tf.float64),
                              tf.constant(-10000.0, dtype=tf.float64),
                              tf.constant(0, dtype=tf.int32),
                              tf.zeros([n_total_p], dtype=tf.float64),
                              self.r,
                              self.f_params,
                              self.vis_transformer_params,
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
                              tf.TensorShape([self.n_f_params]),
                              tf.TensorShape([self.n_vis_t_params]),
                              tf.TensorShape([n_total_p]),
                              tf.TensorShape([n_total_p]),
                              tf.TensorShape([]),
                              tf.TensorShape([]),
                              tf.TensorShape([n_total_p]),
                              tf.TensorShape([]),
                              tf.TensorShape([self.n_f_params]),
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

        # calculating geometric mean of transformed visibilities of active bins!
        self.r_assign = self.r.assign(new_r)
        self.f_assign = self.f_params.assign(new_f_params)
        self.hill_assign = self.vis_transformer_params.assign(new_vis_transformer_params)
        self.req_step = req_step
        self.last_alpha = last_alpha

    def run_model(self):

        if self.sess.run(tf.size(self.insig_interactions)) == 0:
            return

        s_time = time()

        # keeping a backup of parameters
        init_ll = self.ll_calculator.run_model(self.init_regularization_factor)

        # running one loop
        req_step, _, _, _, max_iters, last_alpha, max_ll, extra_took_steps = \
            self.sess.run([self.req_step, self.r_assign, self.f_assign, self.hill_assign, self.max_iters, self.last_alpha, self.max_ll, self.extra_took_steps])

        final_ll = self.ll_calculator.run_model(self.init_regularization_factor)

        steps_str = str(req_step)
        if req_step == max_iters:
            steps_str += ' (max)'

        f_time = time()

        if not self.silent_mode:
            print('Calculating cis shared params ended in %s (+ %d extra burned steps) and in %.2f secs (ll %.4f -> %.4f)' %
                  (steps_str, extra_took_steps, f_time - s_time, init_ll, final_ll))
