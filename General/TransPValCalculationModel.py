import tensorflow as tf
import numpy as np
from scipy.stats import nbinom
from PValCalculationModel import nbinom_logsf
from FuncsDefiner import np_transformed_vis


class TransPValCalculator:

    def __init__(self, objs_holder):

        self.sess = objs_holder.sess

        # universally known variables
        self.vis = objs_holder.vis
        self.r = objs_holder.trans_r
        self.f_param = objs_holder.trans_f_param
        self.vis_transformer_params = objs_holder.trans_vis_transformer_params

    def run_model(self, interactions):

        r_val, f_param_val, vis_transformer_params_vals, vis_val = \
            self.sess.run([self.r, self.f_param, self.vis_transformer_params, self.vis])

        vi_p = np_transformed_vis(vis_transformer_params_vals, vis_val[interactions[:, 0]])
        vj_p = np_transformed_vis(vis_transformer_params_vals, vis_val[interactions[:, 1]])

        # calculating expected value
        exp_interactions = vi_p * vj_p * np.exp(f_param_val)

        log_p_vals = nbinom_logsf(interactions[:, 2], exp_interactions, r_val)

        return log_p_vals
