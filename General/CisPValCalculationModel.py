import numpy as np
from PValCalculationModel import nbinom_logsf
from FuncsDefiner import np_transformed_vis, np_cis_dist_func


class CisPValCalculator:

    def __init__(self, objs_holder):

        self.sess = objs_holder.sess

        # universally known variables
        self.vis = objs_holder.vis
        self.r = objs_holder.cis_r
        self.f_params = objs_holder.cis_f_params
        self.vis_transformer_params = objs_holder.cis_vis_transformer_params

    def run_model(self, interactions):

        r_val, f_params_val, vis_transformer_params_vals, vis_val = \
            self.sess.run([self.r, self.f_params, self.vis_transformer_params, self.vis])

        vi_p = np_transformed_vis(vis_transformer_params_vals, vis_val[interactions[:, 0]])
        vj_p = np_transformed_vis(vis_transformer_params_vals, vis_val[interactions[:, 1]])

        # calculating expected value
        exp_interactions = vi_p * vj_p * np_cis_dist_func(f_params_val, np.abs(interactions[:, 0] - interactions[:, 1]))

        log_p_vals = nbinom_logsf(interactions[:, 2], exp_interactions, r_val)

        return log_p_vals
