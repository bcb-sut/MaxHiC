from PValCalculationModel import tf_nbinom_logsf
from FuncsDefiner import tf_cis_ints_expected_vals, get_b_type


class CisPValCalculator:

    def __init__(self, objs_holder, itype):
        
        self.sess = objs_holder.sess

        # universally known variables
        self.vis = objs_holder.vis
        self.r = objs_holder.r[itype]
        self.dist_params = objs_holder.dist_params[itype]
        self.b1_v_params = objs_holder.v_params[get_b_type(itype, 0)]
        self.b2_v_params = objs_holder.v_params[get_b_type(itype, 1)]

    def run_model(self, interactions):

        log_p_vals = tf_nbinom_logsf(
            interactions[:, 2],
            tf_cis_ints_expected_vals(interactions, self.vis, self.dist_params, self.b1_v_params, self.b2_v_params), self.r)

        return log_p_vals
