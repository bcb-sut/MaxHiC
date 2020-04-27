class AllTFVariablesObjectsHolder:

    def __init__(self, vis, active_bins, chrs_bin_ranges,
                 cis_training_interactions, trans_training_interactions, cis_significant_interactions, trans_significant_interactions,
                 cis_r, cis_f_params, cis_vis_transformer_params,
                 trans_r, trans_f_param, trans_vis_transformer_params,
                 pre_transform_norm_factor,
                 sess):

        self.vis = vis
        self.active_bins = active_bins
        self.chrs_bin_ranges = chrs_bin_ranges

        self.cis_training_interactions = cis_training_interactions
        self.trans_training_interactions = trans_training_interactions

        self.cis_significant_interactions = cis_significant_interactions
        self.trans_significant_interactions = trans_significant_interactions

        self.cis_r = cis_r
        self.cis_f_params = cis_f_params
        self.cis_vis_transformer_params = cis_vis_transformer_params

        self.trans_r = trans_r
        self.trans_f_param = trans_f_param
        self.trans_vis_transformer_params = trans_vis_transformer_params

        self.pre_transform_norm_factor = pre_transform_norm_factor

        self.sess = sess
