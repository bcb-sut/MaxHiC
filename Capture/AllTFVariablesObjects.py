import tensorflow as tf
import numpy as np
from FuncsDefiner import get_b_type, smax


# returns Bait-Bait ints and Bait-OE ints
# bait-OE ints are ordered in a way that bin1 is bait and the 2nd bin is OE
def separate_order_ints(ints, bait_mask, set_ordered=True):
    
    bb_mask = np.logical_and(bait_mask[ints[:, 0]], bait_mask[ints[:, 1]])
    oo_mask = np.logical_and(np.logical_not(bait_mask[ints[:, 0]]), np.logical_not(bait_mask[ints[:, 1]]))
    bo_mask = np.logical_and(np.logical_not(bb_mask), np.logical_not(oo_mask))
    
    bb_ints = ints[bb_mask, :]
    bo_ints = ints[bo_mask, :]
    oo_ints = ints[oo_mask, :]

    # Reordering bo ints to Bait as B1 and O as B2
    if set_ordered:
        b1 = bait_mask[bo_ints[:, 0]] * bo_ints[:, 0] + bait_mask[bo_ints[:, 1]] * bo_ints[:, 1]
        b2 = bait_mask[bo_ints[:, 1]] * bo_ints[:, 0] + bait_mask[bo_ints[:, 0]] * bo_ints[:, 1]

        bo_ints[:, 0] = b1
        bo_ints[:, 1] = b2

    return bb_ints, bo_ints, oo_ints


def cis_in_range_mask(ints, min_dist, max_dist, min_read):
    
    in_range_mask = np.logical_and(
        np.abs(ints[:, 0] - ints[:, 1]) >= min_dist, ints[:, 2] >= min_read)
    
    if max_dist > 0:
        in_range_mask = np.logical_and(
            in_range_mask,
            np.abs(ints[:, 0] - ints[:, 1]) <= max_dist)
    
    return in_range_mask


def trans_in_range_mask(ints, min_read):
    return ints[:, 2] >= min_read


class AllTFVariablesObjectsHolder:

    def __init__(self, chrs_ranges, cis_interactions, trans_interactions, cores_num, min_dist, max_dist, min_read,
                 bait_mask, print_on_flag):

        # dicts to fill: Keys: c/t_[b/o]/[bb/bo/oo]

        self.active_bins, self.r, self.dist_params, self.v_params, \
            self.ints_sig_end, self.ints_training_end, self.tf_ints, self.tf_ints_place, self.tf_ints_assign, \
            self.tf_ints_on_mask, self.tf_ints_on_mask_place, self.tf_ints_on_mask_assign, \
            self.sig_ints, self.insig_ints, self.training_ints = tuple([dict() for _ in range(15)])

        # Processes

        estimated_r = 2.0

        cis_ints, trans_ints = tuple([dict() for _ in range(2)])
        cis_ints['c_bb'], cis_ints['c_bo'], cis_ints['c_oo'] = separate_order_ints(cis_interactions, bait_mask)
        trans_ints['t_bb'], trans_ints['t_bo'], trans_ints['t_oo'] = separate_order_ints(trans_interactions, bait_mask)

        # SESSION

        self.sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=cores_num,
                                                     inter_op_parallelism_threads=cores_num,
                                                     device_count={'GPU': 0}))

        tf.set_random_seed(0)
        tf.device('/cpu:0')

        # VIS INITIALIZATION

        max_bin_id = max(smax(cis_interactions[:, 0:2]), smax(trans_interactions[:, 0:2]))
        vis = np.ones((max_bin_id + 1,), dtype=np.float64)

        # ACTIVE BINS
        chrs_active_bins = np.unique(np.concatenate((cis_interactions[:, 0:2], trans_interactions[:, 0:2]), axis=0))

        # CIS F PARAMS INITIALIZATION
        
        estimated_f_params = {}
        for itype in ['c_bb', 'c_bo', 'c_oo']:
            temp_cis_f_params = np.polyfit(np.log(np.abs(cis_ints[itype][:, 0] - cis_ints[itype][:, 1])),
                                           np.log(cis_ints[itype][:, 2]), 3)
            estimated_f_params[itype] = np.asarray(list(tuple(temp_cis_f_params)) + [1.0], dtype=np.float64)
            estimated_f_params[itype][0] = min(-0.001, estimated_f_params[itype][0])
    
        # MAKING TF VARIABLES

        self.vis = tf.get_variable('vis', initializer=vis, dtype=tf.float64)
        self.chrs_active_bins = tf.get_variable('active_bins', initializer=chrs_active_bins.astype(np.int32), dtype=tf.int32)
        self.chrs_bin_ranges = tf.get_variable('chrs_ranges', initializer=np.asarray(chrs_ranges, dtype=np.int32),
                                          dtype=tf.int32)

        self.pre_transform_norm_factor = tf.get_variable('pre_transform_norm_factor', initializer=np.float64(1.0), dtype=tf.float64)

        # SEPARATING BAITS  
        
        active_bait_ids = np.unique(np.concatenate((
            cis_ints['c_bb'][:, 0], cis_ints['c_bb'][:, 1], cis_ints['c_bo'][:, 0],
            trans_ints['t_bb'][:, 0], trans_ints['t_bb'][:, 1], trans_ints['t_bo'][:, 0]
        ), axis=0))

        active_oe_ids = np.unique(np.concatenate((
            cis_ints['c_oo'][:, 0], cis_ints['c_oo'][:, 1], cis_ints['c_bo'][:, 1],
            trans_ints['t_oo'][:, 0], trans_ints['t_oo'][:, 1], trans_ints['t_bo'][:, 1]
        ), axis=0))

        self.active_bins['b'] = tf.get_variable('active_baits', initializer=active_bait_ids.astype(np.int32), dtype=tf.int32)
        self.active_bins['o'] = tf.get_variable('active_oes', initializer=active_oe_ids.astype(np.int32), dtype=tf.int32)

        # OTHER VARIABLES
        
        for itype in ['bb', 'bo', 'oo']:
            
            self.r['c_' + itype] = tf.get_variable(itype + '_cis_r', initializer=np.float64(estimated_r), dtype=tf.float64)
            self.dist_params['c_' + itype] = tf.get_variable(itype + '_cis_f_params',
                                                             initializer=estimated_f_params['c_' + itype].astype(np.float64), dtype=tf.float64)

            self.r['t_' + itype] = tf.get_variable(itype + '_trans_r', initializer=np.float64(estimated_r), dtype=tf.float64)
            self.dist_params['t_' + itype] = tf.get_variable(itype + '_trans_f_params', initializer=np.float64(1.0),
                                                       dtype=tf.float64)

        for ctype in ['c', 't']:
            for itype in ['bb_b', 'bo_b', 'bo_o', 'oo_o']:
                self.v_params['%s_%s' % (ctype, itype)] = tf.get_variable('%s_%s_v_params' % (ctype, itype),
                                                                          initializer=np.asarray([1, 1], dtype=np.float64), dtype=tf.float64)
        
        # INITIALIZATION

        self.sess.run(tf.variables_initializer([self.vis, self.chrs_active_bins, self.chrs_bin_ranges, self.pre_transform_norm_factor] +
                                               list(self.active_bins.values())))

        self.sess.run(tf.variables_initializer(list(self.r.values()) + list(self.dist_params.values()) + 
                                               list(self.v_params.values())))
        
        # INTERACTIONS

        for itype in ['c_bb', 'c_bo', 'c_oo']:
            self.ints_sig_end[itype], self.ints_training_end[itype], self.tf_ints[itype], self.tf_ints_place[itype], \
                self.tf_ints_assign[itype], self.tf_ints_on_mask[itype], self.tf_ints_on_mask_place[itype], \
                self.tf_ints_on_mask_assign[itype], self.sig_ints[itype], self.insig_ints[itype], self.training_ints[itype] = \
                self.organize_ints(cis_ints[itype], cis_in_range_mask(cis_ints[itype], min_dist, max_dist, min_read),
                                   itype, print_on_flag)

        for itype in ['t_bb', 't_bo', 't_oo']:
            self.ints_sig_end[itype], self.ints_training_end[itype], self.tf_ints[itype], self.tf_ints_place[itype], \
            self.tf_ints_assign[itype], self.tf_ints_on_mask[itype], self.tf_ints_on_mask_place[itype], \
            self.tf_ints_on_mask_assign[itype], self.sig_ints[itype], self.insig_ints[itype], self.training_ints[itype] = \
                self.organize_ints(trans_ints[itype], trans_in_range_mask(trans_ints[itype], min_read),
                                   itype, print_on_flag)

    def organize_ints(self, ints, in_range_mask, title, print_on_flag):

        # ORDER: Sig-On, Sig-Off, Insig-On, Insig-Off (On = In range)

        # Reordering ons and offs

        all_indices = np.arange(len(ints))
        ordered_indices = np.concatenate((all_indices[in_range_mask], all_indices[np.logical_not(in_range_mask)]), axis=0)

        ints = ints[ordered_indices, :]
        in_range_mask = in_range_mask[ordered_indices]

        in_range_num = int(np.sum(in_range_mask))
        if print_on_flag:
            print('Percent of in range interactions for %s: %.2f' % (title, 100.0 * in_range_num / len(ints)))

        sig_end = tf.Variable(0, name='%s_sig_end' % title, dtype=tf.int32, validate_shape=False)
        training_end = tf.Variable(in_range_num, name='%s_training_end' % title, dtype=tf.int32, validate_shape=False)

        self.sess.run(tf.variables_initializer([sig_end, training_end]))

        tf_ints = tf.Variable([0], name='%s_ints' % title, dtype=tf.int32, validate_shape=False)
        tf_ints_place = tf.placeholder(tf.int32, shape=ints.shape)
        tf_ints_assign = tf.assign(tf_ints, tf_ints_place, validate_shape=False)

        tf_on_mask = tf.Variable([True], name='%s_on_mask' % title, dtype=tf.bool, validate_shape=False)
        tf_on_mask_place = tf.placeholder(tf.bool, shape=in_range_mask.shape)
        tf_on_mask_assign = tf.assign(tf_on_mask, tf_on_mask_place, validate_shape=False)

        self.sess.run(
            [tf_ints_assign, tf_on_mask_assign],
            feed_dict={tf_ints_place: ints, tf_on_mask_place: in_range_mask})

        sig_ints = tf_ints[0:sig_end]
        insig_ints = tf_ints[sig_end:]
        training_ints = tf_ints[sig_end:training_end]

        return sig_end, training_end, tf_ints, tf_ints_place, tf_ints_assign, tf_on_mask, \
            tf_on_mask_place, tf_on_mask_assign, sig_ints, insig_ints, training_ints

    def print_cis_params(self, itype):

        r_val, f_params_val, b1_v_params, b2_v_params = self.sess.run([
            self.r[itype], self.dist_params[itype], self.v_params[get_b_type(itype, 0)],
            self.v_params[get_b_type(itype, 1)]])

        print('***')
        print('Current fitted %s shared params:' % itype)
        print('r: %.4f' % r_val)
        print('dist_params: ', ' '.join('%.4f' % x for x in f_params_val))
        print('b1 vis transformation: ', ' '.join('%.4f' % x for x in b1_v_params))
        print('b2 vis transformation: ', ' '.join('%.4f' % x for x in b2_v_params))
        print('***')

    def print_trans_params(self, itype):

        r_val, f_param_val, b1_v_params, b2_v_params = self.sess.run([
            self.r[itype], self.dist_params[itype], self.v_params[get_b_type(itype, 0)],
            self.v_params[get_b_type(itype, 1)]])

        print('***')
        print('Current fitted %s shared params:' % itype)
        print('r: %.4f' % r_val)
        print('dist_params: %.4f' % f_param_val)
        print('b1 vis transformation: ', ' '.join('%.4f' % x for x in b1_v_params))
        print('b2 vis transformation: ', ' '.join('%.4f' % x for x in b2_v_params))
        print('***')

    def return_final_vals(self):

        np_vals = list(self.sess.run(
            [self.vis, self.pre_transform_norm_factor] +
            list(self.dist_params.values()) + list(self.r.values()) + list(self.v_params.values())))

        np_keys = ['vis', 's_norm'] + ['%s_dist_params' % x for x in self.dist_params.keys()] + \
                  ['%s_r' % x for x in self.r.keys()] + ['%s_v_params' % x for x in self.v_params.keys()]

        return dict(zip(np_keys, np_vals))

