# On 1 TitanX, it takes around 5 min for testing (exclude the proposal generation)
# metircs (pre, rec, pairwise fscore)
# test on pretrained model (gt iop, 0.2-0.8): (98.12, 79.18, 87.64)
# test on pretrained model (pred iop, 0.1-0.9): (97.93, 78.98, 87.44)

import os.path as osp
from functools import partial
from proposals import generate_proposals

use_random_seed = True
featureless = False

# model
model = dict(type='dsgcn',
             kwargs=dict(feature_dim=256,
                         featureless=featureless,
                         stage='seg',
                         reduce_method='no_pool',
                         use_random_seed=use_random_seed,
                         hidden_dims=[512, 64]))

batch_size_per_gpu = 1

# misc
workers_per_gpu = 1

log_level = 'INFO'
log_config = dict(interval=100, hooks=[
    dict(type='TextLoggerHook'),
])

# post_process
th_pos = -1
th_iou = 1
th_outlier = 0.5
keep_outlier = True

# testing metrics
metrics = ['pairwise', 'bcubed', 'nmi']

# data locations
prefix = './data'
test_name = 'part1_test'
k = 80
knn_method = 'hnsw'

step_i0 = 0.05
minsz_i0 = 3
maxsz_i0 = 300
th_iter0_lst_i0 = [(0.6, True), (0.7, True), (0.75, False)]

th_i1 = 0.4
step_i1 = 0.05
minsz_i1 = 3
maxsz_i1 = 500
sv_minsz_i1 = 2
k_maxsz_lst_i1 = [(2, 8), (3, 5)]

proposal_params = [
    dict(k=k,
         knn_method=knn_method,
         th_knn=th_i0,
         th_step=step_i0,
         minsz=minsz_i0,
         maxsz=maxsz_i0,
         iter0=iter0,
         iter1_params=[
             dict(k=k_i1,
                  knn_method=knn_method,
                  th_knn=th_i1,
                  th_step=step_i1,
                  minsz=minsz_i1,
                  maxsz=maxsz_i1,
                  sv_minsz=sv_minsz_i1,
                  sv_maxsz=sv_maxsz_i1) for k_i1, sv_maxsz_i1 in k_maxsz_lst_i1
         ]) for th_i0, iter0 in th_iter0_lst_i0
]

feat_path = osp.join(prefix, 'features', '{}.bin'.format(test_name))
label_path = osp.join(prefix, 'labels', '{}.meta'.format(test_name))
proposal_path = osp.join(prefix, 'cluster_proposals')
test_data = dict(wo_weight=True,
                 feat_path=feat_path,
                 label_path=label_path,
                 use_random_seed=use_random_seed,
                 use_max_degree_seed=False,
                 featureless=featureless,
                 th_iop_min=0.1,
                 th_iop_max=0.9,
                 proposal_folders=partial(generate_proposals,
                                          params=proposal_params,
                                          prefix=prefix,
                                          oprefix=proposal_path,
                                          name=test_name,
                                          dim=model['kwargs']['feature_dim'],
                                          no_normalize=False))
