# On 1 TitanX, it takes around 1 min for testing
# test on pretrained model: (pre, rec, fscore) = (32.9, 33.61, 33.25)

import os.path as osp
from proposals import generate_proposals

# model
model = dict(type='dsgcn',
             kwargs=dict(feature_dim=256,
                         featureless=False,
                         reduce_method='max',
                         hidden_dims=[512, 64]))

batch_size_per_gpu = 256

# misc
workers_per_gpu = 1

log_level = 'INFO'
log_config = dict(interval=100, hooks=[
    dict(type='TextLoggerHook'),
])

# post_process
th_pos = -1
th_iou = 1

# testing metrics
metrics = ['pairwise', 'bcubed', 'nmi']

# data locations
prefix = './data'
test_name = 'deepfashion_test'
k = 5
knn_method = 'hnsw'

step_i0 = 0.05
minsz_i0 = 3
maxsz_i0 = 50
th_iter0_lst_i0 = [(0.55, True), (0.6, True), (0.65, True)]

th_i1 = 0.4
step_i1 = 0.05
minsz_i1 = 3
maxsz_i1 = 200
sv_minsz_i1 = 2
k_maxsz_lst_i1 = [(2, 8), (2, 12), (2, 16), (3, 5), (3, 10), (4, 4)]

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
                 proposal_folders=generate_proposals(
                     params=proposal_params,
                     prefix=prefix,
                     oprefix=proposal_path,
                     name=test_name,
                     dim=model['kwargs']['feature_dim'],
                     no_normalize=False))
