# On 1 TitanX, it takes around 3 min for testing (exclude the proposal generation)
# test on pretrained model: (pre, rec, fscore) = (99.03, 72.08, 83.43)

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

knn_method = 'faiss'
k = 80
step = 0.05
minsz = 3
maxsz = 300

thresholds = [0.55, 0.6, 0.65, 0.7, 0.75]

proposal_params = [
    dict(
        k=k,
        knn_method=knn_method,
        th_knn=th_knn,
        th_step=step,
        minsz=minsz,
        maxsz=maxsz,
    ) for th_knn in thresholds
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
                 th_iop_min=0.2,
                 th_iop_max=0.8,
                 proposal_folders=partial(generate_proposals,
                                          params=proposal_params,
                                          prefix=prefix,
                                          oprefix=proposal_path,
                                          name=test_name,
                                          dim=model['kwargs']['feature_dim'],
                                          no_normalize=False))
