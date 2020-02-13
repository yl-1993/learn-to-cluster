import os.path as osp

# data locations
prefix = './data'
test_name = 'part1_test'
knn = 80
knn_method = 'faiss'
thresholds = [0.7, 0.75]
step = 0.05
minsz = 3
maxsz = 300

test_data = dict(
    wo_weight=True,
    feat_path=osp.join(prefix, 'features', '{}.bin'.format(test_name)),
    label_path=osp.join(prefix, 'labels', '{}.meta'.format(test_name)),
    proposal_folders=[
        osp.join(
            prefix, 'cluster_proposals', test_name,
            '{}_k_{}_th_{}_step_{}_minsz_{}_maxsz_{}_iter_0/'.format(
                knn_method, knn, th, step, minsz, maxsz), 'proposals')
        for th in thresholds
    ],
)

# model
model = dict(type='dsgcn',
             kwargs=dict(feature_dim=256,
                         featureless=False,
                         reduce_method='max',
                         hidden_dims=[512, 64]))

batch_size_per_gpu = 128

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
