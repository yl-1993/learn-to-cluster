import os.path as osp

# data locations
prefix = './data'
train_name = 'part0_train'
test_name = 'part1_test'
knn = 80
knn_method = 'faiss'
train_thresholds = [0.6, 0.65, 0.7, 0.75]
test_thresholds = [0.7, 0.75]
step = 0.05
minsz = 3
maxsz = 300

train_data = dict(
    wo_weight=False,
    feat_path=osp.join(prefix, 'features', '{}.bin'.format(train_name)),
    label_path=osp.join(prefix, 'labels', '{}.meta'.format(train_name)),
    proposal_folders=[
        osp.join(
            prefix, 'cluster_proposals', train_name,
            '{}_k_{}_th_{}_step_{}_minsz_{}_maxsz_{}_iter_0/'.format(
                knn_method, knn, th, step, minsz, maxsz), 'proposals')
        for th in train_thresholds
    ],
)

test_data = dict(
    wo_weight=False,
    feat_path=osp.join(prefix, 'features', '{}.bin'.format(test_name)),
    label_path=osp.join(prefix, 'labels', '{}.meta'.format(test_name)),
    proposal_folders=[
        osp.join(
            prefix, 'cluster_proposals', test_name,
            '{}_k_{}_th_{}_step_{}_minsz_{}_maxsz_{}_iter_0/'.format(
                knn_method, knn, th, step, minsz, maxsz), 'proposals')
        for th in test_thresholds
    ],
)

# model
model = dict(type='dsgcn',
             kwargs=dict(feature_dim=256,
                         featureless=False,
                         reduce_method='max',
                         hidden_dims=[512, 64]))

# training args
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-4)
optimizer_config = {}

lr_config = dict(
    policy='step',
    step=[15, 24, 28],
)

iter_size = 1
batch_size_per_gpu = 32
total_epochs = 30
workflow = [('train', 1)]

# misc
workers_per_gpu = 1

checkpoint_config = dict(interval=1)

log_level = 'INFO'
log_config = dict(interval=100, hooks=[
    dict(type='TextLoggerHook'),
])

# post_process
th_pos = -1
th_iou = 1

# testing metrics
metrics = ['pairwise', 'bcubed', 'nmi']
