import os.path as osp

# data locations
prefix = './data'
train_name = 'part0_train'
test_name = 'part1_test'
knn = 80
knn_method = 'faiss'

train_data = dict(
    feat_path=osp.join(prefix, 'features', '{}.bin'.format(train_name)),
    label_path=osp.join(prefix, 'labels', '{}.meta'.format(train_name)),
    knn_graph_path=osp.join(prefix, 'knns', train_name,
                            '{}_k_{}.npz'.format(knn_method, knn)),
    k_at_hop=[200, 10],
    active_connection=10,
    is_norm_feat=True,
    is_sort_knns=True,
)

test_data = dict(
    feat_path=osp.join(prefix, 'features', '{}.bin'.format(test_name)),
    label_path=osp.join(prefix, 'labels', '{}.meta'.format(test_name)),
    knn_graph_path=osp.join(prefix, 'knns', test_name,
                            '{}_k_{}.npz'.format(knn_method, knn)),
    k_at_hop=[80, 10],
    active_connection=10,
    is_norm_feat=True,
    is_sort_knns=True,
    is_test=True,
)

# model
model = dict(type='lgcn', kwargs=dict(feature_dim=256))

# training args
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-4)
optimizer_config = {}

lr_config = dict(
    policy='step',
    step=[1, 2, 3],
)

batch_size_per_gpu = 16
total_epochs = 4
workflow = [('train', 1)]

# testing args
max_sz = 300
step = 0.6
pool = 'avg'

metrics = ['pairwise', 'bcubed', 'nmi']

# misc
workers_per_gpu = 1

checkpoint_config = dict(interval=1)

log_level = 'INFO'
log_config = dict(interval=200, hooks=[
    dict(type='TextLoggerHook'),
])
