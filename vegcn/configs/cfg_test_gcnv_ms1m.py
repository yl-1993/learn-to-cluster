import os.path as osp

# data locations
prefix = './data'
test_name = 'part1_test'
knn = 80
knn_method = 'faiss'
th_sim = 0.  # cut edges with similarity smaller than th_sim

# if `knn_graph_path` is not passed, it will build knn_graph automatically
test_data = dict(feat_path=osp.join(prefix, 'features',
                                    '{}.bin'.format(test_name)),
                 label_path=osp.join(prefix, 'labels',
                                     '{}.meta'.format(test_name)),
                 knn_graph_path=osp.join(prefix, 'knns', test_name,
                                         '{}_k_{}.npz'.format(knn_method,
                                                              knn)),
                 k=knn,
                 is_norm_feat=True,
                 th_sim=th_sim,
                 conf_metric='s_nbr')

# model
model = dict(type='gcn_v',
             kwargs=dict(feature_dim=256, nhid=512, nclass=1, dropout=0.))

batch_size_per_gpu = 16

# testing args
use_gcn_feat = True
max_conn = 1
tau_0 = 0.65
tau = 0.8

metrics = ['pairwise', 'bcubed', 'nmi']

# misc
workers_per_gpu = 1

log_level = 'INFO'
log_config = dict(interval=200, hooks=[
    dict(type='TextLoggerHook'),
])
