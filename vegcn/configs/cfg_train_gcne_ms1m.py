# On 1 TitanX, it takes around 10hr for training
# train with PyTorch (0.5.0a0+e31ab99), the performance is:
# (pre, rec, pairwise fscore) = (90.55, 84.83, 87.6)
# (pre, rec, bcubed fscore) = (93.82, 79.79, 86.23)
# nmi = 96.45

import os.path as osp
from mmcv import Config
from utils import rm_suffix
from vegcn.extract import extract_gcn_v

# data locations
prefix = './data'
train_name = 'part0_train'
test_name = 'part1_test'
knn = 160
knn_method = 'faiss'
th_sim = 0.  # cut edges with similarity smaller than th_sim

# testing args
max_conn = 1
tau = 0.8

metrics = ['pairwise', 'bcubed', 'nmi']

# gcn_v configs
_work_dir = 'work_dir'
ckpt_name = 'latest'  # epoch_80000
gcnv_cfg = './vegcn/configs/cfg_train_gcnv_ms1m.py'
gcnv_cfg_name = rm_suffix(osp.basename(gcnv_cfg))
gcnv_cfg = Config.fromfile(gcnv_cfg)
gcnv_cfg.load_from = '{}/{}/{}/{}.pth'.format(prefix, _work_dir, gcnv_cfg_name,
                                              ckpt_name)

use_gcn_feat = True
feat_paths = []
pred_conf_paths = []
gcnv_nhid = gcnv_cfg.model.kwargs.nhid
for name in [train_name, test_name]:
    gcnv_prefix = '{}/{}/{}/{}_gcnv_k_{}_th_{}'.format(prefix, _work_dir,
                                                       gcnv_cfg_name, name,
                                                       gcnv_cfg.knn,
                                                       gcnv_cfg.th_sim)
    feat_paths.append(
        osp.join(gcnv_prefix, 'features', '{}.bin'.format(ckpt_name)))
    pred_conf_paths.append(
        osp.join(gcnv_prefix, 'pred_confs', '{}.npz'.format(ckpt_name)))

if not use_gcn_feat:
    gcnv_nhid = gcnv_cfg.model.kwargs.feature_dim
    feat_paths = []
    for name in [train_name, test_name]:
        feat_paths.append(osp.join(prefix, 'features', '{}.bin'.format(name)))

train_feat_path, test_feat_path = feat_paths
train_pred_conf_path, test_pred_conf_path = pred_conf_paths

extract_gcn_v(train_feat_path, train_pred_conf_path, 'train_data', gcnv_cfg)
extract_gcn_v(test_feat_path, test_pred_conf_path, 'test_data', gcnv_cfg)

# if `knn_graph_path` is not passed, it will build knn_graph automatically
train_data = dict(feat_path=train_feat_path,
                  label_path=osp.join(prefix, 'labels',
                                      '{}.meta'.format(train_name)),
                  pred_confs=train_pred_conf_path,
                  k=knn,
                  is_norm_feat=True,
                  th_sim=th_sim,
                  max_conn=max_conn,
                  ignore_ratio=0.7)

test_data = dict(feat_path=test_feat_path,
                 label_path=osp.join(prefix, 'labels',
                                     '{}.meta'.format(test_name)),
                 pred_confs=test_pred_conf_path,
                 k=knn,
                 is_norm_feat=True,
                 th_sim=th_sim,
                 max_conn=max_conn,
                 ignore_ratio=0.8)

# model
regressor = False
nclass = 1 if regressor else 2
model = dict(type='gcn_e',
             kwargs=dict(feature_dim=gcnv_nhid,
                         nhid=512,
                         nclass=nclass,
                         dropout=0.))

# training args
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=1e-5)
optimizer_config = {}

total_epochs = 40
lr_config = dict(policy='step',
                 step=[int(r * total_epochs) for r in [0.5, 0.8, 0.9]])

batch_size_per_gpu = 1
iter_size = 32
workflow = [('train_iter_size', 1)]

# misc
workers_per_gpu = 1

checkpoint_config = dict(interval=1)

log_level = 'INFO'
log_config = dict(interval=100, hooks=[
    dict(type='TextLoggerHook'),
])
