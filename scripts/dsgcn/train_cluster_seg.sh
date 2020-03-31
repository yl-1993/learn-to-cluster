# Note that different configs corresponds to different pred_iou_score
# cfg_name=cfg_train_seg_msm1m_4_prpsls
# pred_iou_score=./data/work_dir/cfg_test_ms1m_2_prpsls/pretrained_gcn_d.npz
# cfg_name=cfg_train_seg_msm1m_8_prpsls
# pred_iou_score=./data/work_dir/cfg_test_ms1m_2_prpsls/pretrained_gcn_d.npz
# cfg_name=cfg_train_seg_msm1m_84_prpsls
# pred_iou_score=./data/work_dir/cfg_test_ms1m_20_prpsls/pretrained_gcn_d.npz

stage=seg
cfg_name=cfg_train_seg_ms1m_84_prpsls
config=./dsgcn/configs/$cfg_name.py

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.

# train
python dsgcn/main.py \
    --stage $stage \
    --phase train \
    --config $config


# test
load_from=./data/work_dir/$cfg_name/latest.pth
pred_iou_score=./data/work_dir/cfg_test_ms1m_20_prpsls/pretrained_gcn_d.npz
python dsgcn/main.py \
    --stage $stage \
    --phase test \
    --config $config \
    --load_from $load_from \
    --pred_iou_score $pred_iou_score \
    --save_output
