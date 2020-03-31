load_from=./data/pretrained_models/pretrained_gcn_s.pth

config=./dsgcn/configs/cfg_test_seg_ms1m_8_prpsls.py
pred_iou_score=./data/work_dir/cfg_test_det_ms1m_8_prpsls/pretrained_gcn_d.npz
pred_iop_score=./data/work_dir/cfg_test_det_ms1m_8_prpsls/pretrained_gcn_d_iop.npz

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.

python dsgcn/main.py \
    --stage seg \
    --phase test \
    --config $config \
    --load_from $load_from \
    --pred_iou_score $pred_iou_score \
    --pred_iop_score $pred_iop_score \
    --save_output
