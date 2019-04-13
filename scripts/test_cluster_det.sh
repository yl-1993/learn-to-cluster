PYTHONPATH=. python dsgcn/test_cluster_det.py \
    --config dsgcn/configs/cfg_0.7_0.75.yaml \
    --load_from data/pretrained_models/pretrained_gcn_d.pth.tar \
    --save_output
