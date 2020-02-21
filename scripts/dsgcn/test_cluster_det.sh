config=./dsgcn/configs/cfg_test_ms1m_8_prpsls.py
load_from=./data/pretrained_models/pretrained_gcn_d.pth.tar

export CUDA_VISIBLE_DEVICES=0

PYTHONPATH=. python dsgcn/main.py \
    --stage det \
    --phase test \
    --config $config \
    --load_from $load_from \
    --save_output
