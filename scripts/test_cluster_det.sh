PYTHONPATH=. python dsgcn/main.py \
    --stage det \
    --phase test \
    --config dsgcn/configs/cfg_0.7_0.75.yaml \
    --load_from data/pretrained_models/pretrained_gcn_d.pth.tar \
    --save_output
