CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python dsgcn/main.py \
    --stage det \
    --phase train \
    --config dsgcn/configs/cfg_train_0.7_0.75.yaml
