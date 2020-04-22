config=lgcn/configs/cfg_test_lgcn_ms1m.py
load_from=data/pretrained_models/pretrained_lgcn_ms1m.pth

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.

python lgcn/main.py \
    --config $config \
    --phase 'test' \
    --load_from $load_from \
    --save_output \
    --force
