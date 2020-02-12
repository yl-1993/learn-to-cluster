config=lgcn/configs/cfg_ms1m.py

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.

# train
python lgcn/main.py \
    --config $config \
    --phase 'train'

# test
load_from=data/work_dir/cfg_ms1m/latest.pth
python lgcn/main.py \
    --config $config \
    --phase 'test' \
    --load_from $load_from \
    --save_output \
    --force
