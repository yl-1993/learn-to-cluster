stage=det
cfg_name=cfg_train_det_fashion_84_prpsls
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
python dsgcn/main.py \
    --stage $stage \
    --phase test \
    --config $config \
    --load_from $load_from \
    --save_output
