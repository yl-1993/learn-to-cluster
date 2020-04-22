cfg_name=cfg_train_gcnv_fashion
config=vegcn/configs/$cfg_name.py

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.

# train
python vegcn/main.py \
    --config $config \
    --phase 'train'

# test
load_from=data/work_dir/$cfg_name/latest.pth
python vegcn/main.py \
    --config $config \
    --phase 'test' \
    --load_from $load_from \
    --save_output \
    --force
