#!/usr/bin/env bash

fn_meta=$1
cluster_path=${@:2}

oname='ub'
ofolder='./data/results/gcn_ub/'
th_pos=-1
th_iou=1


# generate gcn upper bound
PYTHONPATH=. python tools/gcn_upper_bound.py \
    --output_name $oname \
    --output_folder $ofolder \
    --force \
    --fn_meta $fn_meta \
    --cluster_path $cluster_path

# evaluate
PYTHONPATH=. python evaluation/evaluate.py \
    --method 'pairwise' \
    --gt_labels $fn_meta \
    --pred_labels $ofolder/$oname\_th_iou_$th_iou\_pos_$th_pos\_pred_labels.txt
