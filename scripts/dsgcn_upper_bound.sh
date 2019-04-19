#!/usr/bin/env bash

# assign gt_labels and cluster_path
prefix=./data
name=part1_test
oprefix=$prefix/cluster_proposals

knn=80
method=faiss
step=0.05
minsz=3
maxsz=300

gt_labels=$prefix/labels/$name.meta
for th in 0.7 0.75;
do
    cluster_path="$cluster_path $oprefix/$name/$method\_k_$knn\_th_$th\_step_$step\_minsz_$minsz\_maxsz_$maxsz\_iter0/pred_labels.txt"
done

# uncomment following lines to pass args through command line
# gt_labels=$1
# cluster_path=${@:2}


export PYTHONPATH=.

oname='ub'
ofolder=$prefix/results/gcn_ub/
th_pos=-1
th_iou=1

# generate gcn upper bound
python tools/dsgcn_upper_bound.py \
    --output_name $oname \
    --output_folder $ofolder \
    --force \
    --gt_labels $gt_labels \
    --cluster_path $cluster_path

# evaluate
python evaluation/evaluate.py \
    --method 'pairwise' \
    --gt_labels $gt_labels \
    --pred_labels $ofolder/$oname\_th_iou_$th_iou\_pos_$th_pos\_pred_labels.txt
