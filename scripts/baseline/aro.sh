prefix=./data
name=part1_test

oprefix=$prefix/baseline_results
gt_labels=$prefix/labels/$name.meta

dim=256

export PYTHONPATH=.


method=aro
th_sim=0.0
knn=80
# knn=80
# method=knn_aro
num_process=16
pred_labels=$oprefix/$name\_$method\_k_$knn\_th_$th_sim/pred_labels.txt
python tools/baseline_cluster.py \
    --prefix $prefix \
    --oprefix $oprefix \
    --name $name \
    --dim $dim \
    --method $method \
    --knn $knn \
    --th_sim $th_sim \
    --num_process $num_process

# eval
metric=pairwise
python evaluation/evaluate.py \
    --metric $metric \
    --gt_labels $gt_labels \
    --pred_labels $pred_labels

metric=bcubed
python evaluation/evaluate.py \
    --metric $metric \
    --gt_labels $gt_labels \
    --pred_labels $pred_labels
