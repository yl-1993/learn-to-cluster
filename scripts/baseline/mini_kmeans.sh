prefix=./data
name=part1_test

oprefix=$prefix/baseline_results
gt_labels=$prefix/labels/$name.meta

dim=256

export PYTHONPATH=.


method=mini_batch_kmeans
nclusters=8573
bs=1000
pred_labels=$oprefix/$name\_$method\_n_$nclusters\_bs_$bs/pred_labels.txt

# cpulimit -l 100 python tools/baseline_cluster.py \
python tools/baseline_cluster.py \
    --prefix $prefix \
    --oprefix $oprefix \
    --name $name \
    --dim $dim \
    --method $method \
    --n_clusters $nclusters \
    --batch_size $bs

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
