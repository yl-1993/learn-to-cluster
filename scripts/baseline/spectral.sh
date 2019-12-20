prefix=./data
dim=256

name=part1_test
nclusters=8573

oprefix=$prefix/baseline_results
gt_labels=$prefix/labels/$name.meta


export PYTHONPATH=.

# method=spectral
method=dask_spectral
pred_labels=$oprefix/$name\_$method\_n_$nclusters/pred_labels.txt
python tools/baseline_cluster.py \
    --prefix $prefix \
    --oprefix $oprefix \
    --name $name \
    --dim $dim \
    --method $method \
    --n_clusters $nclusters \
    --force

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
