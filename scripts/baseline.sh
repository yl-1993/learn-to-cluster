prefix=./data
name=part1_test

oprefix=$prefix/baseline_results
gt_labels=$prefix/labels/$name.meta

dim=256
metric=pairwise

export PYTHONPATH=.

## approx_rank_order
method=approx_rank_order
knn=80
th_sim=0.0
pred_labels=$oprefix/$method\_k_$knn\_th_$th_sim/pred_labels.txt
python tools/baseline_cluster.py \
    --prefix $prefix \
    --oprefix $oprefix \
    --name $name \
    --dim $dim \
    --method $method \
    --knn $knn \
    --th_sim $th_sim
# eval
python evaluation/evaluate.py \
    --metric $metric \
    --gt_labels $gt_labels \
    --pred_labels $pred_labels


## knn_dbscan
method=knn_dbscan
eps=0.7
min=40
knn=80
th_sim=0.7
pred_labels=$oprefix/$method\_eps_$eps\_min_$min\_k_$knn\_th_$th_sim/pred_labels.txt
python tools/baseline_cluster.py \
    --prefix $prefix \
    --oprefix $oprefix \
    --name $name \
    --dim $dim \
    --method $method \
    --knn $knn \
    --th_sim $th_sim \
    --eps $eps \
    --min_samples $min \
    --force
# eval
python evaluation/evaluate.py \
    --metric $metric \
    --gt_labels $gt_labels \
    --pred_labels $pred_labels


## mini_batch_kmeans
method=mini_batch_kmeans
nclusters=5000
bs=100
pred_labels=$oprefix/$method\_n_$nclusters\_bs_$bs/pred_labels.txt
python tools/baseline_cluster.py \
    --prefix $prefix \
    --oprefix $oprefix \
    --name $name \
    --dim $dim \
    --method $method \
    --n_clusters $nclusters \
    --batch_size $bs
# eval
python evaluation/evaluate.py \
    --metric $metric \
    --gt_labels $gt_labels \
    --pred_labels $pred_labels


## fast_hierarchy
method=fast_hierarchy
dist=0.7
hmethod=single
pred_labels=$oprefix/$method\_dist_$dist\_hmethod_$hmethod/pred_labels.txt
python tools/baseline_cluster.py \
    --prefix $prefix \
    --oprefix $oprefix \
    --name $name \
    --dim $dim \
    --method $method \
    --dist $dist \
    --hmethod $hmethod
# eval
python evaluation/evaluate.py \
    --metric $metric \
    --gt_labels $gt_labels \
    --pred_labels $pred_labels
