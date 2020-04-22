prefix=./data
name=deepfashion_test

oprefix=$prefix/baseline_results
gt_labels=$prefix/labels/$name.meta

dim=256

export PYTHONPATH=.


method=knn_dbscan
knn=4
knn_method=faiss
th_sim=0.0
eps=0.1
min=2
num_process=1
pred_labels=$oprefix/$name\_$method\_eps_$eps\_min_$min\_$knn_method\_k_$knn\_th_$th_sim/pred_labels.txt
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
    --num_process $num_process

# eval
for metric in pairwise bcubed nmi
do
    python evaluation/evaluate.py \
        --metric $metric \
        --gt_labels $gt_labels \
        --pred_labels $pred_labels
done
