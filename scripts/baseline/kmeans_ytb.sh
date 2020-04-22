prefix=./data
dim=256

name=ytb_test
nclusters=1436

oprefix=$prefix/baseline_results
gt_labels=$prefix/labels/$name.meta


export PYTHONPATH=.

method=kmeans
num_process=8
pred_labels=$oprefix/$name\_$method\_n_$nclusters/pred_labels.txt
python tools/baseline_cluster.py \
    --prefix $prefix \
    --oprefix $oprefix \
    --name $name \
    --dim $dim \
    --method $method \
    --n_clusters $nclusters \
    --num_process $num_process \
    --force

# eval
for metric in pairwise bcubed nmi
do
    python evaluation/evaluate.py \
        --metric $metric \
        --gt_labels $gt_labels \
        --pred_labels $pred_labels
done
