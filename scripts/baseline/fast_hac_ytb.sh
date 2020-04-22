prefix=./data
name=ytb_test

oprefix=$prefix/baseline_results
gt_labels=$prefix/labels/$name.meta

dim=256

export PYTHONPATH=.


method=fast_hierarchy
dist=0.72
hmethod=single
pred_labels=$oprefix/$name\_$method\_dist_$dist\_hmethod_$hmethod/pred_labels.txt
python tools/baseline_cluster.py \
    --prefix $prefix \
    --oprefix $oprefix \
    --name $name \
    --dim $dim \
    --method $method \
    --distance $dist \
    --hmethod $hmethod \
    --force

# eval
for metric in pairwise bcubed nmi
do
    python evaluation/evaluate.py \
        --metric $metric \
        --gt_labels $gt_labels \
        --pred_labels $pred_labels
done
