prefix=./data
dim=256

name=deepfashion_test
bw=0.5
bin=1

oprefix=$prefix/baseline_results
gt_labels=$prefix/labels/$name.meta

export PYTHONPATH=.


method=meanshift
num_process=8
pred_labels=$oprefix/$name\_$method\_bw_$bw\_bin_$bin/pred_labels.txt
python tools/baseline_cluster.py \
    --prefix $prefix \
    --oprefix $oprefix \
    --name $name \
    --dim $dim \
    --method $method \
    --bw $bw \
    --min_bin_freq $bin \
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
