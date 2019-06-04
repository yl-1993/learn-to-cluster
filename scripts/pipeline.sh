prefix=./data
name=part1_test

oprefix=$prefix/cluster_proposals
work_dir=$prefix/work_dir/cfg_0.7_0.75
gt_labels=$prefix/labels/$name.meta

dim=256
knn=80
method=faiss
step=0.05
minsz=3
maxsz=300
metric=pairwise


export PYTHONPATH=.

# generate proposals
for th in 0.7 0.75;
do
    python proposals/generate_proposals.py \
        --prefix $prefix \
        --oprefix $oprefix \
        --name $name \
        --dim $dim \
        --knn $knn \
        --knn_method $method \
        --th_knn $th \
        --th_step $step \
        --min_size $minsz \
        --max_size $maxsz \
        --is_save_proposals

    # single evaluation
    python evaluation/evaluate.py \
        --metric $metric \
        --gt_labels $gt_labels \
        --pred_labels $oprefix/$name/$method\_k_$knn\_th_$th\_step_$step\_minsz_$minsz\_maxsz_$maxsz\_iter_0/pred_labels.txt
done


# test cluster det
python dsgcn/test_cluster_det.py \
    --config dsgcn/configs/cfg_0.7_0.75.yaml \
    --work_dir $work_dir \
    --load_from data/pretrained_models/pretrained_gcn_d.pth.tar \
    --save_output


# deoverlap
python ./post_process/deoverlap.py \
    --pred_score $work_dir/pretrained_gcn_d.npz \
    --th_pos -1 \
    --th_iou 1 \
    --force


# final evaluation
python evaluation/evaluate.py \
    --metric $metric \
    --gt_labels $gt_labels \
    --pred_labels $work_dir/pretrained_gcn_th_iou_1.0_pos_-1.0_pred_label.txt
