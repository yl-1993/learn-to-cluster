prefix=./data
name=part1_test
oprefix=$prefix/cluster_proposals

dim=256
knn=80
method=faiss
th=0.75
step=0.05
minsz=3
maxsz=300
iter=0

sv_labels=$oprefix/$name/$method\_k_$knn\_th_$th\_step_$step\_minsz_$minsz\_maxsz_$maxsz\_iter_$iter/pred_labels.txt
sv_knn_prefix=$prefix/knns/$name/

# generate proposals iteratively
knn=2
th=0.4
sv_minsz=2
sv_maxsz=16
maxsz=500
iter=$(($iter+1))

# generate iterative proposals
PYTHONPATH=. python proposals/generate_iter_proposals.py \
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
    --sv_min_size $sv_minsz \
    --sv_max_size $sv_maxsz \
    --sv_labels $sv_labels \
    --sv_knn_prefix $sv_knn_prefix \
    --is_save_proposals
