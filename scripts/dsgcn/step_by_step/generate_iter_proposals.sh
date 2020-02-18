prefix=./data
name=part1_test
oprefix=$prefix/cluster_proposals

# iter=0
dim=256
k=80
method=faiss
th=0.75
step=0.05
minsz=3
maxsz=300
iter=0

sv_labels=$oprefix/$name/$method\_k_$k\_th_$th\_step_$step\_minsz_$minsz\_maxsz_$maxsz\_iter_$iter/pred_labels.txt
sv_knn_prefix=$prefix/knns/$name/

# iter=1
k=2
th=0.4
sv_minsz=2
sv_maxsz=8
maxsz=500
iter=$(($iter+1))

# generate iterative proposals
PYTHONPATH=. python proposals/generate_iter_proposals.py \
    --prefix $prefix \
    --oprefix $oprefix \
    --name $name \
    --dim $dim \
    --k $k \
    --knn_method $method \
    --th_knn $th \
    --th_step $step \
    --minsz $minsz \
    --maxsz $maxsz \
    --sv_minsz $sv_minsz \
    --sv_maxsz $sv_maxsz \
    --sv_labels $sv_labels \
    --sv_knn_prefix $sv_knn_prefix \
    --is_save_proposals
