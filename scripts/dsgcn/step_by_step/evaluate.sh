PYTHONPATH=. python evaluation/evaluate.py \
    --metric 'pairwise' \
    --gt_labels $1 \
    --pred_labels $2
