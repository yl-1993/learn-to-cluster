PYTHONPATH=. python evaluation/evaluate.py \
    --method 'pairwise' \
    --gt_labels $1 \
    --pred_labels $2
