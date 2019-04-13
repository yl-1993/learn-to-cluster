PYTHONPATH=. python ./proposals/deoverlap.py \
    --pred_score $1 \
    --th_pos -1 \
    --th_iou 1 \
    --force
