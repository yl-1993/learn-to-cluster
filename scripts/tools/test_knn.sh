export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=.

python tools/test_knn.py \
    --name part0_train \
    --knn_method faiss_gpu
