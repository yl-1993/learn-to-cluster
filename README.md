# Learning to Cluster Faces

This repo provides an official implementation for [1, 2] and a re-implementation of [3].

## Paper
1. [Learning to Cluster Faces on an Affinity Graph](https://arxiv.org/abs/1904.02749), CVPR 2019 (**Oral**) [[Project Page](http://yanglei.me/project/ltc)]
2. [Learning to Cluster Faces via Confidence and Connectivity Estimation](https://arxiv.org/abs/2004.00445), CVPR 2020 [[Project Page](http://yanglei.me/project/ltc_v2)]
3. [Linkage-based Face Clustering via Graph Convolution Network](https://arxiv.org/abs/1903.11306), CVPR 2019


## Requirements
* Python >= 3.6
* PyTorch >= 0.4.0
* [faiss](https://github.com/facebookresearch/faiss)
* [mmcv](https://github.com/open-mmlab/mmcv)


## Setup and get data

Install dependencies
```bash
conda install faiss-gpu -c pytorch
pip install -r requirements.txt
```

## Datasets
Please refer to [DATASET.md](https://github.com/yl-1993/learn-to-cluster/blob/master/DATASET.md) for data preparation.


## Model zoo
Pretrained models are available in the [model zoo](https://github.com/yl-1993/learn-to-cluster/blob/master/MODEL_ZOO.md).


## Run

0. Fetch code & Create soft link

```bash
git clone git@github.com:yl-1993/learn-to-cluster.git
cd learn-to-cluster
ln -s xxx/data data
```

1. Run algorithms

Follow the instructions in [dsgcn](dsgcn/), [vegcn](vegcn/) and [lgcn](lgcn/) to run algorithms.


## Results on part1_test (584K)

| Method | Precision | Recall | F-score |
| ------ |:---------:|:------:|:-------:|
| Chinese Whispers (k=80, th=0.6, iters=20) | 55.49 | 52.46 | 53.93 |
| Approx Rank Order (k=80, th=0) | 99.77 | 7.2 | 13.42 |
| MiniBatchKmeans (ncluster=5000, bs=100) | 45.48 | 80.98 | 58.25 |
| KNN DBSCAN (k=80, th=0.7, eps=0.25, min=1) | 95.25 | 52.79 | 67.93 |
| FastHAC (dist=0.72, single) | 92.07 | 57.28 | 70.63 |
| [DaskSpectral](https://ml.dask.org/clustering.html#spectral-clustering) (ncluster=8573, affinity='rbf') | 78.75 | 66.59 | 72.16 |
| [CDP](https://github.com/XiaohangZhan/cdp) (single model, th=0.7)  | 80.19 | 70.47 | 75.02 |
| [L-GCN](https://github.com/yl-1993/learn-to-cluster/tree/master/lgcn) (k_at_hop=[200, 10], active_conn=10, step=0.6, maxsz=300)  | 74.38 | 83.51 | 78.68 |
| GCN-D (2 prpsls) | 95.41 | 67.77 | 79.25 |
| GCN-D (5 prpsls) | 94.62 | 72.59 | 82.15 |
| GCN-D (8 prpsls) | 94.23 | 79.69 | 86.35 |
| GCN-D (20 prplss) | 94.54 | 81.62 | 87.61 |
| GCN-D + GCN-S (2 prpsls) | 99.07 | 67.22 | 80.1 |
| GCN-D + GCN-S (5 prpsls) | 98.84 | 72.01 | 83.31 |
| GCN-D + GCN-S (8 prpsls) | 97.93 | 78.98 | 87.44 |
| GCN-D + GCN-S (20 prpsls) | 97.91 | 80.86 | 88.57 |
| GCN-V | 92.45 | 82.42 | 87.14 |
| GCN-V + GCN-E | 92.56 | 83.74 | 87.93 |

Note that the `prpsls` in above table indicate the number of parameters for generating proposals, rather than the actual number of proposals.
For example, `2 prpsls` generates 34578 proposals and `20 prpsls` generates 283552 proposals.


## Benchmarks (5.21M)

`1, 3, 5, 7, 9` denotes different scales of clustering.
Details can be found in [Face Clustering Benchmarks](https://github.com/yl-1993/learn-to-cluster/wiki/Face-Clustering-Benchmarks).

| Pairwise F-score | 1 | 3 | 5 | 7 | 9 |
| ---------------- |:-:|:-:|:-:|:-:|:-:|
| CDP (single model, th=0.7) | 75.02 | 70.75 | 69.51 | 68.62 | 68.06 |
| LGCN | 78.68 | 75.83 | 74.29 | 73.7 | 72.99 |
| GCN-D (2 prpsls) | 79.25 | 75.72 | 73.90 | 72.62 | 71.63 |
| GCN-D (5 prpsls) | 82.15 | 77.71 | 75.5 | 73.99 | 72.89 |
| GCN-D (8 prpsls) | 86.35 | 82.41 | 80.32 | 78.98 | 77.87 |
| GCN-D (20 prpsls) | 87.61 | 83.76 | 81.62 | 80.33 | 79.21 |
| GCN-V | 87.14 | 83.49 | 81.51 | 79.97 | 78.77 |
| GCN-V + GCN-E | 87.93 | 84.04 | 82.1 | 80.45 | 79.3 |

| BCubed F-score | 1 | 3 | 5 | 7 | 9 |
| -------------- |:-:|:-:|:-:|:-:|:-:|
| CDP (single model, th=0.7) | 78.7 | 75.82 | 74.58 | 73.62 | 72.92 |
| LGCN | 84.37 | 81.61 | 80.11 | 79.33 | 78.6 |
| GCN-D (2 prpsls) | 78.89 | 76.05 | 74.65 | 73.57 | 72.77 |
| GCN-D (5 prpsls) | 82.56 | 78.33 | 76.39 | 75.02 | 74.04 |
| GCN-D (8 prpsls) | 86.73 | 83.01 | 81.1 | 79.84 | 78.86 |
| GCN-D (20 prpsls) | 87.76 | 83.99 | 82 | 80.72 | 79.71 |
| GCN-V | 85.81 | 82.63 | 81.05 | 79.92 | 79.08 |
| GCN-V + GCN-E | 86.09 | 82.84 | 81.24 | 80.09 | 79.25 |

| NMI | 1 | 3 | 5 | 7 | 9 |
| --- |:-:|:-:|:-:|:-:|:-:|
| CDP (single model, th=0.7) | 94.69 | 94.62 | 94.63 | 94.62 | 94.61 |
| LGCN | 96.12 | 95.78 | 95.63 | 95.57 | 95.49 |
| GCN-D (2 prpsls) | 94.68 | 94.66 | 94.63 | 94.59 | 94.55 |
| GCN-D (5 prpsls) | 95.64 | 95.19 | 95.03 | 94.91 | 94.83 |
| GCN-D (8 prpsls) | 96.75 | 96.29 | 96.08 | 95.95 | 95.85 |
| GCN-D (20 prpsls) | 97.04 | 96.55 | 96.33 | 96.18 | 96.07 |
| GCN-V | 96.37 | 96.01 | 95.83 | 95.69 | 95.6 |
| GCN-V + GCN-E | 96.41 | 96.03 | 95.85 | 95.71 | 95.62 |


## Results on YouTube-Faces

| Method | Pairwise F-score | BCubed F-score | NMI |
| ------ |:---------:|:------:|:-------:|
| Chinese Whispers (k=160, th=0.75, iters=20) | 72.9 | 70.55 | 93.25 |
| Approx Rank Order (k=200, th=0) | 76.45 | 75.45 | 94.34 |
| Kmeans (ncluster=1436) | 67.86 | 75.77 | 93.99 |
| KNN DBSCAN (k=160, th=0., eps=0.3, min=1) | 91.35 | 89.34 | 97.52 |
| FastHAC (dist=0.72, single) | 93.07 | 87.98 | 97.19 |
| GCN-D (4 prpsls) | 94.44 | 91.33 | 97.97 |


## Results on DeepFashion

| Method | Pairwise F-score | BCubed F-score | NMI |
| ------ |:---------:|:------:|:-------:|
| Chinese Whispers (k=5, th=0.7, iters=20) | 31.22 | 53.25 | 89.8 |
| Approx Rank Order (k=10, th=0) | 25.04 | 52.77 | 88.71 |
| Kmeans (ncluster=3991) | 32.02 | 53.3 | 88.91 |
| KNN DBSCAN (k=4, th=0., eps=0.1, min=2) | 25.07 | 53.23 | 90.75 |
| FastHAC (dist=0.4, single) | 22.54 | 48.77 | 90.44 |
| Meanshift (bandwidth=0.5) | 31.61 | 56.73 | 89.29 |
| Spectral (ncluster=3991, affinity='rbf') | 29.6 | 47.12 | 86.95 |
| DaskSpectral (ncluster=3991, affinity='rbf') | 24.25 | 44.11 | 86.21 |
| CDP (single model, k=2, th=0.5, maxsz=200) | 28.28 | 57.83 | 90.93 |
| L-GCN (k_at_hop=[5, 5], active_conn=5, step=0.5, maxsz=50)  | 30.7 | 60.13 | 90.67 |
| GCN-D (2 prpsls) | 29.14 | 59.09 | 89.48 |
| GCN-D (8 prpsls) | 32.52 | 57.52 | 89.54 |
| GCN-D (20 prpsls) | 33.25 | 56.83 | 89.36 |
| GCN-V | 33.59 | 59.41 | 90.88 |
| GCN-V + GCN-E | 38.47 | 60.06 | 90.5 |

## Face Recognition

For training face recognition and feature extraction, you may use any frameworks below, including but not limited to:

[https://github.com/yl-1993/hfsoftmax](https://github.com/yl-1993/hfsoftmax)

[https://github.com/XiaohangZhan/face_recognition_framework](https://github.com/XiaohangZhan/face_recognition_framework)


## Citation
Please cite the following paper if you use this repository in your reseach.

```
@inproceedings{yang2019learning,
  title={Learning to Cluster Faces on an Affinity Graph},
  author={Yang, Lei and Zhan, Xiaohang and Chen, Dapeng and Yan, Junjie and Loy, Chen Change and Lin, Dahua},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
@inproceedings{yang2020learning,
  title={Learning to Cluster Faces via Confidence and Connectivity Estimation},
  author={Yang, Lei and Chen, Dapeng and Zhan, Xiaohang and Zhao, Rui and Loy, Chen Change and Lin, Dahua},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```
