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

Download 1 part testing data from
[Google Drive](https://drive.google.com/open?id=14qS_IH-8Qt01nat4wbrY2X73h7dJv5-K) or
[BaiduYun](https://pan.baidu.com/s/1cN7pD2ZNhw1PENP6TdyH4A) (passwd: sn6s).

or with scripts below:
```bash
python tools/download_data.py
```

If you have downloaded the data, you can download the pretrained gcn-d and gcn-s from
[Google Drive](https://drive.google.com/file/d/1yC2FIAKSSJIgMYvOZy3aT4I3cyatAyQx/view?usp=sharing) or
[BaiduYun](https://pan.baidu.com/s/1_owuUxPG-sjCoVhjXWCMQw) (passwd: r1dg).

```
data
  ├── features
    ├── part0_train.bin            # acbbc780948e7bfaaee093ef9fce2ccb
    ├── part1_test.bin             # ced42d80046d75ead82ae5c2cdfba621
  ├── labels
    ├── part0_train.meta           # 8573, 576494
    ├── part1_test.meta            # 8573, 584013
  ├── knns
    ├── part0_train/faiss_k_80.npz # 5e4f6c06daf8d29c9b940a851f28a925
    ├── part1_test/faiss_k_80.npz  # d4a7f95b09f80b0167d893f2ca0f5be5
  ├── pretrained_models
    ├── pretrained_gcn_d.pth       # 213598e70ddbc50f5e3661a6191a8be1
    ├── pretrained_gcn_s.pth       # 3251d6e7d4f9178f504b02d8238726f7
    ├── pretrained_gcn_d_iop.pth   # 314fba47b5156dcc91383ad611d5bd96
```

Download entire benchmarks data (including above one) from
[GoogleDrive](https://drive.google.com/file/d/10boLBiYq-6wKC_N_71unlMyNrimRjpVa/view?usp=sharing) or
[OneDrive](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155095455_link_cuhk_edu_hk/Ef588F6OV4ZMqqN85Nf-Pv8BcDzSo7DgSG042TA2E4-4CQ?e=ev2Wfl).
The folder structure is the same as the data above.

*[Optional]* Download precomputed knns from
[OneDrive](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155095455_link_cuhk_edu_hk/ES_cbfT-m_hEqSTdmBSySEIBGN664NsSamq3-9C4b7yQow?e=qMA36g)
and move it to `data` folder.
It can save a lot of time to search knn, especially in the large setting.

*[Optional]* Download the splitted image list from
[GoogleDrive](https://drive.google.com/file/d/1kurPWh6dm3dWQOLqUAeE-fxHrdnjaULB/view?usp=sharing) or
[OneDrive](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155095455_link_cuhk_edu_hk/ET7lHxOXSjtDiMsgqzLK9LgBi_QW0WVzgZdv2UBzE1Bgzg?e=jZ7kCS).
You can train your own feature extractor with the list.

*[Optional]* Download MS-Celeb-1M data from
[OneDrive](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155095455_link_cuhk_edu_hk/ErY9MmAhmlZMvO9y9SagNOcBISISEdzBfJshn-poD84QPQ?e=PRRpBe).
We re-align [MS1M-ArcFace](https://www.dropbox.com/s/wpx6tqjf0y5mf6r/faces_ms1m-refine-v2_112x112.zip?dl=0) with our own face alignment model.

*[Optional]* Download Pretrained face recognition model from
[GoogleDrive](https://drive.google.com/open?id=1eKsh7x-RUIHhIJ1R9AlUjsJdsdbh2qim).
For using the model to extract features, please check the [code](https://github.com/yl-1993/hfsoftmax/tree/ltc)
and use [sample data](https://drive.google.com/open?id=1VkZWZmBnaQlTaTNQSQXe-8q8Z8pNuI-D) to have a try.

*[Optional]* Download YTBFace data from
[GoogleDrive](https://drive.google.com/file/d/1hg3PQTOwyduLVyfgJ7qrN52o9QE35XM4/view?usp=sharing) or
[OneDrive](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155095455_link_cuhk_edu_hk/EU7mfU9F6C9AtZ8SV7kM0yAB0MLx9rzh4nD4kT5_AHXGxg?e=O6Fik9).


## Run

0. Fetch code & Create soft link

```bash
git clone git@github.com:yl-1993/learn-to-cluster.git
cd learn-to-cluster
ln -s xxx/data data
```

1. Run algorithms

Follow the instructions in [dsgcn](dsgcn/), [vegcn](vegcn/) and [lgcn](lgcn/) to run algorithms.


## Results on part1_test

| Method | Precision | Recall | F-score |
| ------ |:---------:|:------:|:-------:|
| Approx Rank Order (knn=80, th=0) | 99.77 | 7.2 | 13.42 |
| MiniBatchKmeans (ncluster=5000, bs=100) | 45.48 | 80.98 | 58.25 |
| KNN DBSCAN (knn=80, th=0.7, eps=0.7, min=40) | 62.38 | 50.66 | 55.92 |
| FastHAC (dist=0.72, single) | 92.07 | 57.28 | 70.63 |
| [DaskSpectral](https://ml.dask.org/clustering.html#spectral-clustering) (ncluster=8573, affinity='rbf') | 78.75 | 66.59 | 72.16 |
| [CDP](https://github.com/XiaohangZhan/cdp) (single model, th=0.7)  | 80.19 | 70.47 | 75.02 |
| [L-GCN](https://github.com/Zhongdao/gcn_clustering) (k_at_hop=[200, 10], active_conn=10, maxsz=300, step=0.6)  | 74.38 | 83.51 | 78.68 |
| GCN-D (2 prpsls) | 95.41 | 67.77 | 79.25 |
| GCN-D (5 prpsls) | 94.62 | 72.59 | 82.15 |
| GCN-D (8 prpsls) | 94.23 | 79.69 | 86.35 |
| GCN-D (20 prplss) | 94.54 | 81.62 | 87.61 |
| GCN-D + GCN-S (2 prpsls) | 99.07 | 67.22 | 80.1 |
| GCN-D + GCN-S (5 prpsls) | 98.84 | 72.01 | 83.31 |
| GCN-D + GCN-S (8 prpsls) | 97.93 | 78.98 | 87.44 |
| GCN-D + GCN-S (20 prplss) | 97.91 | 80.86 | 88.57 |

Note that the `prpsls` in above table indicate the number of parameters for generating proposals, rather than the actual number of proposals.
For example, `2 prpsls` generates 34578 proposals and `20 prpsls` generates 283552 proposals.


## Benchmarks

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

| BCubed F-score | 1 | 3 | 5 | 7 | 9 |
| -------------- |:-:|:-:|:-:|:-:|:-:|
| CDP (single model, th=0.7) | 78.7 | 75.82 | 74.58 | 73.62 | 72.92 |
| LGCN | 84.37 | 81.61 | 80.11 | 79.33 | 78.6 |
| GCN-D (2 prpsls) | 78.89 | 76.05 | 74.65 | 73.57 | 72.77 |
| GCN-D (5 prpsls) | 82.56 | 78.33 | 76.39 | 75.02 | 74.04 |
| GCN-D (8 prpsls) | 86.73 | 83.01 | 81.1 | 79.84 | 78.86 |
| GCN-D (20 prpsls) | 87.76 | 83.99 | 82 | 80.72 | 79.71 |

| NMI | 1 | 3 | 5 | 7 | 9 |
| --- |:-:|:-:|:-:|:-:|:-:|
| CDP (single model, th=0.7) | 94.69 | 94.62 | 94.63 | 94.62 | 94.61 |
| LGCN | 96.12 | 95.78 | 95.63 | 95.57 | 95.49 |
| GCN-D (2 prpsls) | 94.68 | 94.66 | 94.63 | 94.59 | 94.55 |
| GCN-D (5 prpsls) | 95.64 | 95.19 | 95.03 | 94.91 | 94.83 |
| GCN-D (8 prpsls) | 96.75 | 96.29 | 96.08 | 95.95 | 95.85 |
| GCN-D (20 prpsls) | 97.04 | 96.55 | 96.33 | 96.18 | 96.07 |

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
