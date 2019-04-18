# Learning to Cluster Faces on an Affinity Graph (LTC) [![pdf](https://img.shields.io/badge/Arxiv-pdf-orange.svg?style=flat)](https://arxiv.org/abs/1904.02749)

## Paper
[Learning to Cluster Faces on an Affinity Graph](https://arxiv.org/abs/1904.02749), CVPR 2019 (**Oral**) [[Project Page](http://yanglei.me/project/ltc)]


## Requirements
* Python >= 3.6
* PyTorch >= 0.4.0
* [faiss](https://github.com/facebookresearch/faiss)
* [mmcv](https://github.com/open-mmlab/mmcv)


## Setup and get data

Install dependencies
```bash
conda install pytorch=0.4.1 cuda90 -c pytorch
conda install faiss-cpu -c pytorch
pip install -r requirements.txt
```

Download 1 part testing data from
[Google Drive](https://drive.google.com/open?id=1o_Eo3_Ac4k7L9J5vixIvcAgafVSxvVYl) or
[BaiduYun](https://pan.baidu.com/s/1tcLeL60Na1eIYF0iWUXn3g) (passwd: dtq4).

```
data
  ├── features
    ├── part0_train.bin          # acbbc780948e7bfaaee093ef9fce2ccb
    ├── part1_test.bin           # ced42d80046d75ead82ae5c2cdfba621
  ├── labels
    ├── part0_train.meta         # 8573, 576494
    ├── part1_test.meta          # 8573, 584013
  ├── pretrained_models
    ├── pretrained_gcn_d.pth.tar # 213598e70ddbc50f5e3661a6191a8be1   
```

Download entire benchmarks data (including above one) from
[GoogleDrive](https://drive.google.com/file/d/10boLBiYq-6wKC_N_71unlMyNrimRjpVa/view?usp=sharing) or
[OneDrive](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155095455_link_cuhk_edu_hk/Ef588F6OV4ZMqqN85Nf-Pv8BcDzSo7DgSG042TA2E4-4CQ?e=ev2Wfl).

The folder structure is the same as the data above.

## Pipeline

Fetch code & Create soft link
```bash
git clone git@github.com:yl-1993/learn-to-cluster.git
cd learn-to-cluster
ln -s xxx/data data
```

Run
```bash
sh scripts/pipeline.sh
```

## Step-by-step

The `scripts/pipeline.sh` can be decomposed into following steps:

1. Cluster Proposals
generate multi-scale proposals with different `k`, `threshold` or `maxsz`.
```bash
sh scripts/generate_proposals.sh
```

2. Cluster Detection
```bash
sh scripts/test_cluster_det.sh
```

3. Deoverlap
```bash
sh scripts/deoverlap.sh [pred_score]
```

4. Evaluation
```bash
sh scripts/evaluate.sh [gt_label] [pred_label]
```

5. [Optional] GCN Upper Bound
It yields the performance when accuracy of GCN is 100%.
```bash
sh scripts/gcn_upper_bound.sh
```


## Results on part1_test

| Method | Precision | Recall | F-score |
| ------ |:---------:|:------:|:-------:|
| CDP (th=0.7)      | 80.19 | 70.47 | 75.02 |
| GCN-D (0.7, 0.75) | 95.41 | 67.79 | 79.26 |
| GCN-D (0.65, 0.7, 0.75) | 94.64 | 71.53 | 81.48 |
| GCN-D (0.6, 0.65, 0.7, 0.75) | 94.60 | 72.52 | 82.10 |

Generally, more proposals leads to better results.
You can control the number of proposals to strike a balance between time and performance.


## Benchmarks

`1, 3, 5, 7, 9` denotes different scales of clustering.
Details can be found in [Face Clustering Benchmarks](https://github.com/yl-1993/learn-to-cluster/wiki/Face-Clustering-Benchmarks).

| Methods | 1 | 3 | 5 | 7 | 9 |
| ------- |:-:|:-:|:-:|:-:|:-:|
| CDP (th=0.7)      | 75.02 | 70.75 | 69.51 | 68.62 | 68.06 |
| GCN-D (0.7, 0.75) | 79.26 | 75.72 | 73.90 | 72.62 | 71.63 |
| GCN-D (0.6, 0.65, 0.7, 0.75) | 82.10 | 77.63 | 75.38 | 73.91 | 72.77 |


## Citation
Please cite the following paper if you use this repository in your reseach.

```
@inproceedings{yang2019learning,
  title={Learning to Cluster Faces on an Affinity Graph},
  author={Yang, Lei and Zhan, Xiaohang and Chen, Dapeng and Yan, Junjie and Loy, Chen Change and Lin, Dahua},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```
