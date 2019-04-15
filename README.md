# Learning to Cluster Faces on an Affinity Graph (LTC) [![pdf](https://img.shields.io/badge/Arxiv-pdf-orange.svg?style=flat)](https://arxiv.org/abs/1904.02749)

## Paper
[Learning to Cluster Faces on an Affinity Graph](https://arxiv.org/abs/1904.02749), CVPR 2019 (**Oral**)


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

Dowload data from
[Google Drive](https://drive.google.com/open?id=1o_Eo3_Ac4k7L9J5vixIvcAgafVSxvVYl) or
[BaiduYun](https://pan.baidu.com/s/1tcLeL60Na1eIYF0iWUXn3g) (passwd: dtq4)
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

## Results

| Method | Precision | Recall | F-score |
| ------ |:---------:|:------:|:-------:|
| CDP    | 80.19     | 70.47  | 75.02   |
| GCN-D (0.7, 0.75) | 95.41 | 67.79 | 79.26 |
| GCN-D (0.65, 0.7, 0.75) | 94.64 | 71.53 | 81.48 |
| GCN-D (0.6, 0.65, 0.7, 0.75) | 94.60 | 72.52 | 82.10 |


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
