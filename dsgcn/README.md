# Learning to Cluster Faces on an Affinity Graph (LTC) [![pdf](https://img.shields.io/badge/Arxiv-pdf-orange.svg?style=flat)](https://arxiv.org/abs/1904.02749)

## Paper
[Learning to Cluster Faces on an Affinity Graph](https://arxiv.org/abs/1904.02749), CVPR 2019 (**Oral**) [[Project Page](http://yanglei.me/project/ltc)]

![pipeline](http://yanglei.me/project/ltc/imgs/pipeline.png)


## Test

Test cluster detection
```bash
sh scripts/dsgcn/test_cluster_det.sh
```

Test cluster segmentation
```bash
# predict iop and then conduct seg
sh scripts/dsgcn/test_cluster_det_iop.sh
sh scripts/dsgcn/test_cluster_seg.sh
```

*[Optional]* GCN-D Upper Bound
It yields the performance when accuracy of GCN-D is 100%.
```bash
sh scripts/dsgcn/step_by_step/gcn_d_upper_bound.sh
```

## Train

Train cluster detection
```bash
sh scripts/dsgcn/train_cluster_det.sh
```
Users can choose different proposals in `dsgcn/configs` or design your own proposals for training and testing.

Generally, more proposals leads to better results.
You can control the number of proposals to strike a balance between time and performance.
