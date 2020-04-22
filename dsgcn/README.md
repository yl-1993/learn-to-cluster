# Learning to Cluster Faces on an Affinity Graph (LTC) [![pdf](https://img.shields.io/badge/Arxiv-pdf-orange.svg?style=flat)](https://arxiv.org/abs/1904.02749)

## Paper
[Learning to Cluster Faces on an Affinity Graph](https://arxiv.org/abs/1904.02749), CVPR 2019 (**Oral**) [[Project Page](http://yanglei.me/project/ltc)]

![pipeline](http://yanglei.me/project/ltc/imgs/pipeline.png)


## Test

Download the pretrained models in the [model zoo](https://github.com/yl-1993/learn-to-cluster/blob/master/MODEL_ZOO.md).

Test cluster detection
```bash
sh scripts/dsgcn/test_cluster_det_ms1m.sh
```

Test cluster segmentation
```bash
# predict iop and then conduct seg
sh scripts/dsgcn/test_cluster_det_iop_ms1m.sh
sh scripts/dsgcn/test_cluster_seg_ms1m.sh
```

*[Optional]* GCN-D Upper Bound
It yields the performance when accuracy of GCN-D is 100%.
```bash
sh scripts/dsgcn/step_by_step/gcn_d_upper_bound.sh
```

## Train

Train cluster detection
```bash
sh scripts/dsgcn/train_cluster_det_ms1m.sh
```

Train cluster segmentation
```bash
# seg training uses the ground-truth iop
sh scripts/dsgcn/train_cluster_seg_ms1m.sh
```

Users can choose different proposals in `dsgcn/configs` or design your own proposals for training and testing.

Generally, more proposals leads to better results.
You can control the number of proposals to strike a balance between time and performance.
