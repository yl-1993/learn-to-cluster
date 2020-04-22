# Learning to Cluster Faces via Confidence and Connectivity Estimation [![pdf](https://img.shields.io/badge/Arxiv-pdf-orange.svg?style=flat)](https://arxiv.org/abs/2004.00445)

## Paper
[Learning to Cluster Faces via Confidence and Connectivity Estimation](https://arxiv.org/abs/2004.00445), CVPR 2020 [[Project Page](http://yanglei.me/project/ltc_v2)]

![pipeline](http://yanglei.me/project/ltc_v2/imgs/pipeline.png)


## Test

Download the pretrained models in the [model zoo](https://github.com/yl-1993/learn-to-cluster/blob/master/MODEL_ZOO.md).

Test GCN-V
```bash
sh scripts/vegcn/test_gcn_v_ms1m.sh
```

Test GCN-E
```bash
sh scripts/vegcn/test_gcn_e_ms1m.sh
```

## Train

Train GCN-V
```bash
sh scripts/vegcn/train_gcn_v_ms1m.sh
```

Train GCN-E
```bash
sh scripts/vegcn/train_gcn_e_ms1m.sh
```
