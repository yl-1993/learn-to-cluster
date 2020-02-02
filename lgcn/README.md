# [Linkage-based Face Clustering via GCN](https://arxiv.org/abs/1903.11306)

## Main modification:

- refactory the training and testing with `Runner` from [mmcv](https://github.com/open-mmlab/mmcv), which is modularized for easy extension.
- replace `torch.Tensor` with `np.ndarray` in dataloader, which makes the dataloader usable in frameworks in addition to PyTorch.
- evaluate lgcn under the same setting as dsgcn.

## Test
0. Download the model pretrained on ms1m (part0_train) from
[Google Drive](https://drive.google.com/open?id=181voJn6yZxNALv-km8MGNSYaNW_gb86z) or
[BaiduYun](https://pan.baidu.com/s/1Qb-UcQ-hVtDRLbGngU1v8A) (passwd: vwet),
and untar it under the root of this project.
```bash
# md5: b9d40d699fddc826b2ac3614ed5fd73f
tar -zxf pretrained_lgcn.tar.gz
```

1. Test

```bash
# Testing takes about 3 hours on 1 TitanX.
sh scripts/lgcn/test_ms1m.sh
```

## Train

We use the training parameters with best performance in our experiments as the default config.

```bash
# Training takes about 27 hours on 1 TitanX.
sh scripts/lgcn/train_ms1m.sh
```

If there is better training config, you are welcome to report to us. 

## Reference

https://github.com/Zhongdao/gcn_clustering
