## Dataset Preparation

### Data format

The data directory is constucted as follows:
```
.
├── data
|   ├── features
|   |   └── xxx.bin
│   ├── labels
|   |   └── xxx.meta
│   ├── knns
|   |   └── ... 
```

- `features` currently supports binary file. (We plan to support `np.save` file in near future.)
- `labels` supports plain text where each line indicates a label corresponding to the feature file.
- `knns` is not necessary as it can be built with the provided functions.

To experiment with custom dataset, it is required to provided extracted features and labels.
For training, the number of features should be equal to the number of labels.
For testing, the F-score will be evaluated if labels are provided, otherwise only clustering results will be generated.

###  Supported datasets
The supported datasets are listed below.

- [MS-Celeb-1M](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/)
    - Part1 (584K): [Google Drive](https://drive.google.com/open?id=14qS_IH-8Qt01nat4wbrY2X73h7dJv5-K) or [BaiduYun](https://pan.baidu.com/s/1cN7pD2ZNhw1PENP6TdyH4A) (passwd: sn6s)
    - Benchmarks (5.21M): [GoogleDrive](https://drive.google.com/file/d/10boLBiYq-6wKC_N_71unlMyNrimRjpVa/view?usp=sharing) or [OneDrive](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155095455_link_cuhk_edu_hk/Ef588F6OV4ZMqqN85Nf-Pv8BcDzSo7DgSG042TA2E4-4CQ?e=ev2Wfl).
    - Precomputed KNN: [OneDrive](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155095455_link_cuhk_edu_hk/ES_cbfT-m_hEqSTdmBSySEIBGN664NsSamq3-9C4b7yQow?e=qMA36g)
    - Image Lists: [GoogleDrive](https://drive.google.com/file/d/1kurPWh6dm3dWQOLqUAeE-fxHrdnjaULB/view?usp=sharing) or [OneDrive](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155095455_link_cuhk_edu_hk/ET7lHxOXSjtDiMsgqzLK9LgBi_QW0WVzgZdv2UBzE1Bgzg?e=jZ7kCS).
    - Original Images: [OneDrive](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155095455_link_cuhk_edu_hk/ErY9MmAhmlZMvO9y9SagNOcBISISEdzBfJshn-poD84QPQ?e=PRRpBe). We re-align [MS1M-ArcFace](https://www.dropbox.com/s/wpx6tqjf0y5mf6r/faces_ms1m-refine-v2_112x112.zip?dl=0) with our own face alignment model.
    - Pretrained Face Recognition Model: [GoogleDrive](https://drive.google.com/open?id=1eKsh7x-RUIHhIJ1R9AlUjsJdsdbh2qim). For using the model to extract features, please check the [code](https://github.com/yl-1993/hfsoftmax/tree/ltc) and use [sample data](https://drive.google.com/open?id=1VkZWZmBnaQlTaTNQSQXe-8q8Z8pNuI-D) to have a try.
- [YouTube-Face](https://www.cs.tau.ac.il/~wolf/ytfaces/): [GoogleDrive](https://drive.google.com/file/d/1hg3PQTOwyduLVyfgJ7qrN52o9QE35XM4/view?usp=sharing) or
[OneDrive](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155095455_link_cuhk_edu_hk/EU7mfU9F6C9AtZ8SV7kM0yAB0MLx9rzh4nD4kT5_AHXGxg?e=O6Fik9).
- [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html): TODO


Now, you can switch to [README.md](https://github.com/yl-1993/learn-to-cluster/blob/master/README.md) to train and test the model.
