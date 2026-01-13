## Version 2.0 (stable)

[Welcome to my homepage!](https://WangLibo1995.github.io)

## News 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transformer-meets-dcfam-a-novel-semantic/semantic-segmentation-on-isprs-potsdam)](https://paperswithcode.com/sota/semantic-segmentation-on-isprs-potsdam?p=transformer-meets-dcfam-a-novel-semantic)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transformer-meets-dcfam-a-novel-semantic/semantic-segmentation-on-isprs-vaihingen)](https://paperswithcode.com/sota/semantic-segmentation-on-isprs-vaihingen?p=transformer-meets-dcfam-a-novel-semantic)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficient-hybrid-transformer-learning-global/semantic-segmentation-on-uavid)](https://paperswithcode.com/sota/semantic-segmentation-on-uavid?p=efficient-hybrid-transformer-learning-global)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficient-hybrid-transformer-learning-global/semantic-segmentation-on-loveda)](https://paperswithcode.com/sota/semantic-segmentation-on-loveda?p=efficient-hybrid-transformer-learning-global)

- The code of [PyramidMamba](./geoseg/models/PyramidMamba.py) is released.
- I have updated this repo to pytorch 2.0 and pytorch-lightning 2.0, support multi-gpu training, etc. 
- Pretrained Weights of backbones can be access from [Google Drive](https://drive.google.com/drive/folders/1ELpFKONJZbXmwB5WCXG7w42eHtrXzyPn?usp=sharing)
- [UNetFormer](https://www.sciencedirect.com/science/article/pii/S0924271622001654) (accepted by ISPRS, [PDF](https://www.researchgate.net/profile/Libo-Wang-17/publication/361736439_UNetFormer_A_UNet-like_transformer_for_efficient_semantic_segmentation_of_remote_sensing_urban_scene_imagery/links/62c2a1ed1cbf3a1d12ac1c87/UNetFormer-A-UNet-like-transformer-for-efficient-semantic-segmentation-of-remote-sensing-urban-scene-imagery.pdf)) and **UAVid dataset** are supported.
- ISPRS Vaihingen and Potsdam datasets are supported. Since private sharing is not allowed, you need to download the datasets from the official website and split them by **Folder Structure**.
- More networks are updated and the link of pretrained weights is provided.
- **config/loveda/dcswin.py** provides a detailed explain about **config** setting.
- Inference on huge RS images are supported (inference_huge_image.py).

## Introduction

**GeoSeg** is an open-source  semantic segmentation toolbox based on PyTorch, [pytorch lightning](https://www.pytorchlightning.ai/) and [timm](https://github.com/rwightman/pytorch-image-models), 
which mainly focuses on developing advanced Vision Transformers for remote sensing image segmentation.


## Major Features

- Unified Benchmark

  we provide a unified training script for various segmentation methods.
  
- Simple and Effective

  Thanks to **pytorch lightning** and **timm** , the code is easy for further development.
  
- Supported Remote Sensing Datasets
 
  - [ISPRS Vaihingen and Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx) 
  - [UAVid](https://uavid.nl/)
  - [LoveDA](https://codalab.lisn.upsaclay.fr/competitions/421)
  - [OpenEarthMap](https://open-earth-map.org/)
  - More datasets will be supported in the future.
  
- Multi-scale Training and Testing
- Inference on Huge Remote Sensing Images

## Supported Networks

- Mamba

  - [PyramidMamba](https://arxiv.org/abs/2406.10828)

- Vision Transformer

  - [UNetFormer](https://authors.elsevier.com/a/1fIji3I9x1j9Fs) 
  - [DC-Swin](https://ieeexplore.ieee.org/abstract/document/9681903)
  - [BANet](https://www.mdpi.com/2072-4292/13/16/3065)
  
- CNN
 
  - [MANet](https://ieeexplore.ieee.org/abstract/document/9487010) 
  - [ABCNet](https://www.sciencedirect.com/science/article/pii/S0924271621002379)
  - [A2FPN](https://www.tandfonline.com/doi/full/10.1080/01431161.2022.2030071)
  
## Folder Structure

Prepare the following folders to organize this repo:
```none
airs
├── GeoSeg (code)
├── pretrain_weights (pretrained weights of backbones, such as vit, swin, etc)
├── model_weights (save the model weights trained on ISPRS vaihingen, LoveDA, etc)
├── fig_results (save the masks predicted by models)
├── lightning_logs (CSV format training logs)
├── data
│   ├── LoveDA
│   │   ├── Train
│   │   │   ├── Urban
│   │   │   │   ├── images_png (original images)
│   │   │   │   ├── masks_png (original masks)
│   │   │   │   ├── masks_png_convert (converted masks used for training)
│   │   │   │   ├── masks_png_convert_rgb (original rgb format masks)
│   │   │   ├── Rural
│   │   │   │   ├── images_png 
│   │   │   │   ├── masks_png 
│   │   │   │   ├── masks_png_convert
│   │   │   │   ├── masks_png_convert_rgb
│   │   ├── Val (the same with Train)
│   │   ├── Test
│   │   ├── train_val (Merge Train and Val)
│   ├── uavid
│   │   ├── uavid_train (original)
│   │   ├── uavid_val (original)
│   │   ├── uavid_test (original)
│   │   ├── uavid_train_val (Merge uavid_train and uavid_val)
│   │   ├── train (processed)
│   │   ├── val (processed)
│   │   ├── train_val (processed)
│   ├── vaihingen
│   │   ├── train_images (original)
│   │   ├── train_masks (original)
│   │   ├── test_images (original)
│   │   ├── test_masks (original)
│   │   ├── test_masks_eroded (original)
│   │   ├── train (processed)
│   │   ├── test (processed)
│   ├── potsdam (the same with vaihingen)
```

## Install

Open the folder **airs** using **Linux Terminal** and create python environment:
```
conda create -n airs python=3.8
conda activate airs
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r GeoSeg/requirements.txt
```

Install Mamba
```
pip install causal-conv1d>=1.4.0
pip install mamba-ssm
```

## Pretrained Weights of Backbones

[Baidu Disk](https://pan.baidu.com/s/1foJkxeUZwVi5SnKNpn6hfg) : 1234 

[Google Drive](https://drive.google.com/drive/folders/1ELpFKONJZbXmwB5WCXG7w42eHtrXzyPn?usp=sharing)

## Data Preprocessing

Download the datasets from the official website and split them yourself.

**Vaihingen**

Generate the training set.
```
python GeoSeg/tools/vaihingen_patch_split.py \
--img-dir "data/vaihingen/train_images" \
--mask-dir "data/vaihingen/train_masks" \
--output-img-dir "data/vaihingen/train/images_1024" \
--output-mask-dir "data/vaihingen/train/masks_1024" \
--mode "train" --split-size 1024 --stride 512 
```
Generate the testing set.
```
python GeoSeg/tools/vaihingen_patch_split.py \
--img-dir "data/vaihingen/test_images" \
--mask-dir "data/vaihingen/test_masks_eroded" \
--output-img-dir "data/vaihingen/test/images_1024" \
--output-mask-dir "data/vaihingen/test/masks_1024" \
--mode "val" --split-size 1024 --stride 1024 \
--eroded
```
Generate the masks_1024_rgb (RGB format ground truth labels) for visualization.
```
python GeoSeg/tools/vaihingen_patch_split.py \
--img-dir "data/vaihingen/test_images" \
--mask-dir "data/vaihingen/test_masks" \
--output-img-dir "data/vaihingen/test/images_1024" \
--output-mask-dir "data/vaihingen/test/masks_1024_rgb" \
--mode "val" --split-size 1024 --stride 1024 \
--gt
```
As for the validation set, you can select some images from the training set to build it.

**Potsdam**
```
python GeoSeg/tools/potsdam_patch_split.py \
--img-dir "data/potsdam/train_images" \
--mask-dir "data/potsdam/train_masks" \
--output-img-dir "data/potsdam/train/images_1024" \
--output-mask-dir "data/potsdam/train/masks_1024" \
--mode "train" --split-size 1024 --stride 1024 --rgb-image 
```

```
python GeoSeg/tools/potsdam_patch_split.py \
--img-dir "data/potsdam/test_images" \
--mask-dir "data/potsdam/test_masks_eroded" \
--output-img-dir "data/potsdam/test/images_1024" \
--output-mask-dir "data/potsdam/test/masks_1024" \
--mode "val" --split-size 1024 --stride 1024 \
--eroded --rgb-image
```

```
python GeoSeg/tools/potsdam_patch_split.py \
--img-dir "data/potsdam/test_images" \
--mask-dir "data/potsdam/test_masks" \
--output-img-dir "data/potsdam/test/images_1024" \
--output-mask-dir "data/potsdam/test/masks_1024_rgb" \
--mode "val" --split-size 1024 --stride 1024 \
--gt --rgb-image
```

**UAVid**
```
python GeoSeg/tools/uavid_patch_split.py \
--input-dir "data/uavid/uavid_train_val" \
--output-img-dir "data/uavid/train_val/images" \
--output-mask-dir "data/uavid/train_val/masks" \
--mode 'train' --split-size-h 1024 --split-size-w 1024 \
--stride-h 1024 --stride-w 1024
```

```
python GeoSeg/tools/uavid_patch_split.py \
--input-dir "data/uavid/uavid_train" \
--output-img-dir "data/uavid/train/images" \
--output-mask-dir "data/uavid/train/masks" \
--mode 'train' --split-size-h 1024 --split-size-w 1024 \
--stride-h 1024 --stride-w 1024
```

```
python GeoSeg/tools/uavid_patch_split.py \
--input-dir "data/uavid/uavid_val" \
--output-img-dir "data/uavid/val/images" \
--output-mask-dir "data/uavid/val/masks" \
--mode 'val' --split-size-h 1024 --split-size-w 1024 \
--stride-h 1024 --stride-w 1024
```

**LoveDA**
```
python GeoSeg/tools/loveda_mask_convert.py --mask-dir data/LoveDA/Train/Rural/masks_png --output-mask-dir data/LoveDA/Train/Rural/masks_png_convert
python GeoSeg/tools/loveda_mask_convert.py --mask-dir data/LoveDA/Train/Urban/masks_png --output-mask-dir data/LoveDA/Train/Urban/masks_png_convert
python GeoSeg/tools/loveda_mask_convert.py --mask-dir data/LoveDA/Val/Rural/masks_png --output-mask-dir data/LoveDA/Val/Rural/masks_png_convert
python GeoSeg/tools/loveda_mask_convert.py --mask-dir data/LoveDA/Val/Urban/masks_png --output-mask-dir data/LoveDA/Val/Urban/masks_png_convert
```

## Training

"-c" means the path of the config, use different **config** to train different models.

```
python GeoSeg/train_supervision.py -c GeoSeg/config/uavid/unetformer.py
```

## Testing

"-c" denotes the path of the config, Use different **config** to test different models. 

"-o" denotes the output path 

"-t" denotes the test time augmentation (TTA), can be [None, 'lr', 'd4'], default is None, 'lr' is flip TTA, 'd4' is multiscale TTA

"--rgb" denotes whether to output masks in RGB format

**Vaihingen**
```
python GeoSeg/vaihingen_test.py -c GeoSeg/config/vaihingen/dcswin.py -o fig_results/vaihingen/dcswin --rgb -t 'd4'
```

**Potsdam**
```
python GeoSeg/potsdam_test.py -c GeoSeg/config/potsdam/dcswin.py -o fig_results/potsdam/dcswin --rgb -t 'lr'
```

**LoveDA** ([Online Testing](https://codalab.lisn.upsaclay.fr/competitions/421))
```
python GeoSeg/loveda_test.py -c GeoSeg/config/loveda/dcswin.py -o fig_results/loveda/dcswin_test -t 'd4'
```

**UAVid** ([Online Testing](https://codalab.lisn.upsaclay.fr/competitions/7302))
```
python GeoSeg/inference_uavid.py \
-i 'data/uavid/uavid_test' \
-c GeoSeg/config/uavid/unetformer.py \
-o fig_results/uavid/unetformer_r18 \
-t 'lr' -ph 1152 -pw 1024 -b 2 -d "uavid"
```

## Inference on huge remote sensing image
```
python GeoSeg/inference_huge_image.py \
-i data/vaihingen/test_images \
-c GeoSeg/config/vaihingen/dcswin.py \
-o fig_results/vaihingen/dcswin_huge \
-t 'lr' -ph 512 -pw 512 -b 2 -d "pv"
```

<div>
<img src="vai.png" width="30%"/>
<img src="pot.png" width="35.5%"/>
</div>

## Reproduction Results
|    Method     |  Dataset  |  F1   |  OA   |  mIoU |
|:-------------:|:---------:|:-----:|:-----:|------:|
|  UNetFormer   |   UAVid   |   -   |   -   | 67.63 |
|  UNetFormer   | Vaihingen | 90.30 | 91.10 | 82.54 |
|  UNetFormer   |  Potsdam  | 92.64 | 91.19 | 86.52 |
|  UNetFormer   |  LoveDA   |   -   |   -   | 52.97 |
| FT-UNetFormer | Vaihingen | 91.17 | 91.74 | 83.98 |
| FT-UNetFormer |  Potsdam  | 93.22 | 91.87 | 87.50 |

Due to some random operations in the training stage, reproduced results (run once) are slightly different from the reported in paper.

## Citation

If you find this project useful in your research, please consider citing：

- [UNetFormer: A UNet-like transformer for efficient semantic segmentation of remote sensing urban scene imagery](https://authors.elsevier.com/a/1fIji3I9x1j9Fs)
- [A Novel Transformer Based Semantic Segmentation Scheme for Fine-Resolution Remote Sensing Images](https://ieeexplore.ieee.org/abstract/document/9681903) 
- [Transformer Meets Convolution: A Bilateral Awareness Network for Semantic Segmentation of Very Fine Resolution Urban Scene Images](https://www.mdpi.com/2072-4292/13/16/3065)
- [ABCNet: Attentive Bilateral Contextual Network for Efficient Semantic Segmentation of Fine-Resolution Remote Sensing Images](https://www.sciencedirect.com/science/article/pii/S0924271621002379)
- [Multiattention network for semantic segmentation of fine-resolution remote sensing images](https://ieeexplore.ieee.org/abstract/document/9487010)
- [A2-FPN for semantic segmentation of fine-resolution remotely sensed images](https://www.tandfonline.com/doi/full/10.1080/01431161.2022.2030071)



## Acknowledgement

We wish **GeoSeg** could serve the growing research of remote sensing by providing a unified benchmark 
and inspiring researchers to develop their own segmentation networks. Many thanks the following projects's contributions to **GeoSeg**.
- [pytorch lightning](https://www.pytorchlightning.ai/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
- [ttach](https://github.com/qubvel/ttach)
- [catalyst](https://github.com/catalyst-team/catalyst)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)


## Pipeline

1. 制作数据集
  - 以1024*1024切割图片，并按照8：1：1分成train、val、test三个子集；
  - 标注test数据集；

2. 分析test数据的实例偏移分布，并用二维高斯分布拟合。运行：
```bash
python instance_offset_analyzer.py # 注意更改代码中的输入路径和保存路径
```
检查"output_dir/image_offset_*.csv"文件内容

3. 采用基于边缘增强与方差一致性约束的偏移估计算法，估计test数据的偏移分布，同样用二维高斯分布拟合。运行：
```bash
python offset_emi.py # 注意更改代码中的输入路径和保存路径
```
检查"output_dir/image_offset_emi_*.csv"文件内容

4. 根据以上两个csv文件获取两个数据之间的分布差异，并记录实际偏移的二维高斯拟合参数。运行：
```bash
python distribution_offset_visual.py # 注意更改代码中的输入路径和保存路径
```
检查"output_dir/kl_analysis_results.csv"文件内容：
```python
[
  "file1_2d_mu_x", # x方向均值
  "file1_2d_mu_y", # y方向均值
  "file1_2d_sigma_x", # x方向的方差
  "file1_2d_sigma_y", # y方向的方差
  "file1_2d_rho", # x和y的相关系数
  "file1_2d_cov" # 协方差矩阵
]
```

5. 根据实际偏移的二维高斯分布更新无监督偏移估计的后验概率，将这个概率作为偏移估计的置信度。然后根据无监督偏移估计图和置信度图，构建基于实例的偏移估计标签，保存在数据集与<"images">同级目录<"instances">中。运行：
```bash
python offset_confidence.py # 运行main()，用少量test数据测试效果，在<"align">中只保存置信度图

python offset_instance.py # 标签保存在instances中，先运行check=False，再改成check=True，检查有无错误
```
instances标签格式如下：
```python
{
    'bboxes': [N_instances, 4] 张量，每个元素为[x1, y1, x2, y2]
    'centroids': [N_instances, 2] 张量，每个元素为[cx, cy]
    'gt_offsets': [N_instances, 2] 张量，每个元素为[dx, dy]
    'gt_confidences': [N_instances, 1] 张量，每个元素为置信度
}
```

6. 接下来需要利用深度神经网络的高泛化能力，估计出更准确的修正标签。
  1. 首先构建对应的<geoseg/datasets/teq_instance_dataset.py>、<geoseg/models/InstanceOffsetNet.py>、<geoseg/losses/InstanceLoss.py>、<config/teq/instance_offset.py>、<train_offset_instance.py>，以及推理测试流程<inference_offset_instance.sh>、<inference_offset_instance.py>。
    - train
    ```bash
    python train_offset_instance.py -c config/teq/instance_offset.py
    ```
    
    - inference
    ```bash
    source inference_offset_instance.sh
    ```
  2. 然后用训练好的实例偏移估计网络，对整个数据集的每个样本生成对应的矫正标签。运行：
  ```bash
  source generate_corrected_labels.sh # 注意修改路径
  ```

7. 最后，将上一步生成的矫正标签作为语义分割训练时的标签，正常训练一个分割网络。同样的，构建对应的<geoseg/datasets/teq_dataset.py>、<geoseg/models/Deeplabv3plus.py>、<geoseg/losses/useful_loss.py>、<config/teq/seg_deeplab.py>、<train_seg_deeplab.py>，以及推理测试流程<inference_seg_deeplab.sh>、<inference_seg_deeplab.py>。
  - train
  ```bash
  python train_seg_deeplab.py -c config/teq/seg_deeplab.py
  ```
  
  - inference
  ```bash
  source inference_seg_deeplab.sh # 注意修改路径
  ```
