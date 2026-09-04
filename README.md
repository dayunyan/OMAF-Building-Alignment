# Revisiting the Necessity of Full Accuracy: Weakly Supervised Object-Level Offset Correction for Misaligned Building Labels

[![Conference](https://img.shields.io/badge/CVPR-2026-blue.svg)](https://cvpr.thecvf.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of the CVPR 2026 paper **"Revisiting the Necessity of Full Accuracy: Weakly Supervised Object-Level Offset Correction for Misaligned Building Labels"** (OMAF).

### 🚨 News
- **[2026.09]** Source code released. 🎉
- **[2026.03]** Paper accepted by **CVPR 2026**.

---

## 📖 Abstract

Severe domain shifts and lack of orthorectification in freely available satellite imagery (e.g., Google Earth) often result in an inherent 2D positional offset between images and open-source building footprints. This misalignment severely degrades the performance of standard segmentation models.

We propose a weakly supervised multi-stage alignment framework (**OMAF**) to tackle object-level spatial misalignment without relying on costly high-precision metadata (RPC/DSM). By estimating the optimal offset vectors based on structural edge agreement and regional variance, our method rapidly corrects misaligned labels, achieving significant mIoU improvements on target regions with minimal human annotation.

## 🔧 Method Overview

OMAF is a multi-stage pipeline:

1. **Data preparation** — crop imagery to 1024×1024 tiles, split train/val/test (8:1:1), annotate only the test split (weak supervision).
2. **Test-set offset analysis** — `instance_offset_analyzer.py` measures the per-instance offset distribution on the annotated test set and fits a 2D Gaussian.
3. **Unsupervised offset estimation** — `offset_emi.py` estimates offsets on the *unlabeled* train set via edge agreement + variance consistency, and fits the corresponding Gaussian.
4. **Distribution matching** — `distribution_offset_visual.py` compares the two offset Gaussians (KL divergence) and determines the actual offset parameters.
5. **Confidence-weighted instance labels** — `offset_confidence.py` + `offset_instance.py` use the Gaussian posterior as per-instance confidence and build instance labels (`instances/`: bboxes, centroids, offsets, confidences).
6. **Instance offset refinement** — `train_offset_instance.py` (InstanceOffsetNet) learns to refine the offsets; `generate_corrected_labels.sh` then produces corrected masks for the whole dataset.
7. **Downstream segmentation** — train any supported seg net (DeepLabV3+, UNetFormer, SegMAN, SegFormer, VMamba, VWFormer) on the corrected labels with `train_seg_*.py` + `inference_seg_*.sh`.

## 📁 Repository Structure

```
├── config/                    # configuration files (dataset / model / training)
│   └── teq/                   # configs for the offset pipeline
├── geoseg/                    # base segmentation toolbox (models, losses, datasets)
├── mmseg/
├── offset_emi.py              # unsupervised offset estimation (stage 3)
├── distribution_offset_visual.py  # offset distribution matching (stage 4)
├── offset_confidence.py       # confidence from Gaussian posterior (stage 5)
├── offset_instance.py         # instance-label construction (stage 5)
├── instance_offset_analyzer.py    # test-set offset analysis (stage 2)
├── train_offset_instance.py   # InstanceOffsetNet training (stage 6)
├── generate_corrected_labels.py   # corrected label generation (stage 6)
├── train_seg_deeplab.py / train_seg_unetformer.py   # downstream seg training (stage 7)
├── inference_seg_deeplab.py / inference_seg_unetformer.py
├── inference_offset_instance.py   # offset inference
└── tools/
```

## 💿 Installation

Tested with Python 3.11 + PyTorch 2.0+ (CUDA 12.x):

```bash
conda create -n omaf python=3.11 -y
conda activate omaf
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 🚀 Quick Start

```bash
# 1. Analyze the offset distribution on the annotated test set
python instance_offset_analyzer.py

# 2. Estimate offsets on the unlabeled train set (unsupervised)
python offset_emi.py

# 3. Match the two offset distributions
python distribution_offset_visual.py

# 4. Build confidence-weighted instance labels
python offset_confidence.py
python offset_instance.py

# 5. Train InstanceOffsetNet to refine offsets
python train_offset_instance.py -c config/teq/instance_offset.py

# 6. Generate corrected labels for the whole dataset
python generate_corrected_labels.py \
    -d <dataset_root> \
    -c config/teq/instance_offset.py

# 7. Train the downstream segmentation model on corrected labels
python train_seg_deeplab.py -c config/teq/seg_deeplab.py

# Inference
python inference_seg_deeplab.py \
    -i <dataset_root>/test/images \
    -c config/teq/seg_deeplab.py \
    -o fig_results/<exp_name> \
    -ph 512 -pw 512 -b 4 -d building
```

Expected dataset layout per split (`train` / `val` / `test`):

```
<dataset_root>/<split>/
├── images/     # 1024x1024 RGB tiles
└── labels/     # misaligned building footprints (OSM-style)
    gt/         # aligned reference masks (only required for test)
```

## 📊 Datasets

- **İslahiye & Antakya (Turkey, 2023 earthquake)** — our real-world misaligned building datasets introduced in the paper.
- **xBD** — used for pretraining and evaluation.
- Rebuttal experiments additionally used **BONAI** and **Massachusetts Buildings**.

> The İslahiye & Antakya datasets will be released separately; please open an issue if you need early access for research purposes.

## 🎓 Citation

```bibtex
@InProceedings{Xu_2026_CVPR,
    author    = {Xu, Junda and Liu, Yanmeng and Zeng, Xiangqiang and Wu, Jinrong and Qu, Ying and Zhang, Libao},
    title     = {Revisiting the Necessity of Full Accuracy: Weakly Supervised Object-Level Offset Correction for Misaligned Building Labels},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {34854-34864}
}
```

## 🙏 Acknowledgements

This repository is built upon the [GeoSeg](https://github.com/WangLibo1995/GeoSeg) semantic segmentation toolbox.

## 📧 Contact

For questions or issues, please open a GitHub issue.

## License

This project is released under the [MIT License](LICENSE).
