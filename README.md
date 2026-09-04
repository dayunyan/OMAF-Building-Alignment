# Revisiting the Necessity of Full Accuracy: Weakly Supervised Object-Level Offset Correction for Misaligned Building Labels

[![Conference](https://img.shields.io/badge/CVPR-2026-blue.svg)](https://cvpr.thecvf.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> This is the official PyTorch implementation for the CVPR 2026 paper **"Revisiting the Necessity of Full Accuracy: Weakly Supervised Object-Level Offset Correction for Misaligned Building Labels"**.

### 🚨 News / Update
- **[2026.03]** Our paper has been accepted by **CVPR 2026**! 🎉
- **[2026.03]** Repository created. The source code, pre-trained models, and the real-world misaligned building dataset are currently being cleaned and organized. 

---

## ⏳ Coming Soon

We are working hard to prepare the release! **The full codebase and dataset will be made publicly available here soon.** 

Please **Star ⭐️** and **Watch 👀** this repository to get notified when the code and data are released. 

**Expected Release Content:**
- [ ] Training and inference code for OMAF.
- [ ] The large-scale real-world misaligned building datasets (İslahiye & Antakya).
- [ ] Pre-trained weights and baseline comparisons.
- [ ] Data preprocessing and visualization scripts.

---

## 📖 Abstract

Severe domain shifts and lack of orthorectification in freely available satellite imagery (e.g., Google Earth) often result in an inherent 2D positional offset between images and open-source building footprints. This misalignment severely degrades the performance of standard segmentation models. 

In this work, we propose a weakly supervised multi-stage alignment framework (**OMAF**) to tackle the object-level spatial misalignment without relying on costly high-precision metadata (RPC/DSM). By estimating the optimal offset vectors based on structural edge agreement and regional variance, our method rapidly corrects misaligned labels, achieving up to a significant mIoU improvement on target regions with minimal human annotation.

---

## 🎓 Citation

If you find our work or this repository useful in your research, please consider citing our paper:

```bibtex
@inproceedings{zhang2026revisiting,
  title={Revisiting the Necessity of Full Accuracy: Weakly Supervised Object-Level Offset Correction for Misaligned Building Labels},
  author={Zhang, San and Li, Si and Wang, Wu and Zhao, Liu and Qian, Qi and Sun, Ba},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

---

## 📧 Contact

If you have any questions before the code release, please feel free to open an issue or contact us at: [Your Email Address].
