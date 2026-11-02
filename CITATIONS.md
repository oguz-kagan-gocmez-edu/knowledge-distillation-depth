# Citations: code, models and data used

The code in this repository is written for this project. It builds on the following
models, libraries, datasets and methods, which must be cited in the report.

## Models and weights

- **Depth Anything V2** (teacher), loaded through Hugging Face `transformers`
  (`depth-anything/Depth-Anything-V2-{Small,Base,Large}-hf`).
  L. Yang et al., "Depth Anything V2", NeurIPS 2024.
- **MobileNetV3** encoder (ImageNet weights from `torchvision`).
  A. Howard et al., "Searching for MobileNetV3", ICCV 2019.
- **EfficientNet-B0** encoder (ImageNet weights from `torchvision`).
  M. Tan and Q. V. Le, "EfficientNet", ICML 2019.

## Data

- **NYU Depth V2**. N. Silberman et al., "Indoor segmentation and support inference from RGBD images", ECCV 2012.
- **DenseDepth training subset** (50,688 images) and its test split.
  I. Alhashim and P. Wonka, "High quality monocular depth estimation via transfer learning", arXiv:1812.11941, 2018.
- Hugging Face parquet copy `sayakpaul/nyu_depth_v2` (only for the streaming option in `scripts/feasibility.py`).

## Methods re-implemented

- Scale-invariant log loss: D. Eigen et al., NeurIPS 2014.
- Scale-and-shift-invariant loss and multi-scale gradient matching: R. Ranftl et al. (MiDaS), TPAMI 2022.
- Depthwise-separable decoder in the style of FastDepth: D. Wofk et al., ICRA 2019.
- Knowledge distillation: G. Hinton et al., arXiv:1503.02531, 2015.

## Libraries

PyTorch, torchvision, Hugging Face `transformers` and `datasets`, NumPy, Pillow, Matplotlib.
