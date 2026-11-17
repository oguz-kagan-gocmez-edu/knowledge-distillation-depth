---
license: mit
library_name: pytorch
pipeline_tag: depth-estimation
tags:
  - depth-estimation
  - monocular-depth
  - knowledge-distillation
  - semi-supervised
  - nyu-depth-v2
datasets:
  - nyu-depth-v2
metrics:
  - abs_rel
  - rmse
---

# Small depth students distilled from Depth Anything V2 with few labels

Three small networks (3.15M parameters) for **metric** monocular depth on indoor scenes. Each was
trained on NYU Depth V2 with only 1%, 5% or 25% of the real depth labels. Depth Anything V2 Small
was the teacher on the unlabeled images. They come from a CS466 Deep Learning term project
(Özyeğin University, Fall 2026); code, logs and the report are in the
[project repository](https://github.com/oguz-kagan-gocmez-edu/knowledge-distillation-depth).

| Variant | Real depth labels | Recipe | Test AbsRel ↓ | Test RMSE ↓ | Test δ1 ↑ | Same student, labels only |
|---|---|---|---|---|---|---|
| `small-1pct` | 1% (462 images) | teacher on all images + edge loss (E4) | 0.166 | 0.581 | 0.753 | 0.195 |
| `small-5pct` | 5% (2,312) | E4, teacher fades out, kept off labeled images | 0.155 | 0.553 | 0.778 | 0.166 |
| `small-25pct` | 25% (11,562) | same as `small-5pct` | 0.151 | 0.542 | 0.788 | 0.154 |

The test set is the 654 NYU Depth V2 test images, Eigen crop, 0.001–10 m, raw metric depth (no
per-image scaling). The prediction at 256×192 is upsampled to 640×480. The last column is the same
student trained on the same labels without the teacher (mean of three seeds). All three models are
seed 0, chosen on validation (1,000 images from 25 held-out rooms) before the single test run.
At 5% seed 0 was the best of three seeds; the 3-seed mean is 0.158.

For reference, the same student with all labels gets 0.150. The teacher, after its scale and
shift are fitted to each test image's real depth, gets 0.062. That number needs the real depth
at test time, so it is only an upper bound.

## Use

Requires `torch`, `torchvision`, `safetensors` and `huggingface_hub`.

```python
from huggingface_hub import hf_hub_download
import importlib.util

REPO = "<user>/<repo>"                     # this repository
path = hf_hub_download(REPO, "modeling.py")
spec = importlib.util.spec_from_file_location("modeling", path)
modeling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modeling)

model = modeling.load_student("small-1pct", repo_id=REPO)

from PIL import Image
depth = modeling.predict(model, Image.open("room.jpg"))   # HxW numpy array, meters

import matplotlib.pyplot as plt
plt.imsave("depth.png", depth, cmap="inferno_r")            # near = bright, far = dark
print(f"depth {depth.min():.2f}-{depth.max():.2f} m")
```

`model(x)` takes a `(B, 3, 192, 256)` RGB tensor in [0, 1] and returns `(B, 192, 256)` depth in
meters (0.001–10 m). `predict` resizes to 256×192 and back. `config.json` lists each variant's
student size, parameter count and validation metrics.

## Model

- **Student:** FastDepth-style encoder–decoder. The encoder is a MobileNetV3-Large backbone
  (ImageNet-pretrained before training). The decoder concatenates skips at strides 16, 8, 4 and 2,
  each followed by a depthwise-separable conv. The output is a sigmoid scaled to 0.001–10 m.
- **Size and cost:** 3.15M parameters and 0.74 GFLOPs at 256×192, 8× smaller than the teacher.
  It runs in 8.5 ms at batch 1 on an Apple M4 Pro GPU, against 45 ms for the teacher at its own
  input size.
- **Training:**
  - Losses: scale-invariant log loss (SILog) on labeled images; scale- and shift-invariant L1 to
    the teacher's inverse depth on all images; a multi-scale gradient-matching edge loss.
  - Schedule: 24k steps, batch 8, AdamW with a one-cycle schedule.
  - `small-5pct` and `small-25pct`: the teacher's weight falls to 0 over the second half, and the
    teacher is off on labeled images.

## Limitations

- Trained and tested only on NYU Depth V2 (indoor, Kinect, depth up to 10 m). The metric scale is
  learned from that camera and setting, so it will not transfer outdoors or to other cameras
  without fine-tuning.
- The error grows with distance; beyond about 6 m it is much higher than at close range.
- One seed per model; seed-to-seed differences are about 0.001–0.004 AbsRel.
- Not meant for safety-critical use.

## License and terms

- These weights and the code are released under the MIT license.
- Inherited terms:
  - **Teacher:** Depth Anything V2 Small, Apache-2.0. Its outputs were the training targets.
  - **Data:** NYU Depth V2 (Silberman et al.), in the commonly used DenseDepth copy of the
    training set. Follow the dataset's terms; it is intended for research.
  - **ImageNet initialization:** the backbone starts from the torchvision ImageNet weights.
- The Base and Large Depth Anything V2 teachers (CC-BY-NC-4.0) were **not** used for these weights.

## Citation

If you use these models, please cite the works they build on:

```bibtex
@inproceedings{yang2024depthanythingv2,
  title     = {Depth Anything V2},
  author    = {Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  booktitle = {NeurIPS},
  year      = {2024}
}
@inproceedings{silberman2012nyu,
  title     = {Indoor Segmentation and Support Inference from {RGBD} Images},
  author    = {Silberman, Nathan and Hoiem, Derek and Kohli, Pushmeet and Fergus, Rob},
  booktitle = {ECCV},
  year      = {2012}
}
@inproceedings{howard2019mobilenetv3,
  title     = {Searching for {MobileNetV3}},
  author    = {Howard, Andrew and Sandler, Mark and Chu, Grace and Chen, Liang-Chieh and Chen, Bo and Tan, Mingxing and Wang, Weijun and Zhu, Yukun and Pang, Ruoming and Vasudevan, Vijay and Le, Quoc V. and Adam, Hartwig},
  booktitle = {ICCV},
  year      = {2019}
}
@inproceedings{wofk2019fastdepth,
  title     = {{FastDepth}: Fast Monocular Depth Estimation on Embedded Systems},
  author    = {Wofk, Diana and Ma, Fangchang and Yang, Tien-Ju and Karaman, Sertac and Sze, Vivienne},
  booktitle = {ICRA},
  year      = {2019}
}
```
