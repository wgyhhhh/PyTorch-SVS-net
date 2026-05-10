# SVS-Net PyTorch

This repository provides a PyTorch implementation of [SVS-Net](https://www.sciencedirect.com/science/article/pii/S0893608020301672) (Sequential vessel segmentation via deep channel attention network).

<img src="https://github.com/wgyhhhh/PyTorch-SVS-net/blob/main/imgs/svsnet.jpg" width="700" height="500"/>

## Installation

```bash
git clone https://github.com/wgyhhhh/PyTorch-SVS-net.git
cd PyTorch-SVS-net/
```

1. Create an anaconda environment.

```bash
conda create -n svs-net python=3.10
conda activate svs-net
```

2. Install packages.

```bash
pip install -r requirements.txt
```

## Data Layout

The PNG reader expects sequence images and labels like:

```text
dataset/
  train/images/image_s40_i0.png
  train/images/image_s40_i1.png
  train/images/image_s40_i2.png
  train/images/image_s40_i3.png
  train/images/image_s40_i4.png
  train/images/image_s40_i5.png
  train/labels/label_s40.png
  validation/images/image_s80_i0.png
  validation/images/image_s80_i1.png
  validation/images/image_s80_i2.png
  validation/images/image_s80_i3.png
  validation/images/image_s80_i4.png
  validation/images/image_s80_i5.png
  validation/labels/label_s80.png
```

Masks correspond to the last input frame. With `--frame-count 4`, the default policy uses the last four frames, such as `i2,i3,i4,i5`.

## Train

```bash
python train_pytorch.py \
  --images-dir /path/to/dataset/train/images \
  --labels-dir /path/to/dataset/train/labels \
  --val-images-dir /path/to/dataset/validation/images \
  --val-labels-dir /path/to/dataset/validation/labels \
  --output-dir runs/svs_pytorch \
  --frame-policy last \
  --batch-size 4 \
  --epochs 601 \
  --device cuda:0
```

## Predict

```bash
python predict_pytorch.py \
  --images-dir /path/to/dataset/test/images \
  --checkpoint runs/svs_pytorch/best.pt \
  --output-dir runs/svs_predictions \
  --threshold 0.5 \
  --save-probability \
  --device cuda:0
```

## Evaluate

```bash
python evaluate_pytorch.py \
  --images-dir /path/to/dataset/test/images \
  --labels-dir /path/to/dataset/test/labels \
  --checkpoint runs/svs_pytorch/best.pt \
  --output-dir runs/svs_eval \
  --threshold 0.5 \
  --save-probability \
  --device cuda:0
```

Evaluation writes `per_sample_metrics.csv`, `summary_metrics.json`, `pred_*.png`, and optional `prob_*.png`.
