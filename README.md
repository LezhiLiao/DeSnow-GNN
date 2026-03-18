# DeSnow-GNN: Spatiotemporal Graph Neural Network for Robust LiDAR Point Cloud Denoising in Adverse Weather

Official implementation of the paper **DeSnow-GNN: Spatiotemporal Graph Neural Network for Robust LiDAR Point Cloud Denoising in Adverse Weather**.

## Overview

DeSnow-GNN is a spatiotemporal graph neural network for LiDAR point cloud denoising under adverse weather conditions. The project supports:

- training on the WADS dataset
- evaluation on the validation split
- end-to-end inference on raw LiDAR frames
- visualization of denoised point clouds and GIF generation

## Results

Qualitative results on the WADS and CADC datasets:

| WADS Dataset | CADC Dataset |
|--------------|--------------|
| ![Original WADS](https://github.com/LezhiLiao/DeSnow-GNN/blob/master/visualization/gif_path/wads_raw.gif) | ![Original CADC](https://github.com/LezhiLiao/DeSnow-GNN/blob/master/visualization/gif_path/cadc_raw.gif) |
| *Original point cloud* | *Original point cloud* |
| ![Denoised WADS](https://github.com/LezhiLiao/DeSnow-GNN/blob/master/visualization/gif_path/wads_denoised.gif) | ![Denoised CADC](https://github.com/LezhiLiao/DeSnow-GNN/blob/master/visualization/gif_path/cadc_denoised.gif) |
| *De-snowed point cloud* | *De-snowed point cloud* |

## Repository Structure

```text
.
|-- preprocess/      # dataset split and graph construction
|-- model/           # DeSnow-GNN network and registration modules
|-- train/           # training script
|-- evaluate/        # validation script
|-- inference/       # inference scripts
|-- visualization/   # point cloud rendering and GIF generation
`-- checkpoint/      # saved model checkpoints
```

## Requirements

### Tested environment

- Ubuntu 20.04
- Python 3.8
- CUDA 11.8
- cuDNN 8.6
- GCC/G++ 9

### Python dependencies

The code depends on the following Python packages:

- `torch`
- `torch-geometric` (`import torch_geometric` in Python)
- `torch-cluster`
- `torch-scatter`
- `numpy`
- `scipy`
- `scikit-learn`
- `open3d`
- `matplotlib`
- `Pillow`
- `joblib`

### System packages relevant to this project

If you are setting up a fresh Ubuntu environment, the following system packages are the ones most relevant to this repository:

- `build-essential`
- `cmake`
- `git`
- `python3.8`
- `libcudnn8`
- `libcudnn8-dev`
- `cuda-toolkit-11-8`
- `libgl1`
- `libglib2.0-0`

## Dataset Preparation

This repository mainly uses the following public datasets:

- [WADS](https://digitalcommons.mtu.edu/wads/)
- [CADC](http://cadcd.uwaterloo.ca/)

### WADS directory layout

```text
wads/
`-- {DRIVE_ID}/
    |-- labels/{FRAME_ID}.label
    `-- velodyne/{FRAME_ID}.bin
```

### CADC directory layout

```text
cadcd/
`-- {DATE}/{DRIVE_ID}/raw/lidar_points_corrected/data/{FRAME_ID}.bin
```

Note:

- the current training pipeline is centered on WADS
- `preprocess/split_data.py` first splits the WADS sequences into training and validation sets
- path variables in the scripts are currently hard-coded and should be edited before running

## Pipeline

### 1. Split the WADS dataset

Run the split script first to create `train/` and `val/` subsets from the original WADS sequences:

```bash
python3 preprocess/split_data.py
```

By default, the script reads from:

```text
/root/autodl-tmp/wads
```

and writes the split dataset to:

```text
/root/autodl-tmp/wads_2
```

### 2. Convert training point clouds into graph data

After splitting the training data, preprocess the point clouds into graph-structured samples:

```bash
python3 preprocess/preprocess.py
```

This script reads the training split and saves graph data under:

```text
/root/autodl-tmp/wads_2/train_graph_data
```

### 3. Train DeSnow-GNN

```bash
python3 train/train.py
```

The training script saves checkpoints to:

```text
./checkpoint
```

## Evaluation

Evaluate a trained model on the validation split:

```bash
python3 evaluate/eval.py
```

Before running evaluation, update the following fields in `evaluate/eval.py`:

- `MODEL_PATH`
- `VAL_DATA_PATH`

## Inference

Run end-to-end denoising on raw LiDAR frames:

```bash
python3 inference/inf.py
```

Before inference, update the following fields in `inference/inf.py`:

- `folder_path`
- `model_path`
- `point_saving_path`
- `channel`

When running inference on CADC, both the input channel setting and the intensity-related feature handling should be adjusted. This is necessary because WADS and CADC are collected by LiDAR sensors with different beam configurations and different intensity distributions.

## Visualization

Convert denoised point clouds to images:

```bash
python3 visualization/point2pic.py
```

Create a GIF from the generated images:

```bash
python3 visualization/pic2gif.py
```

## Notes

- WADS uses 7-dimensional node features in the current setup.
- CADC uses 4-dimensional node features in the current setup.
- Most scripts use absolute paths under `/root/autodl-tmp/...`; adjust them to match your environment.

## Citation

If you find this work useful, please cite:

```bibtex
@article{liao2026desnow,
  title={DeSnow-GNN: Spatiotemporal graph neural network for robust LiDAR point cloud denoising in adverse weather},
  author={Liao, Lezhi and Ding, Xiao and Zhu, Kun and Mei, Liang and Ou, Haiyan},
  journal={Advanced Engineering Informatics},
  volume={72},
  pages={104449},
  year={2026},
  publisher={Elsevier}
}
```
