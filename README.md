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

We recommend creating a clean conda environment first:

```bash
conda create -n desnowgnn python=3.8
conda activate desnowgnn
pip install -r requirements.txt
```

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

## Pipeline

### 1. Split the WADS dataset

Run the split script first to create `train/` and `val/` subsets from the original WADS sequences:

```bash
python3 preprocess/split_data.py
```

### 2. Convert training point clouds into graph data

After splitting the training data, preprocess the point clouds into graph-structured samples:

```bash
python3 preprocess/preprocess.py
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

## Inference

Run end-to-end denoising on raw LiDAR frames:

```bash
python3 inference/inf.py
```

When running inference on CADC, set `channel` to 4 dimensions to account for the LiDAR beam-number difference. The intensity should also be normalized to keep it consistent with WADS.

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
