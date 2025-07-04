# DeSnow-GNN: Spatiotemporal Graph Neural Network for Robust LiDAR Point Cloud Denoising in Adverse Weather
This is the offical implementation of [DeSnow-GNN: Spatiotemporal Graph Neural Network for Robust LiDAR Point Cloud Denoising in Adverse Weather]. 

## Statement
Result for CADC and WADS dataset
| WADS Dataset | CADC Dataset |
|--------------|--------------|
| ![Original WADS](https://github.com/LezhiLiao/DeSnow-GNN/blob/master/gnn_updata/visualization/gif_path/wads_raw.gif) | ![Original CADC](https://github.com/LezhiLiao/DeSnow-GNN/blob/master/gnn_updata/visualization/gif_path/cadc_raw.gif) |
| <div style="text-align:center"><em>Original point cloud</em></div> | <div style="text-align:center"><em>Original point cloud</em></div> |
| ![De-snowed WADS](https://github.com/LezhiLiao/DeSnow-GNN/blob/master/gnn_updata/visualization/gif_path/wads_denoised.gif) | ![De-snowed CADC](https://github.com/LezhiLiao/DeSnow-GNN/blob/master/gnn_updata/visualization/gif_path/cadc_denoised.gif) |
| <div style="text-align:center"><em>De-snowed point cloud</em></div> | <div style="text-align:center"><em>De-snowed point cloud</em></div> |
## Requirements
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) (torch_geometric)
- [Open3D](http://www.open3d.org/docs/release/getting_started.html) (open3d)
- [SciPy](https://scipy.org/install/) (scipy)
- [scikit-learn](https://scikit-learn.org/stable/install.html) (sklearn)

## Dataset
The open-source datasets used in this work are available at:

- [Canadian Adverse Driving Conditions Dataset (CADC)](http://cadcd.uwaterloo.ca/)  
- [Winter Adverse Driving Dataset (WADS)](https://digitalcommons.mtu.edu/wads/)  
- [DENSE: Fog & Rain Dataset](https://www.uni-ulm.de/index.php?id=101568)  

## Train
-Original point cloud and its label in folder 
```
./data
├── wads
    ├── labels/{FRAME_ID}.label
    └── velodyne/{FRAME_ID}.bin
```
-Graph data construction for training
```
python3 preactgnn.py
```
-Divide data into training set and evaluation set
```
python3 divide.py
```

-Training of network
```
python3 train.py
```

## Evaluate
Test network performance in evaluation set
```
python3 testgnn.py
```

## Inference
End to end point cloud denoising
```
python3 inference.py
```

Convert point cloud to images:

```
python3 point2pic.py
```

Create GIF animation from the generated images:

```
python3 pic2gif.py
```
