# DeSnow-GNN: A point cloud denoising algorithm for LiDAR in Adverse weather (DeSnow-GNN)
This is the offical implementation of [DeSnow-GNN: A point cloud denoising algorithm for LiDAR in Adverse weather].

## Statement

-Original point cloud.

-Colored by result, red(False Negetive), green(False Positive), blue(True Positive), black(True Negative).

-De-snowed point cloud.

## Requirments
- [pytorch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)<br>
- `torch`, `open3d`, `scipy`, `numpy`, `subprocess`, `os`, `re`.

## Dataset
-[Canadian Adverse Driving Conditions datase](http://cadcd.uwaterloo.ca/)
-[Winter Adverse Driving dataSet](https://digitalcommons.mtu.edu/wads/)
-[Dense: Fog&Rain](https://www.uni-ulm.de/index.php?id=101568)

## Train
-Original point cloud and its label in folder 
```
./data
├── wads
    └── {DRIVE_ID}
        ├── labels/{FRAME_ID}.label
        └── velodyne/{FRAME_ID}.bin
```
-Graph data construction
```

```
-Divide data into training set and evaluation set
```

```

-Training of network
```

```

## Evaluate
Test network performance in evaluation set
```

```

## Inference
Denoising point cloud and visualize

-Data structure like processes of training

-Inference point cloud by network
```

```

-Visualized results can be found in the folder

