import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# ==================== CONFIGURATION ====================
# User configuration - modify these paths as needed
INPUT_DIRECTORY = "/root/autodl-tmp/DeSnow-GNN/inference/outpoint"
OUTPUT_DIRECTORY = "/root/autodl-tmp/DeSnow-GNN/visualization/pic_path"

# Visualization configuration
X_RANGE = [-40, 40]  # x-axis range [min, max]
Y_RANGE = [-20, 20]  # y-axis range [min, max]
FIGURE_SIZE = (10, 5)  # figure size (width, height)
DPI = 300  # output image resolution
POINT_SIZE = 0.1  # point size in scatter plot
COLORMAP = 'viridis'  # colormap for z-values
# =======================================================

def visualize_point_cloud(points, output_path):
    """Visualize point cloud and save as PNG image (clean version without any annotations)"""
    if isinstance(points, torch.Tensor):
        points = points.numpy()
    
    mask = (points[:, 0] >= X_RANGE[0]) & (points[:, 0] <= X_RANGE[1]) & \
           (points[:, 1] >= Y_RANGE[0]) & (points[:, 1] <= Y_RANGE[1])
    filtered_points = points[mask]
    
    if len(filtered_points) == 0:
        print(f"Warning: No point cloud data within specified range")
        return
    
    x_filtered = filtered_points[:, 0]
    y_filtered = filtered_points[:, 1]
    z_filtered = filtered_points[:, 2]

    z_min, z_max = np.min(z_filtered), np.max(z_filtered)
    colors = (z_filtered - z_min) / (z_max - z_min + 1e-10)

    plt.figure(figsize=FIGURE_SIZE)
    plt.scatter(x_filtered, y_filtered, c=colors, s=POINT_SIZE, marker='.', cmap=COLORMAP)
    plt.xlim(X_RANGE[0], X_RANGE[1])
    plt.ylim(Y_RANGE[0], Y_RANGE[1])
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_point_cloud_files(input_dir, output_dir):
    """Process all .pt files in the directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.pt'):
            input_path = os.path.join(input_dir, filename)
            
            try:
                point_cloud = torch.load(input_path)
                print(f"Processing: {filename}, points count: {len(point_cloud)}")
                
                output_filename = os.path.splitext(filename)[0] + '.png'
                output_path = os.path.join(output_dir, output_filename)
                
                visualize_point_cloud(point_cloud, output_path)
                print(f"Saved visualization to: {output_path}")
                
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")

if __name__ == "__main__":
    process_point_cloud_files(INPUT_DIRECTORY, OUTPUT_DIRECTORY)