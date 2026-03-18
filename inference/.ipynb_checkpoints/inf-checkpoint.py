import torch
import numpy as np
from torch_geometric.data import Data
import os
import time
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from torch_geometric.nn import MLP,GATConv,GCNConv
import os
import re
from torch_cluster import knn
from sklearn.decomposition import PCA
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KDTree
import time
import sys

# Add parent directory to path for imports (similar to eval)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.network import DeSnowGNN

num_clusters = 5000 
radius = 1
denoise_radius = 25
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
folder_path = f"/root/autodl-tmp/wads_2/val/37/velodyne"  
# folder_path = f"/root/autodl-tmp/cadc"
model_path = f"/root/autodl-tmp/DeSnow-GNN/checkpoint/wads_120260314_164129_seed1_final.pth"
point_saving_path = f"/root/autodl-tmp/DeSnow-GNN/inference/outpoint"
channel=7#WADS=7 CADC=4
distin_therehold=0.5
temporal_dim=3

def remove_duplicates(data):
    if torch.is_tensor(data):
        unique_points, inverse_indices = torch.unique(data[:, :3], dim=0, return_inverse=True)
        unique_intensity = torch.zeros_like(unique_points[:, 0])
        unique_intensity.scatter_add_(0, inverse_indices, data[:, 3])
        counts = torch.bincount(inverse_indices, minlength=len(unique_points))
        unique_intensity = torch.div(unique_intensity, counts.float(), out=torch.zeros_like(unique_intensity), where=counts!=0)
        return unique_points, unique_intensity
    else:
        unique_points, inverse_indices = np.unique(data[:, :3], axis=0, return_inverse=True)
        unique_intensity = np.bincount(inverse_indices, weights=data[:, 3]) / np.bincount(inverse_indices)
        return unique_points, unique_intensity

def denoise_point_cloud(points, intensity=None):
    distances = np.linalg.norm(points, axis=1)
    mask = distances <= denoise_radius
    filtered_points = points[mask]
    filtered_intensity = intensity[mask] if intensity is not None else None
    out_point = points[~mask]
    return filtered_points, filtered_intensity, out_point

def downsample_point_cloud(points, num_points):
    num_points_input = points.size(0)
    if num_points_input <= num_points:
        return points
    indices = torch.randperm(num_points_input, device=points.device)[:num_points]
    return points[indices]

def intensity_to_node_inten_and_cluster_neinum(assignments, intensity, le):
    if torch.is_tensor(assignments):
        inten_node = torch.zeros(le, device=assignments.device)
        cluster_number = torch.zeros(le, device=assignments.device)
        inten_node.scatter_add_(0, assignments, intensity)
        cluster_number.scatter_add_(0, assignments, torch.ones_like(intensity))
        inten_node = torch.div(inten_node, cluster_number, out=torch.zeros_like(inten_node), where=cluster_number!=0)
        return inten_node.cpu().tolist()
    else:
        assignments_np = assignments.cpu().numpy() if torch.is_tensor(assignments) else np.array(assignments)
        intensity_np = intensity.cpu().numpy() if torch.is_tensor(intensity) else np.array(intensity)
        
        inten_node = np.zeros(le)
        cluster_number = np.zeros(le)
        np.add.at(inten_node, assignments_np, intensity_np)
        np.add.at(cluster_number, assignments_np, 1)
        inten_node = np.divide(inten_node, cluster_number, out=np.zeros_like(inten_node), where=cluster_number!=0)
        return inten_node.tolist()

def create_ass_with_avg_distance_and_intensity(cent_cuda_tensor, points_tensor_cuda, intensity_cuda_tensor):

    distances = torch.cdist(points_tensor_cuda, cent_cuda_tensor, p=2).pow(2)
    assignment = torch.argmin(distances, dim=1)
    min_distances = distances.gather(1, assignment.unsqueeze(1)).squeeze(1)
    time1=time.time()

    avg_intensity = torch.zeros(cent_cuda_tensor.size(0), device='cuda')
    avg_intensity.scatter_add_(0, assignment, intensity_cuda_tensor)
    time2=time.time()
    avg_distance = torch.zeros(cent_cuda_tensor.size(0), device='cuda')
    counts = torch.zeros(cent_cuda_tensor.size(0), device='cuda')
    avg_distance.scatter_add_(0, assignment, min_distances)
    counts.scatter_add_(0, assignment, torch.ones_like(min_distances))
    time3=time.time()

    mask = counts > 0
    avg_distance[mask] /= counts[mask]
    avg_intensity[mask] /= counts[mask]

    return assignment, avg_distance, avg_intensity, counts

def gpu_radius_neighbors(cent_cuda_tensor, radius):
    
    sq_distances = torch.cdist(cent_cuda_tensor, cent_cuda_tensor, p=2).pow(2)
    sq_distances_clone = sq_distances.clone()  
    sq_distances.fill_diagonal_(float('inf'))  
    
    mask = sq_distances <= radius ** 2

    src, dst = torch.where(mask)
    
    edge_list = torch.stack([src, dst], dim=0)
    
    distances = torch.sqrt(sq_distances_clone[mask])
    
    num_vertices = cent_cuda_tensor.shape[0]
    sum_distances = torch.zeros(num_vertices, device=cent_cuda_tensor.device)
    neighbor_counts = torch.zeros(num_vertices, device=cent_cuda_tensor.device)
    
    sum_distances.scatter_add_(0, src, distances)
    neighbor_counts.scatter_add_(0, src, torch.ones_like(distances))
    
    mean_distances = sum_distances / (neighbor_counts + 1e-10)
    
    sensor_distances = torch.norm(cent_cuda_tensor, p=2, dim=1)
    
    Dp = mean_distances / (sensor_distances + 1e-10)  
    return edge_list, Dp

def indices_to_edge(result):
    
    edge_list = []
    for i, neighbors in enumerate(result):
        for neighbor in neighbors:
            edge_list.append([i, neighbor])
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()

def compute_pca_features(points_tensor, k_neighbors=10):
    
    points_np = points_tensor.cpu().numpy() if torch.is_tensor(points_tensor) else points_tensor
    normals = np.zeros((len(points_np), 3))
    
    tree = KDTree(points_np)
    
    for i, point in enumerate(points_np):
   
        query_point = point.reshape(1, -1)
        
        distances, neighbor_indices = tree.query(query_point, k=k_neighbors)
        neighbors = points_np[neighbor_indices[0]]  # 注意取第一个结果
        
        if len(neighbors) < 3:
            normals[i] = np.array([0.0, 0.0, 1.0])
            continue
            
        cov_matrix = np.cov(neighbors.T)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            normals[i] = eigenvectors[:, np.argmin(eigenvalues)]
        except:
            normals[i] = np.array([0.0, 0.0, 1.0])
    
    return normals

def graph_construction(point, intensity):
  
    with torch.no_grad():

        points_np, intensity, out_point = denoise_point_cloud(point, intensity)

        points_tensor = torch.tensor(points_np, dtype=torch.float32, device=device)
        intensity_cuda_tensor = torch.tensor(intensity, dtype=torch.float32, device=device)

        centroids = downsample_point_cloud(points_tensor, num_clusters)
        assignment, avg_distance, avg_intensity, counts = create_ass_with_avg_distance_and_intensity(
            centroids, points_tensor, intensity_cuda_tensor
        )
        edge_index, Dp = gpu_radius_neighbors(centroids,  radius) 
        # normals
        # normals = compute_pca_features(centroids)

        features = torch.stack([
            centroids[:, 0],
            centroids[:, 1],
            centroids[:, 2],
            avg_intensity,
            Dp,#CADC remove Dp,avg_distance,counts,
            avg_distance,#CADC remove Dp,avg_distance,counts,
            counts,#CADC remove Dp,avg_distance,counts,
        ], dim=1).float()
        data = Data(x=features, edge_index=edge_index)
        return data, assignment, out_point, points_np


def file_loading(file):
    bin_file_path = os.path.join(folder_path, file)
    id = bin_file_path[38:44]
    data = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)

    # remove duplicate point
    points_np_uni, intensity_uni = remove_duplicates(data)

    return points_np_uni, intensity_uni

def model_loading(model_path):
    """Load model using DeSnowGNN from network.py (similar to eval)"""
    model = DeSnowGNN(in_channels=channel)
    
    # Load checkpoint and extract model_state_dict (similar to eval)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model state dict from checkpoint")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model state dict directly")
    
    model.to(device)
    model.eval()
    return model

def gnn_inference(graph_data, prev_data, model):
    graph_data.to(device)
    if prev_data is not None:
        prev_data.to(device)
    with torch.no_grad():
        # Model returns predictions directly (no sigmoid applied in eval)
        pred_vet = model(graph_data, prev_data)
    return pred_vet

def point_reconstruction(pred_vet, ass, out_point, in_point):
    pred = pred_vet.detach().cpu().numpy()
    assigment = ass.detach().cpu().numpy()

    indices = np.where(pred < distin_therehold)[0]
    mask = np.isin(assigment, indices)
    selected_indices = np.where(mask)[0]  


    selected_points = in_point[selected_indices]
    stacked_points = np.vstack([out_point, selected_points])
    return stacked_points



if __name__ == "__main__":
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.bin')])
    model = model_loading(model_path)
    prev_data = None
    prev_file_prefix = None
    
    for file in files:
        current_file_prefix = file[:7]
        
        # Reset prev_data if file sequence is not continuous
        if prev_file_prefix is not None and current_file_prefix != prev_file_prefix:
            prev_data = None
        
        # Loading point cloud
        points, intensity = file_loading(file)
        intensity = intensity
        begin_time = time.time()
        
        # Graph data construction
        graph_data, ass, out_point, in_point = graph_construction(points, intensity)
        construction_time = time.time()
        
        # GNN inference graph data
        pred_vet = gnn_inference(graph_data, prev_data, model)
        GNN_inference_time = time.time()
        # Point cloud reconstruction
        point = point_reconstruction(pred_vet, ass, out_point, in_point)
        Reconstruction_time = time.time()
        
        # Update previous data for next iteration
        prev_data = graph_data
        prev_file_prefix = current_file_prefix
        end_time = time.time()
        torch.save(point, f"{point_saving_path}/{file.split('.')[0]}.pt")
        print(f"{file} | total_time: {(end_time - begin_time) * 1000:.2f}ms | cons: {(construction_time - begin_time) * 1000:.2f}ms | inf: {(GNN_inference_time - construction_time) * 1000:.2f}ms | recon: {(Reconstruction_time - GNN_inference_time) * 1000:.2f}ms | updata:{(end_time - Reconstruction_time) * 1000:.2f}ms")