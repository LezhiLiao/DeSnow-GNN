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
from scipy.spatial import cKDTree
import time
import sys
from collections import defaultdict

# Add parent directory to path for imports (similar to eval)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.network import DeSnowGNN

num_clusters = 5000  # 聚类的大小
radius = 1
denoise_radius = 25
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
folder_path = f"/root/autodl-tmp/wads_2/val/37/velodyne"  # 编辑bin文件目录
# folder_path = f"/root/autodl-tmp/cadc2"
# model_path = f"/root/autodl-tmp/DeSnow-GNN/checkpoint/wads_120260314_164129_seed1_final.pth"
model_path = f"/root/autodl-tmp/DeSnow-GNN/checkpoint/20260316_165825_seed1_final.pth"
point_saving_path = f"/root/autodl-tmp/DeSnow-GNN/inference/outpoint"
channel=7#WADS=7 CADC=4
threshold=0.5  # 用于判断是否为雪点的阈值，与eval中的threshold保持一致
temporal_dim=3

# 用于统计总时间的字典
total_times = defaultdict(float)
file_count = 0

def remove_duplicates(data):
    """Remove duplicate points from point cloud (matches eval)"""
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

def denoise_point_cloud(points, intensity=None, max_distance=25):
    """
    Filter points based on distance from origin
    返回滤波后的点（半径内）和半径外的点
    """
    distances = np.linalg.norm(points, axis=1)
    mask = distances <= max_distance
    inside_points = points[mask]
    inside_intensity = intensity[mask] if intensity is not None else None
    outside_points = points[~mask]
    outside_intensity = intensity[~mask] if intensity is not None else None
    return inside_points, inside_intensity, outside_points, outside_intensity

def downsample_point_cloud(points, num_points):
    """Randomly downsample point cloud to fixed number of points (matches eval)"""
    num_points_input = points.size(0)
    if num_points_input <= num_points:
        return points
    indices = torch.randperm(num_points_input, device=points.device)[:num_points]
    return points[indices]

def aggregate_distances_and_counts(assignment, min_distances_global, num_clusters):
    """Aggregate distances and counts for each cluster (matches eval)"""
    unique_assignments = torch.unique(assignment)
    distance_sums = torch.zeros(num_clusters, device=assignment.device)
    counts = torch.zeros(num_clusters, device=assignment.device)

    for index in unique_assignments:
        mask = (assignment == index)
        distance_sums[index] = torch.sum(min_distances_global[mask])
        counts[index] = torch.sum(mask)

    averages = torch.zeros_like(distance_sums)
    non_zero = counts > 0
    averages[non_zero] = distance_sums[non_zero] / counts[non_zero]
    return averages, counts

def create_assignments(centroids, points, num_clusters):
    """Create assignments by finding nearest centroids for each point (matches eval exactly)"""
    save_flag = 1
    for _ in range(10):
        chunk_size = num_clusters // 10
        min_distances = []
        min_indices = []

        for i in range(10):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, num_clusters)
            dist_chunk = torch.cdist(points, centroids[start:end])
            min_dist, min_idx = torch.min(dist_chunk, dim=1)
            min_distances.append(min_dist)
            min_indices.append(min_idx + start)

        min_distances_tensor = torch.stack(min_distances, dim=1)
        min_indices_tensor = torch.stack(min_indices, dim=1)
        min_dist_global, min_idx_global = torch.min(min_distances_tensor, dim=1)
        assignment = torch.gather(min_indices_tensor, 1, min_idx_global.unsqueeze(1)).squeeze()

        complete_tensor = torch.arange(num_clusters, device=assignment.device)
        missing = complete_tensor[~torch.isin(complete_tensor, assignment)]
        if len(missing) == 0:
            save_flag = 0
            break

    distance_sums, counts = aggregate_distances_and_counts(assignment, min_dist_global, num_clusters)
    return assignment, save_flag, distance_sums, counts

def kd_tree_radius_neighbors(data, dis):
    """Build graph using radius-based neighbor search (matches eval/preact.py exactly)"""
    kdtree = cKDTree(data)
    neighbors = []

    # Query 4 points (including self) to ensure at least 3 neighbors
    distances, indices = kdtree.query(data, k=4)
    avg_distances = np.mean(distances[:, 1:], axis=1)  # Exclude self

    for i in range(len(data)):
        # Use fixed radius query (r=1 as in preact.py/eval)
        neighbors_in_radius = kdtree.query_ball_point(data[i], r=1)
        neighbors_i = [idx for idx in neighbors_in_radius if idx != i]

        # If less than 3 neighbors in radius, use 3 nearest neighbors (excluding self)
        if len(neighbors_i) < 3:
            neighbors_i = indices[i, 1:4].tolist()  # Take 2nd-4th points (excluding self)

        neighbors.append(neighbors_i)

    return neighbors, avg_distances

def indices_to_edge_index(neighbors):
    """Convert neighbor indices to edge indices for PyG (matches eval)"""
    edge_list = []
    for i, neighbor_list in enumerate(neighbors):
        for neighbor in neighbor_list:
            edge_list.append([i, neighbor])
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()

def intensity_to_node_features(assignments, intensity, num_clusters):
    """Convert point intensities to node intensities (matches eval)"""
    assignments_np = assignments.cpu().numpy()
    intensity_np = intensity

    inten_node = np.zeros(num_clusters)
    cluster_count = np.zeros(num_clusters)
    np.add.at(inten_node, assignments_np, intensity_np)
    np.add.at(cluster_count, assignments_np, 1)
    inten_node = np.divide(inten_node, cluster_count, out=np.zeros_like(inten_node), where=cluster_count != 0)
    return inten_node, cluster_count

def pointcloud_to_graph(points_np, intensity):
    """
    Convert raw point cloud to graph data format (matching eval's pointcloud_to_graph exactly)
    添加详细的计时统计
    
    Args:
        points_np: Nx3 numpy array of point coordinates
        intensity: Numpy array of intensity values
    
    Returns:
        data: PyG Data object
        assignments: Point-to-node assignments
        inside_points: Points within radius after filtering
        save_flag: Whether clustering was successful
        timings: Dictionary of timing information
    """
    timings = {}
    
    # 1. Remove duplicates
    t0 = time.time()
    points_np_uni, intensity_uni = remove_duplicates(np.column_stack([points_np, intensity]))
    timings['remove_duplicates'] = time.time() - t0

    # 2. Filter points within distance range
    t0 = time.time()
    inside_points, inside_intensity, outside_points, outside_intensity = denoise_point_cloud(
        points_np_uni, intensity_uni, max_distance=denoise_radius
    )
    timings['filter_points'] = time.time() - t0
    
    # 3. Convert to tensor and move to device
    t0 = time.time()
    points_tensor = torch.tensor(inside_points, dtype=torch.float32, device=device)
    timings['to_tensor'] = time.time() - t0

    # 4. Downsample and cluster
    t0 = time.time()
    centroids = downsample_point_cloud(points_tensor, num_clusters)
    timings['downsample'] = time.time() - t0
    
    t0 = time.time()
    assignments, save_flag, dist_sums, counts = create_assignments(centroids, points_tensor, num_clusters)
    timings['create_assignments'] = time.time() - t0

    # 5. Compute node features
    t0 = time.time()
    centroids_cpu = centroids.cpu()
    centroid_distances = torch.norm(centroids, p=2, dim=1).cpu()
    intensity_node, cluster_counts = intensity_to_node_features(assignments, inside_intensity, num_clusters)
    timings['compute_node_features'] = time.time() - t0

    # 6. Build graph structure using radius-based neighbor search
    t0 = time.time()
    neighbors, avg_dist = kd_tree_radius_neighbors(centroids_cpu.numpy(), centroid_distances.numpy())
    timings['kdtree_search'] = time.time() - t0
    
    t0 = time.time()
    edge_index = indices_to_edge_index(neighbors)
    Dp = avg_dist / (centroid_distances.numpy() + 1e-8)  # Avoid division by zero
    timings['build_edges'] = time.time() - t0

    # 7. Construct feature vector
    t0 = time.time()
    features = torch.stack([
        centroids_cpu[:, 0],
        centroids_cpu[:, 1],
        centroids_cpu[:, 2],
        torch.tensor(intensity_node),
        torch.tensor(Dp),
        dist_sums.cpu(),
        counts.cpu(),
    ], dim=1).float()
    timings['construct_features'] = time.time() - t0

    # 8. Create data object
    t0 = time.time()
    data = Data(x=features, edge_index=edge_index)
    timings['create_data'] = time.time() - t0
    
    return data, assignments, inside_points, outside_points, save_flag, timings

def node_label_to_point_mask(pred, assignment, threshold=0.85):
    """
    Convert node-level predictions to point-level mask for points within radius
    返回掩码，True表示非雪点（需要保留），False表示雪点（需要去除）
    """
    t0 = time.time()
    pred = pred.detach().cpu().numpy()
    assignment = assignment.detach().cpu().numpy()
    
    # 创建点级别的掩码，初始都为True（表示非雪点）
    point_mask = np.ones(len(assignment), dtype=bool)
    
    # 获取所有唯一的节点索引
    ass_total = list(set(assignment))
    ass_total_dic = {value: index for index, value in enumerate(ass_total)}
    
    # 对于每个点，如果其所属节点的预测值大于阈值，则标记为False（雪点，需要去除）
    for i in range(len(assignment)):
        node_idx = ass_total_dic.get(assignment[i], None)
        if node_idx is not None and pred[node_idx].item() > threshold:
            point_mask[i] = False  # 雪点，需要去除
    
    mask_time = time.time() - t0
    return point_mask, mask_time

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
    """GNN inference on graph data"""
    t0 = time.time()
    
    graph_data.to(device)
    
    # graph_data.x = graph_data.x[:, 0:4]
    
    with torch.no_grad():
        if prev_data is not None:
            prev_data.to(device)
            # prev_data.x = prev_data.x[:, 0:4]
            pred_vet = model(graph_data, prev_data)
        else:
            pred_vet = model(graph_data, None)
    
    inference_time = time.time() - t0
    return pred_vet, inference_time

def file_loading(file):
    """Load bin file and return points and intensity"""
    bin_file_path = os.path.join(folder_path, file)
    data = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)
    
    # 分离坐标和强度
    points = data[:, :3]
    intensity = data[:, 3]
    
    return points, intensity

def print_timing_summary():
    """打印所有文件的平均耗时统计"""
    global total_times, file_count
    
    if file_count == 0:
        return
    
    print("\n" + "="*80)
    print("TIMING SUMMARY (Average per file)")
    print("="*80)
    
    # 图构建相关时间
    print("\n📊 GRAPH CONSTRUCTION STEPS:")
    graph_steps = [
        ('remove_duplicates', '去重'),
        ('filter_points', '距离滤波'),
        ('to_tensor', '转Tensor'),
        ('downsample', '降采样'),
        ('create_assignments', '聚类分配'),
        ('compute_node_features', '节点特征计算'),
        ('kdtree_search', 'KDTree搜索'),
        ('build_edges', '边构建'),
        ('construct_features', '特征构建'),
        ('create_data', '创建Data对象')
    ]
    
    total_graph_time = 0
    for step_key, step_name in graph_steps:
        if step_key in total_times:
            avg_time = total_times[step_key] / file_count * 1000  # 转换为毫秒
            total_graph_time += avg_time
            print(f"  {step_name:20s} ({step_key:20s}): {avg_time:8.2f} ms")
    
    print(f"  {'-'*60}")
    print(f"  {'Total Graph Construction':42s}: {total_graph_time:8.2f} ms")
    
    # 推理时间
    print("\n🤖 INFERENCE:")
    if 'inference' in total_times:
        avg_inference = total_times['inference'] / file_count * 1000
        print(f"  {'GNN Inference':42s}: {avg_inference:8.2f} ms")
    
    # 点云重构时间
    print("\n🔄 POINT CLOUD RECONSTRUCTION:")
    recon_steps = [
        ('mask_creation', '掩码创建'),
        ('point_selection', '点选择'),
        ('merge_points', '点合并'),
        ('save_pointcloud', '保存点云')
    ]
    total_recon_time = 0
    for step_key, step_name in recon_steps:
        if step_key in total_times:
            avg_time = total_times[step_key] / file_count * 1000
            total_recon_time += avg_time
            print(f"  {step_name:20s} ({step_key:20s}): {avg_time:8.2f} ms")
    
    print(f"  {'-'*60}")
    print(f"  {'Total Reconstruction':42s}: {total_recon_time:8.2f} ms")
    
    # 总时间
    print("\n📈 OVERALL:")
    if 'total' in total_times:
        avg_total = total_times['total'] / file_count * 1000
        print(f"  {'Total Time per file':42s}: {avg_total:8.2f} ms")
    
    print("="*80)

if __name__ == "__main__":
    # 创建保存目录
    os.makedirs(point_saving_path, exist_ok=True)
    
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.bin')])
    model = model_loading(model_path)
    prev_data = None
    prev_file_prefix = None
    
    print(f"Processing {len(files)} files...")
    print(f"Threshold for snow points: {threshold}")
    print(f"Points outside radius {denoise_radius}m will be automatically preserved as non-snow")
    
    for file_idx, file in enumerate(files):
        current_file_prefix = file[:7]
        file_start_time = time.time()
        
        # Reset prev_data if file sequence is not continuous
        if prev_file_prefix is not None and current_file_prefix != prev_file_prefix:
            prev_data = None
        
        # Loading point cloud
        points, intensity = file_loading(file)
        
        # Convert to graph with detailed timing
        graph_data, assignments, inside_points, outside_points, save_flag, graph_timings = pointcloud_to_graph(points, intensity)
        
        if save_flag != 0:
            print(f"Warning: Clustering incomplete for {file}, skipping...")
            continue
        
        # GNN inference
        pred_vet, inference_time = gnn_inference(graph_data, prev_data, model)
        
        # 获取半径内非雪点的掩码
        t0 = time.time()
        inside_point_mask, mask_time = node_label_to_point_mask(pred_vet, assignments, threshold=threshold)
        total_times['mask_creation'] += mask_time
        
        # 保留半径内的非雪点
        t0 = time.time()
        non_snow_inside_points = inside_points[inside_point_mask]
        total_times['point_selection'] += time.time() - t0
        
        # 半径外的点全部保留（视为非雪）
        t0 = time.time()
        if len(outside_points) > 0:
            final_point_cloud = np.vstack([non_snow_inside_points, outside_points])
        else:
            final_point_cloud = non_snow_inside_points
        total_times['merge_points'] += time.time() - t0
        
        # 保存最终点云为pt格式
        t0 = time.time()
        output_file = os.path.join(point_saving_path, f"{file.split('.')[0]}.pt")
        torch.save(final_point_cloud, output_file)
        total_times['save_pointcloud'] += time.time() - t0
        
        # Update previous data for next iteration
        prev_data = graph_data
        prev_file_prefix = current_file_prefix
        
        # 累加各项时间到总统计
        for key, value in graph_timings.items():
            total_times[key] += value
        total_times['inference'] += inference_time
        total_times['total'] += time.time() - file_start_time
        file_count += 1
        
        # 输出统计信息
        total_points = len(points)
        total_inside = len(inside_points)
        total_outside = len(outside_points)
        inside_snow = total_inside - len(non_snow_inside_points)
        final_count = len(final_point_cloud)
        
        # 计算当前文件各项时间（毫秒）
        total_ms = (time.time() - file_start_time) * 1000
        graph_total_ms = sum(graph_timings.values()) * 1000
        recon_total_ms = (mask_time + 
                         (time.time() - (file_start_time + graph_total_ms/1000 + inference_time)) ) * 1000
        
        print(f"\n📁 {file}:")
        print(f"   Points: total={total_points}, inside={total_inside}, outside={total_outside}, "
              f"inside_snow={inside_snow}, final={final_count}")
        print(f"   Times: total={total_ms:.2f}ms | graph={graph_total_ms:.2f}ms | "
              f"inf={inference_time*1000:.2f}ms | recon={recon_total_ms:.2f}ms")
        
        # 打印当前文件的图构建详细时间
        print(f"   📊 Graph details:")
        graph_details = [
            ('remove_duplicates', '去重'),
            ('filter_points', '滤波'),
            ('to_tensor', '转Tensor'),
            ('downsample', '降采样'),
            ('create_assignments', '聚类'),
            ('compute_node_features', '特征计算'),
            ('kdtree_search', 'KDTree'),
            ('build_edges', '边构建'),
            ('construct_features', '特征构建'),
            ('create_data', '创建Data')
        ]
        for step_key, step_name in graph_details:
            if step_key in graph_timings:
                step_time = graph_timings[step_key] * 1000
                print(f"      {step_name:15s}: {step_time:6.2f}ms")
    
    # 打印总体统计
    print_timing_summary()
    print(f"\n✅ Processing complete. Results saved to {point_saving_path}")