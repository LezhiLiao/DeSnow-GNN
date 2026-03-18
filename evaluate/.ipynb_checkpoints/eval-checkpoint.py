"""
DeSnow-GNN Integrated Evaluation Script
Directly processes raw point cloud data from val directory and computes metrics
"""

import os
import re
import numpy as np
import torch
from torch_geometric.data import Data
import sys
from scipy.spatial import cKDTree
from torch_cluster import knn
from sklearn.decomposition import PCA

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.network import DeSnowGNN


# ==================== CONFIGURATION ====================
# Paths
MODEL_PATH = "/root/autodl-tmp/DeSnow-GNN/checkpoint/wads_120260314_164129_seed1_final.pth"
VAL_DATA_PATH = "/root/autodl-tmp/wads_2/val"  # Direct path to val data

# Parameters
NUM_CLUSTERS = 5000          # Number of clusters for downsampling
VOTE_THRESHOLD = 0.5         # Voting threshold for node labeling
FILTER_RADIUS = 25           # Distance filter radius (same as denoise_radius)
KNN_NEIGHBORS = 3            # Number of KNN neighbors for graph construction
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output
PRINT_DETAILED = True        # Print per-file results
# ========================================================


# ==================== UTILITY FUNCTIONS ====================
def remove_duplicates(points, intensity, labels):
    """Remove duplicate points from point cloud"""
    seen = set()
    unique_points = []
    unique_intensity = []
    unique_labels = []

    for point, inten, label in zip(points, intensity, labels):
        point_tuple = tuple(point.tolist() if torch.is_tensor(point) else point)
        if point_tuple not in seen:
            seen.add(point_tuple)
            unique_points.append(point)
            unique_intensity.append(inten)
            unique_labels.append(label)

    return (torch.stack(unique_points) if torch.is_tensor(points) else np.array(unique_points),
            np.array(unique_intensity),
            np.array(unique_labels))


def denoise_point_cloud(points, intensity=None, labels=None, max_distance=30.0):
    """Filter points based on distance from origin"""
    distances = np.linalg.norm(points, axis=1)
    mask = distances <= max_distance
    
    filtered_points = points[mask]
    filtered_intensity = intensity[mask] if intensity is not None else None
    filtered_labels = labels[mask] if labels is not None else None
    
    return filtered_points, filtered_intensity, filtered_labels


def downsample_point_cloud(points, num_points):
    """Randomly downsample point cloud to fixed number of points"""
    num_points_input = points.size(0)
    if num_points_input <= num_points:
        return points
    indices = torch.randperm(num_points_input, device=points.device)[:num_points]
    return points[indices]


def aggregate_distances_and_counts(assignment, min_distances_global, num_clusters):
    """Aggregate distances and counts for each cluster"""
    unique_assignments = torch.unique(assignment)
    distance_sums = torch.zeros(num_clusters, device=DEVICE)
    counts = torch.zeros(num_clusters, device=DEVICE)

    for index in unique_assignments:
        mask = (assignment == index)
        distance_sums[index] = torch.sum(min_distances_global[mask])
        counts[index] = torch.sum(mask)

    averages = torch.zeros_like(distance_sums)
    non_zero = counts > 0
    averages[non_zero] = distance_sums[non_zero] / counts[non_zero]
    return averages.cpu(), counts.cpu()


def create_assignments(centroids, points, num_clusters):
    """Create assignments by finding nearest centroids for each point"""
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

        complete_tensor = torch.arange(num_clusters, device=DEVICE)
        missing = complete_tensor[~torch.isin(complete_tensor, assignment)]
        if len(missing) == 0:
            save_flag = 0
            break

    distance_sums, counts = aggregate_distances_and_counts(assignment, min_dist_global, num_clusters)
    return assignment, save_flag, distance_sums, counts


def compute_node_labels(assignments, snow_indices, points_labels, num_clusters):
    """Compute node labels based on voting of point labels within each cluster"""
    assignments_np = assignments.cpu().numpy()
    assignments_uni = np.unique(assignments_np[snow_indices])
    ass_total = list(set(assignments_np))
    ass_total_dic = {value: index for index, value in enumerate(ass_total)}

    node_labels = np.zeros(num_clusters)
    for val in assignments_uni:
        indices = np.where(assignments_np == val)[0]
        snow_count = np.sum(points_labels[indices] == 110)
        if snow_count > VOTE_THRESHOLD * len(indices):
            node_labels[ass_total_dic[val]] = 1
    return node_labels.reshape(-1, 1)


def intensity_to_node_features(assignments, intensity, num_clusters):
    """Convert point intensities to node intensities and count points per cluster"""
    assignments_np = assignments.cpu().numpy()
    intensity_np = intensity.cpu().numpy() if torch.is_tensor(intensity) else np.array(intensity)

    inten_node = np.zeros(num_clusters)
    cluster_count = np.zeros(num_clusters)
    np.add.at(inten_node, assignments_np, intensity_np)
    np.add.at(cluster_count, assignments_np, 1)
    inten_node = np.divide(inten_node, cluster_count, out=np.zeros_like(inten_node), where=cluster_count != 0)
    return inten_node.tolist(), cluster_count.tolist()


def kd_tree_radius_neighbors(data, dis):
    """Build graph using radius-based neighbor search (matching preact.py)"""
    kdtree = cKDTree(data)
    neighbors = []

    # Query 4 points (including self) to ensure at least 3 neighbors
    distances, indices = kdtree.query(data, k=4)
    avg_distances = np.mean(distances[:, 1:], axis=1)  # Exclude self

    for i in range(len(data)):
        # Use fixed radius query (r=1 as in preact.py)
        neighbors_in_radius = kdtree.query_ball_point(data[i], r=1)
        neighbors_i = [idx for idx in neighbors_in_radius if idx != i]

        # If less than 3 neighbors in radius, use 3 nearest neighbors (excluding self)
        if len(neighbors_i) < 3:
            neighbors_i = indices[i, 1:4].tolist()  # Take 2nd-4th points (excluding self)

        neighbors.append(neighbors_i)

    return neighbors, avg_distances


def indices_to_edge_index(neighbors):
    """Convert neighbor indices to edge indices for PyG"""
    edge_list = []
    for i, neighbor_list in enumerate(neighbors):
        for neighbor in neighbor_list:
            edge_list.append([i, neighbor])
    return torch.tensor(edge_list, dtype=torch.long).t()


def compute_pca_features(indices, pointcloud):
    """Compute PCA features for each node (matching preact.py but commented out)"""
    pointcloud_np = pointcloud.cpu().numpy()
    min_variances = np.zeros(len(indices))
    normals = np.zeros((len(indices), 3))

    for i, neighbor_indices in enumerate(indices):
        neighbors = pointcloud_np[neighbor_indices].reshape(-1, 3)
        n_samples = neighbors.shape[0]

        # Dynamically adjust n_components based on number of neighbors
        n_components = min(3, n_samples)

        if n_components < 1:  # If no neighbors, use default values
            min_variances[i] = 0.0
            normals[i] = np.array([0.0, 0.0, 1.0])  # Default normal
            continue

        try:
            pca = PCA(n_components=n_components)
            pca.fit(neighbors)
            min_variances[i] = np.min(pca.explained_variance_ratio_)

            # If n_components < 3, pad with zeros
            if n_components == 3:
                normals[i] = pca.components_[-1]
            else:
                temp_normal = np.zeros(3)
                temp_normal[:n_components] = pca.components_[-1]
                normals[i] = temp_normal
        except:
            min_variances[i] = 0.0
            normals[i] = np.array([0.0, 0.0, 1.0])  # Default normal

    return min_variances, normals


def pointcloud_to_graph(points_np, intensity, labels):
    """
    Convert raw point cloud to graph data format (matching preact.py exactly)
    
    Args:
        points_np: Nx3 numpy array of point coordinates
        intensity: Numpy array of intensity values
        labels: Numpy array of point labels
    
    Returns:
        data: PyG Data object
        assignments: Point-to-node assignments
        filtered_labels: Labels after filtering (for evaluation)
        save_flag: Whether clustering was successful
    """
    # Remove duplicates
    points_np_uni, intensity_uni, labels_uni = remove_duplicates(points_np, intensity, labels)

    # Filter points within distance range
    filtered_points, filtered_intensity, filtered_labels = denoise_point_cloud(
        points_np_uni, intensity_uni, labels_uni, max_distance=FILTER_RADIUS
    )
    
    # Convert to tensor and move to device
    snow_indices = np.where(filtered_labels == 110)[0]
    points_tensor = torch.tensor(filtered_points, dtype=torch.float32, device=DEVICE)

    # Downsample and cluster
    centroids = downsample_point_cloud(points_tensor, NUM_CLUSTERS)
    assignments, save_flag, dist_sums, counts = create_assignments(centroids, points_tensor, NUM_CLUSTERS)
    
    # Compute node labels
    node_labels = compute_node_labels(assignments, snow_indices, filtered_labels, NUM_CLUSTERS)

    # Compute node features
    centroids_cpu = centroids.cpu()
    centroid_distances = torch.norm(centroids, p=2, dim=1).cpu()
    intensity_node, cluster_counts = intensity_to_node_features(assignments, filtered_intensity, NUM_CLUSTERS)

    # Build graph structure using radius-based neighbor search (matching preact.py)
    neighbors, avg_dist = kd_tree_radius_neighbors(centroids_cpu.numpy(), centroid_distances.numpy())
    edge_index = indices_to_edge_index(neighbors)
    Dp = avg_dist / (centroid_distances.numpy() + 1e-8)  # Avoid division by zero

    # NOTE: PCA features are computed in preact.py but commented out
    # Uncomment the following lines if PCA features are needed
    # Rp, normals = compute_pca_features(neighbors, centroids_cpu)

    # Construct feature vector (7 dimensions: x,y,z,intensity,Dp,dist_sums,counts)
    features = torch.stack([
        centroids_cpu[:, 0], centroids_cpu[:, 1], centroids_cpu[:, 2],
        torch.tensor(intensity_node),
        torch.tensor(Dp),
        # torch.tensor(Rp),  # Uncomment if PCA features are needed
        dist_sums,
        counts,
    ], dim=1).float()

    # Uncomment to add normals if needed
    # features = torch.cat([
    #     features,
    #     torch.tensor(normals)
    # ], dim=1)

    # Create data object
    data = Data(x=features, edge_index=edge_index, y=torch.tensor(node_labels, dtype=torch.float32))
    
    return data, assignments, filtered_labels, save_flag


# ==================== EVALUATION FUNCTIONS ====================
def evaluate(model, data, data_prev=None):
    """Evaluate model on a single data sample"""
    model.eval()
    with torch.no_grad():
        data = data.to(DEVICE)
        if data_prev is not None:
            data_prev = data_prev.to(DEVICE)
        # pred = torch.sigmoid(model(data, data_prev))
        pred = model(data, data_prev)
    return pred


def node_label_to_cloud(pred, assignment, threshold=0.85):
    """Convert node-level predictions to point-level labels"""
    pred = pred.detach().cpu().numpy()
    assignment = assignment.detach().cpu().numpy()
    
    pred_label = [0] * len(assignment)
    ass_total = list(set(assignment))
    ass_total_dic = {value: index for index, value in enumerate(ass_total)}
    
    for i in range(len(assignment)):
        index = ass_total_dic.get(assignment[i], None)
        if index is not None and pred[index].item() > threshold:
            pred_label[i] = 110
    
    return pred_label


def confusion_matrix(pred_labels, true_labels):
    """Compute confusion matrix elements"""
    pred_labels = np.array(pred_labels)
    true_labels = np.array(true_labels)
    
    tp = np.sum(np.logical_and(pred_labels == 110, true_labels == 110))
    fp = np.sum(np.logical_and(pred_labels == 110, true_labels != 110))
    fn = np.sum(np.logical_and(pred_labels != 110, true_labels == 110))
    tn = np.sum(np.logical_and(pred_labels != 110, true_labels != 110))
    
    return tp, fp, fn, tn


def compute_metrics(tp, fp, fn, tn):
    """Compute precision, recall, and F1 score"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


def load_raw_data(seq_path, file_id):
    """
    Load raw point cloud and labels for a specific file
    
    Args:
        seq_path: Path to sequence folder
        file_id: File ID (e.g., '000001')
    
    Returns:
        tuple: (points, labels)
    """
    velodyne_path = os.path.join(seq_path, "velodyne", f"{file_id}.bin")
    label_path = os.path.join(seq_path, "labels", f"{file_id}.label")
    
    points = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)
    labels = np.fromfile(label_path, dtype=np.uint32)
    
    return points, labels


def filter_outside_radius(points, labels, min_distance=25):
    """
    Filter points outside the radius to account for points not processed by the model
    
    Args:
        points: Raw point cloud (N, 4)
        labels: Raw labels
        min_distance: Minimum distance threshold
    
    Returns:
        fp, tn for points outside radius
    """
    distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
    mask = distances > min_distance
    outside_labels = labels[mask]
    
    fp = np.sum(outside_labels == 110)
    tn = np.sum(outside_labels != 110)
    
    return fp, tn


def get_sequence_files(seq_path):
    """Get all .bin files in a sequence's velodyne folder"""
    velodyne_path = os.path.join(seq_path, "velodyne")
    if not os.path.exists(velodyne_path):
        return []
    
    files = [f.replace('.bin', '') for f in os.listdir(velodyne_path) if f.endswith('.bin')]
    return sorted(files)


def seq_result(metrics_dict):
    """Print results by region (每101个文件为一个区域)"""
    batch_numbers = sorted(metrics_dict.keys())
    region_metrics = {}
    region = 0
    start_value = 1
    
    for batch_number in batch_numbers:
        end_value = start_value + 101
        
        if batch_number >= end_value:
            region += 1
            start_value = end_value
            end_value = start_value + 101
        
        if region not in region_metrics:
            region_metrics[region] = {
                'F1': [],
                'precision': [],
                'recall': []
            }
        
        if batch_number in metrics_dict:
            region_metrics[region]['F1'].append(metrics_dict[batch_number]['F1'])
            region_metrics[region]['precision'].append(metrics_dict[batch_number]['precision'])
            region_metrics[region]['recall'].append(metrics_dict[batch_number]['recall'])
    
    print("\n" + "="*50)
    print("RESULTS BY REGION")
    print("="*50)
    for region, metrics in region_metrics.items():
        avg_F1 = sum(metrics['F1']) / len(metrics['F1']) if metrics['F1'] else 0
        avg_precision = sum(metrics['precision']) / len(metrics['precision']) if metrics['precision'] else 0
        avg_recall = sum(metrics['recall']) / len(metrics['recall']) if metrics['recall'] else 0
        
        print(f"Region {region + 1}:")
        print(f"  Avg F1: {avg_F1:.4f}")
        print(f"  Avg Precision: {avg_precision:.4f}")
        print(f"  Avg Recall: {avg_recall:.4f}")


def calculate_average_metrics(metrics_dict, exclude_ranges=None):
    """Calculate overall average metrics"""
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    num_batches = 0

    for batch_number, metrics in metrics_dict.items():
        if exclude_ranges and any(start <= batch_number <= end for start, end in exclude_ranges):
            continue

        total_precision += metrics['precision']
        total_recall += metrics['recall']
        total_f1 += metrics['F1']
        num_batches += 1

    if num_batches == 0:
        print("All batches excluded, cannot compute averages")
        return None

    avg_precision = total_precision / num_batches
    avg_recall = total_recall / num_batches
    avg_f1 = total_f1 / num_batches

    print("\n" + "="*50)
    print("OVERALL AVERAGES:")
    print(f"  Avg F1: {avg_f1:.4f}")
    print(f"  Avg Precision: {avg_precision:.4f}")
    print(f"  Avg Recall: {avg_recall:.4f}")
    
    return avg_precision, avg_recall, avg_f1


# ==================== MAIN EVALUATION ====================
def main():
    """Main evaluation function"""
    
    print("=" * 70)
    print("DeSnow-GNN Integrated Evaluation")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")
    print(f"Validation data: {VAL_DATA_PATH}")
    print(f"Parameters: NUM_CLUSTERS={NUM_CLUSTERS}, FILTER_RADIUS={FILTER_RADIUS}")
    print("=" * 70)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return
    
    # Load model
    print("\nLoading model...")
    model = DeSnowGNN()
    
    # Load checkpoint and extract model_state_dict
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model state dict from checkpoint")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'loss' in checkpoint:
            print(f"  Loss: {checkpoint['loss']:.6f}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model state dict directly")
    
    model.to(DEVICE)
    print("Model loaded successfully")
    
    # Get validation sequences
    if not os.path.exists(VAL_DATA_PATH):
        print(f"ERROR: Validation data path not found: {VAL_DATA_PATH}")
        return
        
    seq_folders = sorted([f for f in os.listdir(VAL_DATA_PATH) 
                          if os.path.isdir(os.path.join(VAL_DATA_PATH, f))])
    print(f"\nFound validation sequences: {seq_folders}")
    
    # Store all metrics
    all_metrics = {}
    global_file_counter = 0
    
    # Evaluate each sequence
    for seq_name in seq_folders:
        print(f"\n{'='*50}")
        print(f"Evaluating sequence: {seq_name}")
        print(f"{'='*50}")
        
        seq_path = os.path.join(VAL_DATA_PATH, seq_name)
        file_ids = get_sequence_files(seq_path)
        
        if not file_ids:
            print(f"  Warning: No files found in {seq_path}/velodyne")
            continue
            
        print(f"  Found {len(file_ids)} files")
        
        seq_metrics = {}
        prev_data = None
        prev_file_id = None
        
        for i, file_id in enumerate(file_ids):
            try:
                print(f"\r  Processing {file_id}...", end="")
                
                # Load raw data
                raw_points, raw_labels = load_raw_data(seq_path, file_id)
                points_np = raw_points[:, :3]
                intensity = raw_points[:, 3]
                
                # Convert to graph - now matches preact.py preprocessing exactly
                data, assignments, filtered_labels, save_flag = pointcloud_to_graph(points_np, intensity, raw_labels)
                
                if save_flag != 0:
                    print(f"\n  Warning: Clustering incomplete for {file_id}, skipping...")
                    continue
                
                # Prepare data
                data.x = data.x.float()
                
                # Evaluate
                pred = evaluate(model, data, prev_data)
                
                # Convert predictions to point labels (for filtered points)
                pred_labels_filtered = node_label_to_cloud(pred, assignments)
                
                # Compute confusion matrix on filtered points
                tp, fp, fn, tn = confusion_matrix(pred_labels_filtered, filtered_labels)
                
                # Get points outside radius and their labels
                out_fp, out_tn = filter_outside_radius(raw_points, raw_labels, min_distance=FILTER_RADIUS)
                
                # Add outside points to FP and TN (as in original code)
                total_fp = fp + out_fp
                total_tn = tn + out_tn
                
                # Compute metrics with outside points included
                precision, recall, f1 = compute_metrics(tp, total_fp, fn, total_tn)
                
                global_file_counter += 1
                
                # Store metrics
                seq_metrics[file_id] = {
                    'precision': precision,
                    'recall': recall,
                    'F1': f1,
                    'file_num': global_file_counter
                }
                
                all_metrics[global_file_counter] = {
                    'precision': precision,
                    'recall': recall,
                    'F1': f1
                }
                
                if PRINT_DETAILED:
                    print(f"\r  File {file_id}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f} | TP={tp}, FP={fp}+{out_fp}, FN={fn}, TN={tn}+{out_tn}")
                
                # Update previous data for next frame
                prev_data = data
                prev_file_id = file_id
                
            except Exception as e:
                print(f"\n  Error processing {file_id}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Compute sequence averages
        if seq_metrics:
            precisions = [m['precision'] for m in seq_metrics.values()]
            recalls = [m['recall'] for m in seq_metrics.values()]
            f1s = [m['F1'] for m in seq_metrics.values()]
            
            avg_precision = np.mean(precisions)
            avg_recall = np.mean(recalls)
            avg_f1 = np.mean(f1s)
            std_f1 = np.std(f1s)
            
            print(f"\n\nSequence {seq_name} Results:")
            print(f"  Files processed: {len(seq_metrics)}/{len(file_ids)}")
            print(f"  Precision: {avg_precision:.4f} ± {np.std(precisions):.4f}")
            print(f"  Recall: {avg_recall:.4f} ± {np.std(recalls):.4f}")
            print(f"  F1: {avg_f1:.4f} ± {std_f1:.4f}")
    
    # Print results by region and overall averages
    if all_metrics:
        seq_result(all_metrics)
        calculate_average_metrics(all_metrics, exclude_ranges=[])
    else:
        print("\nNo valid results obtained. Check your data and model.")


if __name__ == "__main__":
    main()