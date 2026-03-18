import open3d as o3d
import torch
import numpy as np
from scipy.spatial import cKDTree
from torch_geometric.data import Data
import os
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

#Optional configurable settings
num_clusters = 5000  
vote = 0.5
denoise_radius = 25
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# Base path for training data
base_path = "/root/autodl-tmp/wads_2/train"
# base_path = "/root/autodl-tmp/wads"
save_path = f"/root/autodl-tmp/wads_2/train_graph_data"

def remove_column(array, column_to_remove):
    """Remove a specific column from a numpy array."""
    return np.delete(array, column_to_remove, axis=1)


def node_label(assignments, location_indices, points_labels, le):
    """Compute node labels based on voting of point labels within each cluster."""
    assignments_np = assignments.cpu().numpy()
    assignments_uni = np.unique(assignments_np[location_indices])
    ass_total = list(set(assignments_np))
    ass_total_dic = {value: index for index, value in enumerate(ass_total)}

    node_labels = np.zeros(le)
    for val in assignments_uni:
        indices = np.where(assignments_np == val)[0]
        snow_count = np.sum(points_labels[indices] == 110)
        if snow_count > vote * len(indices):
            node_labels[ass_total_dic[val]] = 1
    return node_labels.reshape(-1, 1)


def intensity_to_node_inten_and_cluister_neinum(assignments, intensity, le):
    """Convert point intensities to node intensities and count points per cluster."""
    assignments_np = assignments.cpu().numpy()
    intensity_np = intensity.cpu().numpy() if torch.is_tensor(intensity) else np.array(intensity)

    inten_node = np.zeros(le)
    cluster_number = np.zeros(le)
    np.add.at(inten_node, assignments_np, intensity_np)
    np.add.at(cluster_number, assignments_np, 1)
    inten_node = np.divide(inten_node, cluster_number, out=np.zeros_like(inten_node), where=cluster_number != 0)
    return inten_node.tolist(), cluster_number.tolist()


def downsample_point_cloud(points, num_points):
    """Randomly downsample point cloud to a fixed number of points."""
    num_points_input = points.size(0)
    if num_points_input <= num_points:
        return points
    indices = torch.randperm(num_points_input, device=points.device)[:num_points]
    return points[indices]


def aggregate_distances_and_counts(assignment, min_distances_global):
    """Aggregate distances and counts for each cluster."""
    unique_assignments = torch.unique(assignment)
    distance_sums = torch.zeros(num_clusters, device=device)
    counts = torch.zeros(num_clusters, device=device)

    for index in unique_assignments:
        mask = (assignment == index)
        distance_sums[index] = torch.sum(min_distances_global[mask])
        counts[index] = torch.sum(mask)

    averages = torch.zeros_like(distance_sums)
    non_zero = counts > 0
    averages[non_zero] = distance_sums[non_zero] / counts[non_zero]
    return averages.cpu(), counts.cpu()


def create_ass(cent_cuda_tensor, points_tensor_cuda, num):
    """Create assignments by finding nearest centroids for each point."""
    save = 1
    for _ in range(10):
        chunk_size = num // 10
        min_distances = []
        min_indices = []

        for i in range(10):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, num)
            dist_chunk = torch.cdist(points_tensor_cuda, cent_cuda_tensor[start:end])
            min_dist, min_idx = torch.min(dist_chunk, dim=1)
            min_distances.append(min_dist)
            min_indices.append(min_idx + start)

        min_distances_tensor = torch.stack(min_distances, dim=1)
        min_indices_tensor = torch.stack(min_indices, dim=1)
        min_dist_global, min_idx_global = torch.min(min_distances_tensor, dim=1)
        assignment = torch.gather(min_indices_tensor, 1, min_idx_global.unsqueeze(1)).squeeze()

        complete_tensor = torch.arange(num_clusters, device=device)
        missing = complete_tensor[~torch.isin(complete_tensor, assignment)]
        if len(missing) == 0:
            save = 0
            break

    distance_sums, counts = aggregate_distances_and_counts(assignment, min_dist_global)
    return assignment, save, distance_sums, counts


def indices_to_edge(result):
    """Convert neighbor indices to edge indices for graph construction."""
    edge_list = []
    for i, neighbors in enumerate(result):
        for neighbor in neighbors:
            edge_list.append([i, neighbor])
    return torch.tensor(edge_list, dtype=torch.long).t()


def remove_duplicates(points, intensity, labels):
    """Remove duplicate points from the point cloud."""
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


def compute_pca_features(indices, pointcloud):
    """Compute PCA features (minimum variance ratio and normals) for each node."""
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
            normals[i] = np.array([0.0, 0.0, 1.0])  # Default normal vector
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
            normals[i] = np.array([0.0, 0.0, 1.0])  # Default normal vector

    return min_variances, normals


def kd_tree_radius_neighbors(data, dis):
    """Find radius neighbors using KD-tree."""
    kdtree = cKDTree(data)
    neighbors = []

    # Ensure each point has at least 3 neighbors
    distances, indices = kdtree.query(data, k=4)  # Query 4 points (including self)
    avg_distances = np.mean(distances[:, 1:], axis=1)  # Exclude the first point (self)

    for i in range(len(data)):
        # Query with fixed radius
        neighbors_in_radius = kdtree.query_ball_point(data[i], r=1)
        neighbors_i = [idx for idx in neighbors_in_radius if idx != i]

        # If less than 3 neighbors in radius, use the 3 nearest neighbors (excluding self)
        if len(neighbors_i) < 3:
            neighbors_i = indices[i, 1:4].tolist()  # Take the 2nd-4th points (excluding self)

        neighbors.append(neighbors_i)

    return neighbors, avg_distances, indices


def denoise_point_cloud(points, intensity=None, labels=None, max_distance=30.0):
    """Filter points based on distance from origin."""
    # Calculate distance from origin (0,0,0) for each point
    distances = np.linalg.norm(points, axis=1)

    # Create mask to keep points within max_distance
    mask = distances <= max_distance

    # Apply filtering
    filtered_points = points[mask]
    filtered_intensity = intensity[mask] if intensity is not None else None
    filtered_labels = labels[mask] if labels is not None else None

    return filtered_points, filtered_intensity, filtered_labels


def process_file(seq_folder, file, seq_name):
    """Process a single point cloud file and save as graph data."""
    # Construct file paths
    velodyne_path = os.path.join(seq_folder, "velodyne", file)
    seq_id = seq_name  # Use sequence name as identifier (e.g., '13', '15', etc.)
    file_id = file.replace('.bin', '')  # Remove .bin extension
    print(f"Processing sequence {seq_id}, file {file_id}")

    # Read point cloud data
    data = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)
    points_np = data[:, :3]
    intensity = data[:, 3]
    
    # Read corresponding label file
    label_path = os.path.join(seq_folder, "labels", f"{file_id}.label")
    labels = np.fromfile(label_path, dtype=np.uint32)

    # Remove duplicates
    points_np_uni, intensity_uni, labels_uni = remove_duplicates(points_np, intensity, labels)

    # Filter points within distance range (30 meters)
    points_np, intensity, labels = denoise_point_cloud(
        points_np_uni, intensity_uni, labels_uni, max_distance=denoise_radius
    )
    
    # Convert to tensor and move to GPU
    snow_indices = np.where(labels == 110)[0]
    points_tensor = torch.tensor(points_np, dtype=torch.float32, device=device)

    # Downsample and cluster
    centroids = downsample_point_cloud(points_tensor, num_clusters)
    assignments, save, dist_sums, counts = create_ass(centroids, points_tensor, num_clusters)
    
    # Compute node labels
    node_labels = node_label(assignments, snow_indices, labels, num_clusters)

    # Compute node features
    centroids_cpu = centroids.cpu()
    distances = torch.norm(centroids, p=2, dim=1).cpu()
    intensity_node, cluster_num = intensity_to_node_inten_and_cluister_neinum(assignments, intensity, num_clusters)

    # Build graph structure
    neighbors, avg_dist, _ = kd_tree_radius_neighbors(centroids_cpu.numpy(), distances.numpy())
    edge_index = indices_to_edge(neighbors)
    Dp = avg_dist / distances.numpy()

    # PCA features (commented out as in original code)
    # Rp, normals = compute_pca_features(neighbors, centroids_cpu)

    # Construct feature vector
    features = torch.stack([
        centroids_cpu[:, 0], centroids_cpu[:, 1], centroids_cpu[:, 2],
        torch.tensor(intensity_node),
        torch.tensor(Dp),
        # torch.tensor(Rp),
        dist_sums,
        counts,
    ], dim=1).float()

    # Create data object
    data = Data(x=features, edge_index=edge_index, y=torch.tensor(node_labels, dtype=torch.float32))
    
    # Save results
    if save == 0:
        # Ensure save directories exist
        os.makedirs(f"{save_path}/data", exist_ok=True)
        os.makedirs(f"{save_path}/as", exist_ok=True)
        os.makedirs(f"{save_path}/lab", exist_ok=True)
        os.makedirs(f"{save_path}/point", exist_ok=True)
        os.makedirs(f"{save_path}/vet", exist_ok=True)
        
        torch.save(data, f"{save_path}/data/data{file_id}.pt")
        torch.save(assignments, f"{save_path}/as/assigment{file_id}.pt")
        torch.save(labels, f"{save_path}/lab/label{file_id}.pt")
        torch.save(points_tensor.cpu(), f"{save_path}/point/points{file_id}.pt")
        torch.save(intensity, f"{save_path}/vet/vet{file_id}.pt")


# Main processing
if __name__ == "__main__":
    
    # Get all sequence folders (e.g., '13', '15', '17', ...)
    seq_folders = sorted([f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))])
    
    print(f"Found sequence folders: {seq_folders}")
    
    # Process each sequence folder
    for seq_name in seq_folders:
        seq_folder = os.path.join(base_path, seq_name)
        velodyne_folder = os.path.join(seq_folder, "velodyne")
        
        if not os.path.exists(velodyne_folder):
            print(f"Warning: {velodyne_folder} does not exist, skipping...")
            continue
            
        # Get all .bin files in velodyne folder
        files = sorted([f for f in os.listdir(velodyne_folder) if f.endswith('.bin')])
        
        print(f"Processing sequence {seq_name} with {len(files)} files")
        
        # Process each file in the sequence
        for file in files:
            process_file(seq_folder, file, seq_name)