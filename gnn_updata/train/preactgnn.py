import open3d as o3d
import torch
import numpy as np
from scipy.spatial import cKDTree
from torch_geometric.data import Data
import os
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

# 常量定义
# num_clusters = 5000  # 聚类的大小
# vote = 0.5
# denoise_radius = 20
num_clusters = 5000  # 聚类的大小
vote = 0.5
denoise_radius = 25
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def remove_column(array, column_to_remove):
    return np.delete(array, column_to_remove, axis=1)


def node_label(assignments, location_indices, points_labels, le):
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
    assignments_np = assignments.cpu().numpy()
    intensity_np = intensity.cpu().numpy() if torch.is_tensor(intensity) else np.array(intensity)

    inten_node = np.zeros(le)
    cluster_number = np.zeros(le)
    np.add.at(inten_node, assignments_np, intensity_np)
    np.add.at(cluster_number, assignments_np, 1)
    inten_node = np.divide(inten_node, cluster_number, out=np.zeros_like(inten_node), where=cluster_number != 0)
    return inten_node.tolist(), cluster_number.tolist()


def downsample_point_cloud(points, num_points):
    num_points_input = points.size(0)
    if num_points_input <= num_points:
        return points
    indices = torch.randperm(num_points_input, device=points.device)[:num_points]
    return points[indices]


def aggregate_distances_and_counts(assignment, min_distances_global):
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
    edge_list = []
    for i, neighbors in enumerate(result):
        for neighbor in neighbors:
            edge_list.append([i, neighbor])
    return torch.tensor(edge_list, dtype=torch.long).t()


def remove_duplicates(points, intensity, labels):
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
    pointcloud_np = pointcloud.cpu().numpy()
    min_variances = np.zeros(len(indices))
    normals = np.zeros((len(indices), 3))

    for i, neighbor_indices in enumerate(indices):
        neighbors = pointcloud_np[neighbor_indices].reshape(-1, 3)
        n_samples = neighbors.shape[0]

        # 根据邻居数量动态调整n_components
        n_components = min(3, n_samples)

        if n_components < 1:  # 如果没有邻居，使用默认值
            min_variances[i] = 0.0
            normals[i] = np.array([0.0, 0.0, 1.0])  # 默认法向量
            continue

        try:
            pca = PCA(n_components=n_components)
            pca.fit(neighbors)
            min_variances[i] = np.min(pca.explained_variance_ratio_)

            # 如果n_components < 3，补零处理
            if n_components == 3:
                normals[i] = pca.components_[-1]
            else:
                temp_normal = np.zeros(3)
                temp_normal[:n_components] = pca.components_[-1]
                normals[i] = temp_normal
        except:
            min_variances[i] = 0.0
            normals[i] = np.array([0.0, 0.0, 1.0])  # 默认法向量

    return min_variances, normals


def kd_tree_radius_neighbors(data, dis):
    kdtree = cKDTree(data)
    neighbors = []

    # 确保每个点至少有3个邻居
    distances, indices = kdtree.query(data, k=4)  # 查询4个点(包含自己)
    avg_distances = np.mean(distances[:, 1:], axis=1)  # 排除第一个点(自己)

    for i in range(len(data)):
        # 使用固定半径查询
        neighbors_in_radius = kdtree.query_ball_point(data[i], r=1)
        neighbors_i = [idx for idx in neighbors_in_radius if idx != i]

        # 如果半径内邻居不足3个，使用最近的3个邻居(排除自己)
        if len(neighbors_i) < 3:
            neighbors_i = indices[i, 1:4].tolist()  # 取第2-4个点(排除自己)

        neighbors.append(neighbors_i)

    return neighbors, avg_distances, indices


def denoise_point_cloud(points, intensity=None, labels=None, max_distance=30.0):
    # 计算每个点到原点(0,0,0)的距离
    distances = np.linalg.norm(points, axis=1)

    # 创建掩码，保留距离<=max_distance的点
    mask = distances <= max_distance

    # 应用过滤
    filtered_points = points[mask]
    filtered_intensity = intensity[mask] if intensity is not None else None
    filtered_labels = labels[mask] if labels is not None else None

    return filtered_points, filtered_intensity, filtered_labels


def process_file(file, i):
    folder_path = "/root/autodl-tmp/data_set/al/velodyne"
    bin_file_path = os.path.join(folder_path, file)
    id = bin_file_path[38:44]
    print(id)
    # 读取数据
    data = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)
    points_np = data[:, :3]
    intensity = data[:, 3]
    labels = np.fromfile(f"/root/autodl-tmp/data_set/al_lab/{id}.label", dtype=np.uint32)

    # 去重
    points_np_uni, intensity_uni, labels_uni = remove_duplicates(points_np, intensity, labels)

    # 距离筛选(30米范围内)
    points_np, intensity, labels = denoise_point_cloud(
        points_np_uni, intensity_uni, labels_uni, max_distance=denoise_radius
    )
    # 转换为tensor并移到GPU
    snow_indices = np.where(labels == 110)[0]
    points_tensor = torch.tensor(points_np, dtype=torch.float32, device=device)

    # 降采样和聚类
    centroids = downsample_point_cloud(points_tensor, num_clusters)
    assignments, save, dist_sums, counts = create_ass(centroids, points_tensor, num_clusters)
    # 计算节点标签
    node_labels = node_label(assignments, snow_indices, labels, num_clusters)

    # 计算节点特征
    centroids_cpu = centroids.cpu()
    distances = torch.norm(centroids, p=2, dim=1).cpu()
    intensity_node, cluster_num = intensity_to_node_inten_and_cluister_neinum(assignments, intensity, num_clusters)

    # 构建图结构
    neighbors, avg_dist, _ = kd_tree_radius_neighbors(centroids_cpu.numpy(), distances.numpy())
    edge_index = indices_to_edge(neighbors)
    Dp = avg_dist / distances.numpy()

    # 计算PCA特征
    # Rp, normals = compute_pca_features(neighbors, centroids_cpu)

    # 构建特征向量
    features = torch.stack([
        centroids_cpu[:, 0], centroids_cpu[:, 1], centroids_cpu[:, 2],
        torch.tensor(intensity_node),
        torch.tensor(Dp),
        # torch.tensor(Rp),
        dist_sums,
        counts,
    ], dim=1).float()
    # print(len(features[0,:]))
    # features = torch.cat([
    #     features,
    #     torch.tensor(normals)
    # ], dim=1)
    # 创建数据对象
    data = Data(x=features, edge_index=edge_index, y=torch.tensor(node_labels, dtype=torch.float32))
    # 保存结果
    if save == 0:
        save_path = f"/root/autodl-tmp/graph_ab_test/fil_rad/99"
        torch.save(data, f"{save_path}/data/data_batch{id}.pt")
        torch.save(assignments, f"{save_path}/ass/assigment_batch{id}.pt")
        torch.save(labels, f"{save_path}/lab/label_batch{id}.pt")
        torch.save(points_tensor.cpu(), f"{save_path}/point/points_batch{id}.pt")  # Fixed from {id} to {i}
        torch.save(intensity, f"{save_path}/vet/vet_batch{id}.pt")


# Main processing
if __name__ == "__main__":
    folder_path = "/root/autodl-tmp/data_set/al/velodyne"
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.bin')])

    # 使用enumerate获取索引
    for i, file in enumerate(files):
        process_file(file, i)  # 传入i
