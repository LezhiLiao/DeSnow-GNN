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
from registration import registration
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KDTree
import time
num_clusters = 5000  # 聚类的大小
radius = 1
denoise_radius = 25
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
folder_path = f"/root/autodl-tmp/gnn_updata/inference/denoise_sample_pc/wads"  # 编辑bin文件目录
model_path = f"/root/autodl-tmp/gnn_updata/train/model/WADS_allfea_sample_model.pth"
point_saving_path = f"/root/autodl-tmp/gnn_updata/inference/point_saving_path/wads"
channel=4#WADS=7 CADC=4
distin_therehold=0.5
temporal_dim=3

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ###########################构建节点特征：根据空间信息#####################
        self.conv1_1 = GATConv(channel, channel)
        self.mlp1_1 = MLP(in_channels=3, hidden_channels=64, out_channels=3, num_layers=3)  # 自校准的mlp
        self.mlp1_2 = MLP(in_channels=channel, hidden_channels=64, out_channels=11, num_layers=3)  #
        self.mlp1_3 = MLP(in_channels=16, hidden_channels=64, out_channels=32, num_layers=3)
        self.mlp1_4 = MLP(in_channels=4, hidden_channels=64, out_channels=7, num_layers=3)#######zhehhhhh
        self.conv1_2 = GATConv(11, 16)
        self.lin1_1 = torch.nn.Linear(32, 1)
        self.bn1_1 = torch.nn.BatchNorm1d(channel)
        self.bn1_2 = torch.nn.BatchNorm1d(11)
        self.bn1_3 = torch.nn.BatchNorm1d(16)
        self.bn1_4 = torch.nn.BatchNorm1d(32)
        self.bn1_5 = torch.nn.BatchNorm1d(1)  # edge_weight的归一化
        ####################全局特征部分###########
        self.mlp1_5 = MLP(in_channels=3, hidden_channels=32, out_channels=3, num_layers=3)
        # self.attention = SelfAttention(3, 3)
        self.mlp1_6 = MLP(in_channels=3, hidden_channels=32, out_channels=3, num_layers=3)
        ###################第二层对准############
        self.mlp1_7 = MLP(in_channels=3, hidden_channels=32, out_channels=3, num_layers=3)
        self.mlp1_8 = MLP(in_channels=14, hidden_channels=32, out_channels=7, num_layers=3)
        self.mlp1_9 = MLP(in_channels=11, hidden_channels=32, out_channels=11, num_layers=3)
         # 时域特征处理（轻量版）
        self.feature_similarity = MLP(in_channels=11, hidden_channels=32, out_channels=1, num_layers=2)  # 仅用于筛选KNN结果
        self.temporal_mlp = MLP(in_channels=temporal_dim, hidden_channels=32, out_channels=temporal_dim, num_layers=2)
        self.motion_mlp = MLP(in_channels=1, hidden_channels=16, out_channels=11, num_layers=2)  # 输入差异得分
        self.GCN = GCNConv(11, 11)
        self.GAT = GATConv(11, 11)
        
        
        #######################新加的网络################
        self.mlp1_1_t = MLP(in_channels=3, hidden_channels=64, out_channels=3, num_layers=3)  # 自校准的mlp
        self.mlp1_4_t = MLP(in_channels=4, hidden_channels=64, out_channels=7, num_layers=3)
        self.conv1_1_t = GATConv(channel, channel)
        self.bn1_1_t = torch.nn.BatchNorm1d(channel)
        self.mlp1_2_t = MLP(in_channels=channel, hidden_channels=64, out_channels=11, num_layers=3)
        self.bn1_2_t = torch.nn.BatchNorm1d(11)
        self.mlp1_7_t = MLP(in_channels=3, hidden_channels=32, out_channels=3, num_layers=3)
        self.mlp1_8_t = MLP(in_channels=14, hidden_channels=32, out_channels=7, num_layers=3)
        self.conv1_2_t = GATConv(11, 16)
        self.bn1_3_t = torch.nn.BatchNorm1d(16)
        self.mlp1_3_t = MLP(in_channels=16, hidden_channels=64, out_channels=11, num_layers=3)
        self.bn1_4_t = torch.nn.BatchNorm1d(11)
        self.lin = torch.nn.Linear(11, 32)
        
    def forward(self, data, data_prev=None):
        
        # ######################构建节点特征：根据空间信息########################
        registration_dis1 = self.mlp1_1(data.x[:, 0:3])  # mlp
        registration_dis1 = registration(data.x, data.edge_index, registration_dis1)
        registration_dis1 = torch.cat((registration_dis1, data.x[:, 3:4][data.edge_index[1]]), dim=1)
        registration_dis1 = self.mlp1_4(registration_dis1)  # 求dege_weight
        registration_dis1 = torch.relu(registration_dis1)
        # ##############################传播##############################
        node_feature1 = self.conv1_1(data.x[:, 0:11], data.edge_index, edge_attr=registration_dis1)  # conv1
        node_feature1 = self.bn1_1(node_feature1)  # 正则1
        node_feature1 = self.mlp1_2(node_feature1)  # mlp2
        node_feature1 = self.bn1_2(torch.relu(node_feature1))  # 正则2
        del registration_dis1
        # 2. 时域信息融合
        if data_prev is not None:
            registration_dis1_t = self.mlp1_1_t(data_prev.x[:, 0:3])  # mlp
            registration_dis1_t = registration(data_prev.x, data_prev.edge_index, registration_dis1_t)
            registration_dis1_t = torch.cat((registration_dis1_t, data_prev.x[:, 3:4][data_prev.edge_index[1]]), dim=1)
            registration_dis1_t = self.mlp1_4_t(registration_dis1_t)  # 求dege_weight
            registration_dis1_t = torch.relu(registration_dis1_t)
            # ##############################传播##############################
            node_feature1_t = self.conv1_1_t(data_prev.x, data_prev.edge_index, edge_attr=registration_dis1_t)  # conv1
            node_feature1_t = self.bn1_1_t(node_feature1_t)  # 正则1
            node_feature1_t = self.mlp1_2_t(node_feature1_t)  # mlp2
            node_feature1_t = self.bn1_2_t(torch.relu(node_feature1_t))  # 正则2
            del registration_dis1_t
            registration_dis2_t = self.mlp1_7_t(data_prev.x[:, 0:3])  # mlp
            registration_dis2_t = registration(data_prev.x, data_prev.edge_index, registration_dis2_t)
            registration_dis2_t = torch.cat((registration_dis2_t, node_feature1_t[data_prev.edge_index[1]]), dim=1)
            registration_dis2_t = self.mlp1_8_t(registration_dis2_t)  # 求dege_weight
            registration_dis2_t = torch.relu(registration_dis2_t)
            ####################################################################
            node_feature1_t = self.conv1_2_t(node_feature1_t, data_prev.edge_index, edge_attr=registration_dis2_t)  # conv2
            del registration_dis2_t
            node_feature1_t = self.bn1_3_t(node_feature1_t)  # 正则3
            node_feature1_t = self.mlp1_3_t(node_feature1_t)  # mlp3
            node_feature1_t = self.bn1_4_t(torch.relu(node_feature1_t))  # 正则4
            
            x=data.x
            # Step 1: 质心对齐
            delta_centroid = x[:, 0:3].mean(dim=0) - data_prev.x[:, 0:3].mean(dim=0)
            corrected_pos = x[:, 0:3] - delta_centroid  # 修正后的当前帧坐标

            # Step 2: knn搜索 (修正后的位置 vs 上一帧的位置)
            knn_edge_index = knn(x=data_prev.x[:, 0:3], y=corrected_pos, k=3)
            source_idx, target_idx = knn_edge_index  
            
            curr_features = node_feature1[target_idx]  - node_feature1_t[source_idx]                  
            similarity = self.feature_similarity(curr_features) 
            similarity = similarity.view(node_feature1.size(0), 3)               

            best_match_idx = similarity.argmax(dim=1)                 # [N]

            # 找到最佳匹配点在source_idx中的位置
            indices = best_match_idx + torch.arange(node_feature1.size(0), device=x.device) * 3  
            matched_prev_pos = node_feature1_t[source_idx[indices]]               

            # 计算相对位移
            delta_x = node_feature1 - matched_prev_pos  # [N, 3]
            
            motion_feat = self.GCN(delta_x,data.edge_index)
            motion_feat = self.mlp1_9(motion_feat)
            motion_gate = torch.sigmoid(motion_feat)  # [N, 11]
            node_feature1 = node_feature1 * motion_gate
            
                    ##############################第二层################################
        registration_dis2 = self.mlp1_7(data.x[:, 0:3])  # mlp
        registration_dis2 = registration(data.x, data.edge_index, registration_dis2)
        registration_dis2 = torch.cat((registration_dis2, node_feature1[data.edge_index[1]]), dim=1)
        registration_dis2 = self.mlp1_8(registration_dis2)  # 求dege_weight
        registration_dis2 = torch.relu(registration_dis2)
        ####################################################################
        node_feature1 = self.conv1_2(node_feature1, data.edge_index, edge_attr=registration_dis2)  # conv2
        del registration_dis2
        node_feature1 = self.bn1_3(node_feature1)  # 正则3
        node_feature1 = self.mlp1_3(node_feature1)  # mlp3
        node_feature1 = self.bn1_4(torch.relu(node_feature1))  # 正则4
        node_feature1 = torch.sigmoid(self.lin1_1(node_feature1))  # 输出MSE
        # node_feature1 = self.lin1_1(node_feature1)
        return node_feature1


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
    # 使用cdist计算距离矩阵
    distances = torch.cdist(points_tensor_cuda, cent_cuda_tensor, p=2).pow(2)
    assignment = torch.argmin(distances, dim=1)
    min_distances = distances.gather(1, assignment.unsqueeze(1)).squeeze(1)
    time1=time.time()
    # 使用scatter_add统计
    avg_intensity = torch.zeros(cent_cuda_tensor.size(0), device='cuda')
    avg_intensity.scatter_add_(0, assignment, intensity_cuda_tensor)
    time2=time.time()
    avg_distance = torch.zeros(cent_cuda_tensor.size(0), device='cuda')
    counts = torch.zeros(cent_cuda_tensor.size(0), device='cuda')
    avg_distance.scatter_add_(0, assignment, min_distances)
    counts.scatter_add_(0, assignment, torch.ones_like(min_distances))
    time3=time.time()
    # 避免除零
    mask = counts > 0
    avg_distance[mask] /= counts[mask]
    avg_intensity[mask] /= counts[mask]
    # print(f" I_time: {(time2-time1) * 1000:.2f}ms  |    Dp_time: {(time3-time2) * 1000:.2f}ms")
    return assignment, avg_distance, avg_intensity, counts

def gpu_radius_neighbors(cent_cuda_tensor, radius):
    # 计算平方距离矩阵（避免开方，提升性能）
    sq_distances = torch.cdist(cent_cuda_tensor, cent_cuda_tensor, p=2).pow(2)
    sq_distances_clone = sq_distances.clone()  # 创建副本用于计算真实距离
    sq_distances.fill_diagonal_(float('inf'))  # 排除自身
    
    # 直接筛选半径内的点（使用平方距离比较，避免计算平方根）
    mask = sq_distances <= radius ** 2
    # 获取所有满足条件的边 (i,j) 的索引
    src, dst = torch.where(mask)
    
    # 组合成 edge_list [2, edge_num]
    edge_list = torch.stack([src, dst], dim=0)
    
    # 计算真实距离（对平方距离开方）
    distances = torch.sqrt(sq_distances_clone[mask])
    
    # 计算每个顶点与所有邻居的平均距离
    num_vertices = cent_cuda_tensor.shape[0]
    sum_distances = torch.zeros(num_vertices, device=cent_cuda_tensor.device)
    neighbor_counts = torch.zeros(num_vertices, device=cent_cuda_tensor.device)
    
    # 使用scatter_add计算每个顶点的距离总和
    sum_distances.scatter_add_(0, src, distances)
    neighbor_counts.scatter_add_(0, src, torch.ones_like(distances))
    
    # 计算平均距离（避免除以零）
    mean_distances = sum_distances / (neighbor_counts + 1e-10)
    
    # 计算顶点到传感器的距离（假设传感器在原点）
    sensor_distances = torch.norm(cent_cuda_tensor, p=2, dim=1)
    
    # 计算Dp = 平均邻居距离 / 传感器距离
    Dp = mean_distances / (sensor_distances + 1e-10)  # 添加小量防止除零
    return edge_list, Dp

def indices_to_edge(result):
    """将邻居索引转换为边索引"""
    edge_list = []
    for i, neighbors in enumerate(result):
        for neighbor in neighbors:
            edge_list.append([i, neighbor])
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()

def compute_pca_features(points_tensor, k_neighbors=10):
    """使用协方差矩阵计算PCA特征（法向量）"""
    points_np = points_tensor.cpu().numpy() if torch.is_tensor(points_tensor) else points_tensor
    normals = np.zeros((len(points_np), 3))
    
    # 使用KDTree快速查找邻居
    tree = KDTree(points_np)
    
    for i, point in enumerate(points_np):
        # 将点重塑为(1, 3)形状
        query_point = point.reshape(1, -1)
        # 查找k个最近邻（包括点自身）
        distances, neighbor_indices = tree.query(query_point, k=k_neighbors)
        neighbors = points_np[neighbor_indices[0]]  # 注意取第一个结果
        
        if len(neighbors) < 3:
            normals[i] = np.array([0.0, 0.0, 1.0])
            continue
            
        # 计算协方差矩阵
        cov_matrix = np.cov(neighbors.T)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            # 最小特征值对应的特征向量即为法向量
            normals[i] = eigenvectors[:, np.argmin(eigenvalues)]
        except:
            normals[i] = np.array([0.0, 0.0, 1.0])
    
    return normals

def graph_construction(point, intensity):
    """构建点云图结构"""
    with torch.no_grad():
        # 距离筛选
        points_np, intensity, out_point = denoise_point_cloud(point, intensity)
        # 转换为tensor并移到GPU
        points_tensor = torch.tensor(points_np, dtype=torch.float32, device=device)
        intensity_cuda_tensor = torch.tensor(intensity, dtype=torch.float32, device=device)
        # 降采样和聚类
        centroids = downsample_point_cloud(points_tensor, num_clusters)
        assignment, avg_distance, avg_intensity, counts = create_ass_with_avg_distance_and_intensity(
            centroids, points_tensor, intensity_cuda_tensor
        )
        # 邻居搜索和图构建
        edge_index, Dp = gpu_radius_neighbors(centroids,  radius) 
        # normals
        # normals = compute_pca_features(centroids)
        # 构建特征矩阵
        features = torch.stack([
            centroids[:, 0],
            centroids[:, 1],
            centroids[:, 2],
            avg_intensity,
            Dp,
            avg_distance,
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
    model = Net()
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model

def gnn_inference(graph_data,prev_data,model):
    graph_data.to(device)
    with torch.no_grad():
        pred_vet = model(graph_data,prev_data)
    return pred_vet

def point_reconstruction(pred_vet,ass,out_point,in_point):
    pred = pred_vet.detach().cpu().numpy()
    assigment = ass.detach().cpu().numpy()

    indices = np.where(pred < distin_therehold)[0]
    mask = np.isin(assigment, indices)
    selected_indices = np.where(mask)[0]  # 符合条件的索引 [1, 2, 4, 5]

    # 用 selected_indices 选取 point 中的行
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
        intensity = intensity*5000
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
