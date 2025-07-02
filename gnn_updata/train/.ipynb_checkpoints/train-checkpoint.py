from torch_geometric.loader import DataLoader
import numpy as np
import torch
from torch_geometric.nn import MLP, GraphConv, GATConv,GCNConv
import os
import re
from registration import registration
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch_cluster import knn
import time
import random
def set_random_seed(seed):
    # Python 随机种子
    random.seed(seed)
    # NumPy 随机种子
    np.random.seed(seed)
    # PyTorch 随机种子
    torch.manual_seed(seed)
    # 如果使用 GPU，设置 GPU 随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 确保每次训练结果的一致性（影响 CUDA 后端的行为）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机种子（设置为固定的值，保证复现性）
set_random_seed(327)  # 可以更改成你需要的任何值

from torch_scatter import scatter_mean  # 用于邻域聚合


channel=7
data_prev=None
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
        node_feature1 = self.conv1_1(data.x, data.edge_index, edge_attr=registration_dis1)  # conv1
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
            source_idx, target_idx = knn_edge_index  # source: prev点索引, target: current点索引
            
            curr_features = node_feature1[target_idx]  - node_feature1_t[source_idx]                   # [N*5, feature_dim]
            # print(curr_features)
            similarity = self.feature_similarity(curr_features)  # [N*5, 1]
            similarity = similarity.view(node_feature1.size(0), 3)                # [N, 5]

            best_match_idx = similarity.argmax(dim=1)                 # [N]

            # 找到最佳匹配点在source_idx中的位置
            indices = best_match_idx + torch.arange(node_feature1.size(0), device=x.device) * 3  # 注意batch偏移
            matched_prev_pos = node_feature1_t[source_idx[indices]]                # [N, 3]

            # Step 4: 计算相对位移
            delta_x = node_feature1 - matched_prev_pos  # [N, 3]
#             #新加入的
            
            motion_feat = self.GCN(delta_x,data.edge_index)
            motion_feat = self.mlp1_9(motion_feat)
#             # Step 5: 运动门控 (调整输入维度)
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


from torch.utils.data import Dataset  # Add this import

class CustomDataset(Dataset):  # Change inheritance to Dataset
    def __init__(self, folder_path, max_skip=1):
        self.folder_path = folder_path
        self.files = sorted(
            [f for f in os.listdir(folder_path) if f.endswith('.pt')],
            key=lambda x: int(x.split('.')[0][-5:])  # Assuming last 5 digits are numbers
        )
        self.max_skip = max_skip  # Maximum skip count
        
    def __len__(self):
        return len(self.files) - 1  # Theoretical maximum length

    def __getitem__(self, idx):
        for skip in range(self.max_skip + 1):
            if idx + skip + 1 >= len(self.files):
                raise StopIteration("Reached end of dataset")
                
            current_file = self.files[idx + skip + 1]
            prev_file = self.files[idx + skip]
            
            current_num = self._extract_num(current_file)
            prev_num = self._extract_num(prev_file)
            
            if current_num == prev_num + 1:
                current_data = torch.load(os.path.join(self.folder_path, current_file))
                prev_data = torch.load(os.path.join(self.folder_path, prev_file))
                return current_data, prev_data
                
        raise ValueError(f"Could not find consecutive frames within {self.max_skip} skips")

    def _extract_num(self, filename):
        """Extract numeric index from filename"""
        return int(filename.split('.')[0][-5:])  # Adjust based on actual format




def train(data,prev):
    data.x = data.x.float()
    prev.x = prev.x.float()
    # data.x = torch.cat((data.x[:, 0:5], data.x[:, 6:8]), dim=1)
    # prev.x = torch.cat((prev.x[:, 0:5], prev.x[:, 6:8]), dim=1)#rad_fil10和15需要这个
    model.train()
    optimizer.zero_grad()
    # Move input data to GPU
    node_feature = model(data,data_prev = prev)
    label = data.y
    # label = label.long().view(-1)
    loss = crit1(node_feature, label)
    loss.backward()
    optimizer.step()
    return loss.item()

folder_path = "/root/autodl-tmp/graph_ab_test/fil_rad/99/data"
dataset = CustomDataset(folder_path)
loader_train = DataLoader(dataset, batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(42))
model = Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

crit1 = torch.nn.MSELoss()
# class_weights = torch.tensor([1.0, 250 / 500], device=device)
# crit1 = nn.CrossEntropyLoss(weight=class_weights)

step_size = 8
gamma = 0.7
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
model.to(device)


for epoch in range(51):
    print("进入循环")
    loss_all = 0
    k=0
    for current, prev in loader_train:
        data = current.to(device)
        prev = prev.to(device)
        # print(len(data.x[0,:]))
        # data.x = torch.cat((data.x[:, 0:3], data.x[:, 5:11]), dim=1)
        loss = train(data,prev)
        loss_all += loss
        k=k+1
        # print(k)
        torch.cuda.empty_cache()
    k=0
    current_learning_rate = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch + 1}, Loss: {loss_all}, Learning Rate: {current_learning_rate}")
    scheduler.step()
    if epoch in [0, 10, 20, 30, 40, 50]:
        torch.save(model.state_dict(), f"/root/autodl-tmp/temporal_gnn2/train/model/fil_rad/allfea_v16_5k_filrad99_limit20_r327_{epoch}.pth")