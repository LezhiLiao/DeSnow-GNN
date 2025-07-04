from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from torch_geometric.nn import MLP,GATConv, GCNConv
import os
import re
from registration import registration
import torch.nn as nn
import torch.nn.functional as F
import time
from torch_cluster import knn
from torch_scatter import scatter_mean  # 用于邻域聚合

fil_rad=25
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



def evaluate(data,data_prev):
    model.eval()
    with torch.no_grad():
        pred = model(data,data_prev)
    return pred

def node_label_to_cloud(pred, assigment):
    pred=pred.detach().cpu().numpy()
    assigment=assigment.detach().cpu().numpy()
    pred_label = [0] * len(assigment)
    ass_total = list(set(assigment))
    ass_total_dic = {value: index for index, value in enumerate(ass_total)}
    for i in range(len(assigment)):
        index = ass_total_dic.get(assigment[i], None)
        if pred[index].item()>0.5:
            pred_label[i] = 110
    return pred_label

def Confusion(pred_label, points_label):
    pred_label = np.array(pred_label)
    points_label = np.array(points_label)

    tp = np.sum(np.logical_and(pred_label == 110, points_label == 110))
    fp = np.sum(np.logical_and(pred_label == 110, points_label != 110))
    fn = np.sum(np.logical_and(pred_label != 110, points_label == 110))
    tn = np.sum(np.logical_and(pred_label != 110, points_label != 110))

    return tp, fp, fn, tn

def loadpoint(id):
    outpointpath=f"/root/autodl-tmp/data_set"
    point = np.fromfile(f"{outpointpath}/al/velodyne/0{id}.bin", dtype=np.float32).reshape(-1, 4)
    labels = np.fromfile(f"{outpointpath}/al_lab/0{id}.label", dtype=np.uint32)
    return point,labels
    
def filter_labels_by_distance(point, labels, min_distance=20):
    distances = np.sqrt(point[:, 0]**2 + point[:, 1]**2 + point[:, 2]**2)
    mask = distances > min_distance
    filtered_labels = labels[mask]
    return filtered_labels
 
def outconfusion(label):
    fp = np.sum(label==110)
    tn = np.sum(label!= 110)
    return fp,tn
   

def classification_indicators(tp,fp,fn,tn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1 = 2* precision * recall /(precision+recall)
    return precision,recall, F1

def seq_result(metric):
    # 获取所有 batch_number（即key值），并按顺序排序
    batch_numbers = sorted(metric.keys())
    
    # 用来存储每个区域的平均指标
    region_metrics = {}
    
    # 初始化当前区域索引
    region = 0
    # 当前区域的batch_number的起始值
    start_value = 1
    
    # 遍历所有 batch_numbers
    for i, batch_number in enumerate(batch_numbers):
        # 计算当前批次所在区域的结束值（按每 101 为一段）
        end_value = start_value + 101
        
        # 如果当前 batch_number 超过了这个区间，则切换到下一个区域
        if batch_number >= end_value:
            region += 1
            start_value =  end_value # 更新新的区域起始值
            end_value = start_value + 101
        
        # 如果该区域还没有记录数据，则初始化
        if region not in region_metrics:
            region_metrics[region] = {
                'F1': [],
                'precision': [],
                'recall': []
            }

        # 收集当前区域的指标数据
        if batch_number in metric:
            region_metrics[region]['F1'].append(metric[batch_number]['F1'])
            region_metrics[region]['precision'].append(metric[batch_number]['precision'])
            region_metrics[region]['recall'].append(metric[batch_number]['recall'])

    # 计算并打印每个区域的平均值
    for region, metrics in region_metrics.items():
        avg_F1 = sum(metrics['F1']) / len(metrics['F1']) if metrics['F1'] else 0
        avg_precision = sum(metrics['precision']) / len(metrics['precision']) if metrics['precision'] else 0
        avg_recall = sum(metrics['recall']) / len(metrics['recall']) if metrics['recall'] else 0
        
        print(f"区域 {region + 1}:")
        print(f"  平均 F1: {avg_F1:.4f}")
        print(f"  平均 Precision: {avg_precision:.4f}")
        print(f"  平均 Recall: {avg_recall:.4f}")


def calculate_average_metrics(metrics_dict, exclude_ranges=None):
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    num_batches = 0

    for batch_number, metrics in metrics_dict.items():
        # 检查是否在任意一个排除区间内
        if exclude_ranges and any(start <= batch_number <= end for start, end in exclude_ranges):
            continue

        total_precision += metrics['precision']
        total_recall += metrics['recall']
        total_f1 += metrics['F1']
        num_batches += 1  # 只计入未排除的批次

    # 确保没有全部批次被排除，避免除以零
    if num_batches == 0:
        print("所有批次均被排除，无法计算平均值")
        return None

    average_precision = total_precision / num_batches
    average_recall = total_recall / num_batches
    average_f1 = total_f1 / num_batches

    print(f"  全局的平均 F1: {average_f1:.4f}")
    print(f"  全局的平均 Precision: {average_precision:.4f}")
    print(f"  全局的平均 Recall: {average_recall:.4f}")

  
    
model=Net()
model.load_state_dict(torch.load(f"/root/autodl-tmp/temporal_gnn2/train/model/fil_rad/allfea_v16_5k_filrad99_limit20_r327_50.pth"))
# model.load_state_dict(torch.load(f"/root/autodl-tmp/gnn2/train/model/4-8/allfe_v5_5k_limit20_r327_30.pth"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
folder_path=f"/root/autodl-tmp/graph_ab_test/fil_rad/99/"
metrics_dict = {}

# files = [f for f in os.listdir(str(folder_path+"yanz/data")) if f.endswith('.pt')]
files = [f for f in os.listdir(str(folder_path+"data")) if f.endswith('.pt')]
files.sort()
i=0
for file in files:
    data_prev=None
    # data_path = os.path.join(str(folder_path+"yanz/data"), file)
    data_path = os.path.join(str(folder_path+"data"), file)
    print("打开的data是",data_path)
    batch_number = int(re.search(r'\d+', data_path[57:66]).group())
    # batch_number = int(re.search(r'\d+', data_path[34:58]).group())
    data_test = torch.load(data_path)
    
    prev_path =  f"{folder_path}data/data_batch0{batch_number-1}.pt"
     # 尝试加载前一帧
    if os.path.exists(prev_path):
        data_prev = torch.load(prev_path)
        data_prev.x = data_prev.x.float()
        data_prev = data_prev.to(device)
        # print("打开的predata是",prev_path)
        # data_prev.x = torch.cat((data_prev.x[:, 0:5], data_prev.x[:, 6:8]), dim=1)
    else:
        data_prev = None
    
    data_test.x = data_test.x.float()    
    points_label = torch.load(f"{folder_path}/lab/label_batch0{batch_number}.pt")
    data_test = data_test.to(device)#导入待测试data数据
    
    # data_test.x = torch.cat((data_test.x[:, 0:5], data_test.x[:, 6:8]), dim=1)
    
    pred = evaluate(data_test,data_prev)#预测data数据类型及损失
    # pred = torch.argmax(pred, dim=1)
    # assigment = torch.load(str(folder_path+f"yanz/ass/assigment_batch{batch_number}.pt"))
    assigment = torch.load(str(folder_path+f"ass/assigment_batch0{batch_number}.pt"))
    pred_label = node_label_to_cloud(pred, assigment)
    tp, fp, fn, tn = Confusion(pred_label,points_label)
    #######################读原始数据，以补全半径以外部分#####
    outrawpoint,outrawlabel=loadpoint(batch_number)
    filtered_labels = filter_labels_by_distance(outrawpoint, outrawlabel, min_distance=fil_rad)
    outfp,outtn = outconfusion(filtered_labels)
    
    ######################################################
    i=i+1
    precision,recall, F1 = classification_indicators(tp,fp+outfp,fn,tn)
    print(precision,recall, F1)
    metrics_dict[i] = {
        'F1': F1,
        'precision': precision,
        'recall': recall
    }
seq_result(metrics_dict)
calculate_average_metrics(metrics_dict, exclude_ranges=[])
