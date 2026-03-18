import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP, GraphConv, GATConv, GCNConv
from model.registration import registration
from torch_scatter import scatter_mean

class DeSnowGNN(nn.Module):
    """
    Temporal Graph Attention Network for point cloud sequence processing.
    """
    def __init__(self, in_channels=7, temporal_dim=3):
        super(DeSnowGNN, self).__init__()  # 必须调用父类的构造函数
        
        self.in_channels = in_channels
        self.temporal_dim = temporal_dim
        
        ########################### Node Feature Construction based on Spatial Information #####################
        # Spatial feature extraction layers
        self.conv1_1 = GATConv(in_channels, in_channels)
        self.mlp1_1 = MLP(in_channels=3, hidden_channels=64, out_channels=3, num_layers=3)  # Self-calibration MLP
        self.mlp1_2 = MLP(in_channels=in_channels, hidden_channels=64, out_channels=11, num_layers=3)
        self.mlp1_3 = MLP(in_channels=16, hidden_channels=64, out_channels=32, num_layers=3)
        self.mlp1_4 = MLP(in_channels=4, hidden_channels=64, out_channels=7, num_layers=3)
        self.conv1_2 = GATConv(11, 16)
        self.lin1_1 = torch.nn.Linear(32, 1)
        self.bn1_1 = torch.nn.BatchNorm1d(in_channels)
        self.bn1_2 = torch.nn.BatchNorm1d(11)
        self.bn1_3 = torch.nn.BatchNorm1d(16)
        self.bn1_4 = torch.nn.BatchNorm1d(32)
        self.bn1_5 = torch.nn.BatchNorm1d(1)  # Normalization for edge_weight
        
        #################### Global Feature Part ###########
        self.mlp1_5 = MLP(in_channels=3, hidden_channels=32, out_channels=3, num_layers=3)
        self.mlp1_6 = MLP(in_channels=3, hidden_channels=32, out_channels=3, num_layers=3)
        
        ################### Second Layer Alignment ############
        self.mlp1_7 = MLP(in_channels=3, hidden_channels=32, out_channels=3, num_layers=3)
        self.mlp1_8 = MLP(in_channels=14, hidden_channels=32, out_channels=7, num_layers=3)
        self.mlp1_9 = MLP(in_channels=11, hidden_channels=32, out_channels=11, num_layers=3)
        
        # Temporal feature processing (lightweight version)
        self.feature_similarity = MLP(in_channels=11, hidden_channels=32, out_channels=1, num_layers=2)  # For KNN result filtering
        self.temporal_mlp = MLP(in_channels=temporal_dim, hidden_channels=32, out_channels=temporal_dim, num_layers=2)
        self.motion_mlp = MLP(in_channels=1, hidden_channels=16, out_channels=11, num_layers=2)  # Input difference score
        self.GCN = GCNConv(11, 11)
        self.GAT = GATConv(11, 11)
        
        ####################### Newly Added Networks ################
        self.mlp1_1_t = MLP(in_channels=3, hidden_channels=64, out_channels=3, num_layers=3)  # Self-calibration MLP
        self.mlp1_4_t = MLP(in_channels=4, hidden_channels=64, out_channels=7, num_layers=3)
        self.conv1_1_t = GATConv(in_channels, in_channels)
        self.bn1_1_t = torch.nn.BatchNorm1d(in_channels)
        self.mlp1_2_t = MLP(in_channels=in_channels, hidden_channels=64, out_channels=11, num_layers=3)
        self.bn1_2_t = torch.nn.BatchNorm1d(11)
        self.mlp1_7_t = MLP(in_channels=3, hidden_channels=32, out_channels=3, num_layers=3)
        self.mlp1_8_t = MLP(in_channels=14, hidden_channels=32, out_channels=7, num_layers=3)
        self.conv1_2_t = GATConv(11, 16)
        self.bn1_3_t = torch.nn.BatchNorm1d(16)
        self.mlp1_3_t = MLP(in_channels=16, hidden_channels=64, out_channels=11, num_layers=3)
        self.bn1_4_t = torch.nn.BatchNorm1d(11)
        self.lin = torch.nn.Linear(11, 32)
        
    def forward(self, data, data_prev=None):
        """
        Forward pass of the temporal GAT network.
        
        Args:
            data: Current frame graph data
            data_prev: Previous frame graph data (optional)
            
        Returns:
            Node feature predictions
        """
        ###################### Node Feature Construction based on Spatial Information ########################
        # First layer processing
        registration_dis1 = self.mlp1_1(data.x[:, 0:3])
        registration_dis1 = registration(data.x, data.edge_index, registration_dis1)
        registration_dis1 = torch.cat((registration_dis1, data.x[:, 3:4][data.edge_index[1]]), dim=1)
        registration_dis1 = self.mlp1_4(registration_dis1)  # Compute edge_weight
        registration_dis1 = torch.relu(registration_dis1)
        
        ############################## Message Propagation ##############################
        node_feature1 = self.conv1_1(data.x, data.edge_index, edge_attr=registration_dis1)
        node_feature1 = self.bn1_1(node_feature1)
        node_feature1 = self.mlp1_2(node_feature1)
        node_feature1 = self.bn1_2(torch.relu(node_feature1))
        del registration_dis1
        
        # 2. Temporal information fusion
        if data_prev is not None:
            # Process previous frame
            registration_dis1_t = self.mlp1_1_t(data_prev.x[:, 0:3])
            registration_dis1_t = registration(data_prev.x, data_prev.edge_index, registration_dis1_t)
            registration_dis1_t = torch.cat((registration_dis1_t, data_prev.x[:, 3:4][data_prev.edge_index[1]]), dim=1)
            registration_dis1_t = self.mlp1_4_t(registration_dis1_t)
            registration_dis1_t = torch.relu(registration_dis1_t)
            
            node_feature1_t = self.conv1_1_t(data_prev.x, data_prev.edge_index, edge_attr=registration_dis1_t)
            node_feature1_t = self.bn1_1_t(node_feature1_t)
            node_feature1_t = self.mlp1_2_t(node_feature1_t)
            node_feature1_t = self.bn1_2_t(torch.relu(node_feature1_t))
            del registration_dis1_t
            
            # Second layer for previous frame
            registration_dis2_t = self.mlp1_7_t(data_prev.x[:, 0:3])
            registration_dis2_t = registration(data_prev.x, data_prev.edge_index, registration_dis2_t)
            registration_dis2_t = torch.cat((registration_dis2_t, node_feature1_t[data_prev.edge_index[1]]), dim=1)
            registration_dis2_t = self.mlp1_8_t(registration_dis2_t)
            registration_dis2_t = torch.relu(registration_dis2_t)
            
            node_feature1_t = self.conv1_2_t(node_feature1_t, data_prev.edge_index, edge_attr=registration_dis2_t)
            del registration_dis2_t
            node_feature1_t = self.bn1_3_t(node_feature1_t)
            node_feature1_t = self.mlp1_3_t(node_feature1_t)
            node_feature1_t = self.bn1_4_t(torch.relu(node_feature1_t))
            
            # Temporal alignment and motion modeling
            x = data.x
            # Step 1: Centroid alignment
            delta_centroid = x[:, 0:3].mean(dim=0) - data_prev.x[:, 0:3].mean(dim=0)
            corrected_pos = x[:, 0:3] - delta_centroid  # Corrected current frame coordinates

            # Step 2: KNN search (corrected positions vs previous frame positions)
            from torch_cluster import knn
            knn_edge_index = knn(x=data_prev.x[:, 0:3], y=corrected_pos, k=3)
            source_idx, target_idx = knn_edge_index  # source: prev point index, target: current point index
            
            curr_features = node_feature1[target_idx] - node_feature1_t[source_idx]
            similarity = self.feature_similarity(curr_features)
            similarity = similarity.view(node_feature1.size(0), 3)

            best_match_idx = similarity.argmax(dim=1)

            # Find the best matching point in source_idx
            indices = best_match_idx + torch.arange(node_feature1.size(0), device=x.device) * 3
            matched_prev_pos = node_feature1_t[source_idx[indices]]

            # Step 4: Compute relative displacement
            delta_x = node_feature1 - matched_prev_pos
            
            # Motion feature processing
            motion_feat = self.GCN(delta_x, data.edge_index)
            motion_feat = self.mlp1_9(motion_feat)
            
            # Step 5: Motion gating
            motion_gate = torch.sigmoid(motion_feat)
            node_feature1 = node_feature1 * motion_gate
                    
        ############################## Second Layer ################################
        registration_dis2 = self.mlp1_7(data.x[:, 0:3])
        registration_dis2 = registration(data.x, data.edge_index, registration_dis2)
        registration_dis2 = torch.cat((registration_dis2, node_feature1[data.edge_index[1]]), dim=1)
        registration_dis2 = self.mlp1_8(registration_dis2)
        registration_dis2 = torch.relu(registration_dis2)
        
        node_feature1 = self.conv1_2(node_feature1, data.edge_index, edge_attr=registration_dis2)
        del registration_dis2
        node_feature1 = self.bn1_3(node_feature1)
        node_feature1 = self.mlp1_3(node_feature1)
        node_feature1 = self.bn1_4(torch.relu(node_feature1))
        node_feature1 = torch.sigmoid(self.lin1_1(node_feature1))  
        
        return node_feature1