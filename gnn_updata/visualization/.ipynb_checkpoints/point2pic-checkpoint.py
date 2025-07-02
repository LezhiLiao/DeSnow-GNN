import os
import numpy as np
import matplotlib.pyplot as plt
import torch

def visualize_point_cloud(points, output_path):
    """可视化点云并保存为PNG图像（纯净版，无任何标注）"""
    # 转换为numpy数组（如果已经是numpy数组则跳过）
    if isinstance(points, torch.Tensor):
        points = points.numpy()
    
    # 1. 过滤点云范围 (x:[-40,40], y:[-20,20])
    mask = (points[:, 0] >= -40) & (points[:, 0] <= 40) & \
           (points[:, 1] >= -20) & (points[:, 1] <= 20)
    filtered_points = points[mask]
    
    if len(filtered_points) == 0:
        print(f"警告: 点云在指定范围内无数据")
        return
    
    # 提取各坐标轴数据
    x_filtered = filtered_points[:, 0]
    y_filtered = filtered_points[:, 1]
    z_filtered = filtered_points[:, 2]

    # 2. 创建颜色映射（基于z值）
    z_min, z_max = np.min(z_filtered), np.max(z_filtered)
    colors = (z_filtered - z_min) / (z_max - z_min + 1e-10)  # 添加小量防止除零

    # 3. 纯净可视化（无任何标注）
    plt.figure(figsize=(10, 5))
    plt.scatter(x_filtered, y_filtered, c=colors, s=0.1, marker='.', cmap='viridis')
    plt.xlim(-40, 40)
    plt.ylim(-20, 20)
    plt.gca().set_aspect('equal', adjustable='box')
    
    # 去除坐标轴和边框
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_point_cloud_files(input_dir, output_dir):
    """处理目录中的所有.pt文件"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.pt'):
            # 构建完整文件路径
            input_path = os.path.join(input_dir, filename)
            
            # 加载点云数据
            try:
                point_cloud = torch.load(input_path)
                print(f"正在处理: {filename}，点数量: {len(point_cloud)}")
                
                # 生成输出文件名（相同名称但改为.png）
                output_filename = os.path.splitext(filename)[0] + '.png'
                output_path = os.path.join(output_dir, output_filename)
                
                # 可视化并保存
                visualize_point_cloud(point_cloud, output_path)
                print(f"已保存可视化结果到: {output_path}")
                
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")

if __name__ == "__main__":
    # 设置输入输出目录
    input_directory = "/root/autodl-tmp/gnn_updata/inference/point_saving_path/cadc2_pre"
    output_directory = "/root/autodl-tmp/gnn_updata/visualization/pic_saving_path/cadc2_pre"
    
    # 处理所有点云文件
    process_point_cloud_files(input_directory, output_directory)
