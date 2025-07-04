#需手动将data和assigment复制到训练集目标文件夹
import os
import random
import shutil

def move_random_files(source_folder , num_files_to_move):
    # 获取源文件夹中的所有文件
    all_files = os.listdir(source_folder)
    num_files=len(all_files)
    random_integers =random.sample(range(num_files), num_files_to_move)
    #升序排列
    sorted_integers = sorted(random_integers)
    for i in range(num_files_to_move):
        source_path1 =f"/root/autodl-tmp/0.6k/data/data_batch{sorted_integers[i]}.pt"
        destination_path1=f"/root/autodl-tmp/0.6k/yanz/data"
        source_path2 = f"/root/autodl-tmp/0.6k/ass/assigment_batch{sorted_integers[i]}.pt"
        destination_path2 = f"/root/autodl-tmp/0.6k/yanz/ass/"
        shutil.copy2(source_path1, destination_path1)
        shutil.copy2(source_path2, destination_path2)
    #shutil.copy(f"D:\\CADC_dataset\\pre_handle\\data", f"D:\\CADC_dataset\\pre_handle\\xunlianji\\data")
    #shutil.copy(f"D:\\CADC_dataset\\pre_handle\\assigment",f"D:\\CADC_dataset\\pre_handle\\xunlianji\\assigment")
    for i in range(num_files_to_move):
        os.remove(f"/root/autodl-tmp/0.6k/data/data_batch{sorted_integers[i]}.pt")
        os.remove(f"/root/autodl-tmp/0.6k/ass/assigment_batch{sorted_integers[i]}.pt")

# 指定源文件夹和目标文件夹以及要移动的文件数量
source_folder = "/root/autodl-tmp/0.6k/data/"
num_files_to_move = 200

# 调用函数移动文件
move_random_files(source_folder,  num_files_to_move)