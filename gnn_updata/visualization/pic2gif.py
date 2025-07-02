import os
from PIL import Image

# 设置图片目录路径
image_folder = '/root/autodl-tmp/gnn_updata/visualization/pic_saving_path/cadc2_pre'
output_gif = '/root/autodl-tmp/gnn_updata/visualization/gif_path/cadc_raw.gif'  # 输出的GIF路径

# 获取目录下所有PNG文件并按名称排序
png_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
png_files.sort()  # 按文件名排序

# 打开所有图片并存入列表
images = []
for png_file in png_files:
    file_path = os.path.join(image_folder, png_file)
    images.append(Image.open(file_path))

# 保存为GIF
# 参数说明:
# save_all=True 保存所有帧
# append_images=images[1:] 附加其余帧
# duration=200 每帧显示时间(毫秒)
# loop=0 无限循环
if len(images) > 0:
    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=200,  # 控制每帧显示时间(毫秒)
        loop=0  # 0表示无限循环
    )
    print(f'GIF已成功保存到: {output_gif}')
else:
    print('未找到PNG图像文件')