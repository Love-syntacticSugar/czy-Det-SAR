import os
from PIL import Image


# 使用s1obtain4Point.py生成的groundtruth_not_use_int.txt,将其转换为yolo格式
def label_to_yolo_label(text_file_path, image_root_path, save_root):
    # 如果保存目录不存在，则创建它
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    # 打开输入的txt文件
    with open(text_file_path, 'r') as file:
        lines = file.readlines()

    # 处理每一行数据
    for line in lines:
        # 将行内容分割成不同部分
        data = line.strip().split()
        image_name = data[0]  # xxx.jpg
        coordinates = list(map(float, data[1:]))  # 将坐标转换为浮点数

        # 获取图片编号并构建保存路径
        image_basename = os.path.splitext(image_name)[0]  # 去掉.jpg
        save_file_path = os.path.join(save_root, f"{image_basename}.txt")

        # 获取图片路径并打开图片
        image_path = os.path.join(image_root_path, image_name)
        with Image.open(image_path) as img:
            width, height = img.size

        # 归一化坐标（x/W, y/H）
        normalized_coords = []
        for i in range(0, len(coordinates), 2):
            x = coordinates[i] / width
            y = coordinates[i + 1] / height
            normalized_coords.extend([x, y])

        # 将归一化的坐标转换为字符串格式
        normalized_coords_str = " ".join(map(str, normalized_coords))

        # 将结果保存到对应的xxx.txt文件中
        with open(save_file_path, 'a') as output_file:
            output_file.write(f"0 {normalized_coords_str}\n")


import os


# 你希望为每一张图片生成一个对应的txt文件，即使该图片没有目标。要实现这个目标，我们需要遍历image_root_path中的所有图片，
# 检查每一张图片对应的txt文件是否存在于save_root中。如果不存在，就生成一个空的txt文件。
def ensure_all_txt_files(image_root_path, save_root):
    # 获取图片目录中的所有图片文件名
    image_files = [f for f in os.listdir(image_root_path) if f.endswith('.jpg')]

    # 遍历每一个图片文件
    for image_file in image_files:
        # 获取图片的基名称（不带扩展名）
        image_basename = os.path.splitext(image_file)[0]

        # 构建对应的txt文件路径
        txt_file_path = os.path.join(save_root, f"{image_basename}.txt")

        # 检查txt文件是否存在，如果不存在则创建一个空的txt文件
        if not os.path.exists(txt_file_path):
            with open(txt_file_path, 'w') as txt_file:
                pass  # 创建空文件

    print(f"已检查并补全缺失的txt文件。")


label_to_yolo_label(r"D:\DeepLearning\3-SAR\SAR数据集\groundtruth_not_use_int.txt",
                    r"D:\DeepLearning\3-SAR\SAR数据集\images\train",
                    r"D:\DeepLearning\3-SAR\SAR数据集\labels\train",
                    )
ensure_all_txt_files(r"D:\DeepLearning\3-SAR\SAR数据集\images\train",
                     r"D:\DeepLearning\3-SAR\SAR数据集\labels\train",
                     )
