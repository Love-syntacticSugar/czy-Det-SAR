import shutil
import os


def merge_val_to_train(data_root):
    """
    将val文件夹中的文件合并回train文件夹

    Args:
      data_root: 数据根目录
    """

    # 获取图像和标签的train和val文件夹路径
    image_train_dir = os.path.join(data_root, 'images', 'train')
    image_val_dir = os.path.join(data_root, 'images', 'val')
    label_train_dir = os.path.join(data_root, 'labels', 'train')
    label_val_dir = os.path.join(data_root, 'labels', 'val')

    # 获取val文件夹中的所有文件
    val_image_files = os.listdir(image_val_dir)
    val_label_files = os.listdir(label_val_dir)

    # 将val文件夹中的文件移动到train文件夹
    for file in val_image_files:
        src = os.path.join(image_val_dir, file)
        dst = os.path.join(image_train_dir, file)
        shutil.move(src, dst)

    for file in val_label_files:
        src = os.path.join(label_val_dir, file)
        dst = os.path.join(label_train_dir, file)
        shutil.move(src, dst)


# 示例用法
data_root = r'D:\DeepLearning\3-SAR\SAR数据集'  # 替换为你的数据根目录
merge_val_to_train(data_root)
