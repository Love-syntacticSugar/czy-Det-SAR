import os
import shutil
import re


def natural_sort(l):
    """
    自然排序函数，用于对包含数字的文件名进行排序

    Args:
        l: 文件名列表

    Returns:
        排序后的文件名列表
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def split_data_interval(data_root, split_ratio=0.25):
    """
    将训练集按间隔抽取部分文件到验证集

    Args:
        data_root: 数据根目录
        interval: 取样的间隔，每隔interval个文件取一个
    """
    interval = round(67219 / (67219 * split_ratio))

    image_train_dir = os.path.join(data_root, 'images', 'train')
    image_val_dir = os.path.join(data_root, 'images', 'val')
    label_train_dir = os.path.join(data_root, 'labels', 'train')
    label_val_dir = os.path.join(data_root, 'labels', 'val')

    # 获取训练集图像文件名列表，并按自然顺序排序
    image_files = natural_sort(os.listdir(image_train_dir))
    label_files = natural_sort(os.listdir(label_train_dir))

    for i, file in enumerate(image_files):
        if i % interval == 0:
            # 移动文件到val文件夹
            shutil.move(os.path.join(image_train_dir, file), os.path.join(image_val_dir, file))
            shutil.move(os.path.join(label_train_dir, label_files[i]), os.path.join(label_val_dir, label_files[i]))


# 示例用法
data_root = r'D:\DeepLearning\3-SAR\SAR数据集'  # 替换为你的数据根目录

split_data_interval(data_root, 0.25)
