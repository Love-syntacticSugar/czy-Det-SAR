import os
from PIL import Image

"""
1.在所有图片中，最少几个目标，最多几个目标（温馨提示：可以通过统计每个txt文件的行数做到）
2.txt文件中的坐标有没有大于1的（注意：txt文件中每一行的第一个值表示目标的类别，剩下的8个值才是坐标）
3.图片的size有几种，每一种有多少张（这个要通过遍历image_root_path中的图像做到）
"""


def analyze_images_and_labels(image_root_path, save_root):
    min_targets = float('inf')  # 初始化最小目标数
    max_targets = 0  # 初始化最大目标数
    coord_greater_than_1 = False  # 检查是否有坐标大于1的情况
    size_count = {}  # 用于统计图片尺寸及其数量

    # 遍历所有图片
    image_files = [f for f in os.listdir(image_root_path) if f.endswith('.jpg')]

    for image_file in image_files:
        # 获取图片路径并获取其尺寸
        image_path = os.path.join(image_root_path, image_file)
        with Image.open(image_path) as img:
            size = img.size  # (width, height)
            # 更新尺寸统计
            if size in size_count:
                size_count[size] += 1
            else:
                size_count[size] = 1

        # 对应的txt文件路径
        txt_file_path = os.path.join(save_root, f"{os.path.splitext(image_file)[0]}.txt")

        # 检查txt文件是否存在
        if os.path.exists(txt_file_path):
            with open(txt_file_path, 'r') as txt_file:
                lines = txt_file.readlines()

                # 更新最少和最多目标数
                num_targets = len(lines)
                min_targets = min(min_targets, num_targets)
                max_targets = max(max_targets, num_targets)

                # 检查是否有坐标大于1的情况
                for line in lines:
                    data = list(map(float, line.strip().split()[1:]))  # 跳过类别，取8个坐标值
                    if any(coord > 1 for coord in data):
                        coord_greater_than_1 = True

    # 如果没有txt文件，min_targets初始化为-1
    if min_targets == float('inf'):
        min_targets = -1

    # 打印结果
    print(f"最少目标数: {min_targets}")
    print(f"最多目标数: {max_targets}")
    print(f"是否存在大于1的坐标值: {'是' if coord_greater_than_1 else '否'}")
    print("图片尺寸统计:")
    for size, count in size_count.items():
        print(f"尺寸 {size} 的图片有 {count} 张")


def analyze_test_image(image_root_path):
    size_count = {}  # 用于统计图片尺寸及其数量

    # 遍历所有图片
    image_files = [f for f in os.listdir(image_root_path) if f.endswith('.jpg')]

    for image_file in image_files:
        # 获取图片路径并获取其尺寸
        image_path = os.path.join(image_root_path, image_file)
        with Image.open(image_path) as img:
            size = img.size  # (width, height)
            # 更新尺寸统计
            if size in size_count:
                size_count[size] += 1
            else:
                size_count[size] = 1

    print("图片尺寸统计:")
    for size, count in size_count.items():
        print(f"尺寸 {size} 的图片有 {count} 张")


# 使用方法
# analyze_images_and_labels(r"D:\DeepLearning\3-SAR\SAR数据集\images\train",
#                           r"D:\DeepLearning\3-SAR\SAR数据集\labels\train", )
analyze_test_image(r"D:\DeepLearning\3-SAR\SAR数据集\images\test")
"""
analyze_images_and_labels函数的结果：
最少目标数: 0
最多目标数: 103
是否存在大于1的坐标值: 是【说明有坐标溢出了边界】
图片尺寸统计:
尺寸 (1000, 1000) 的图片有 210 张
尺寸 (800, 800) 的图片有 10222 张
尺寸 (256, 256) 的图片有 50376 张
尺寸 (2048, 2048) 的图片有 212 张
尺寸 (512, 512) 的图片有 4900 张
尺寸 (3000, 3000) 的图片有 20 张
尺寸 (4140, 4140) 的图片有 1 张
尺寸 (1024, 1024) 的图片有 466 张
尺寸 300~500左右，大概有600张
"""
"""
analyze_test_image的结果：
尺寸 (1000, 1000) 的图片有 90 张
尺寸 (800, 800) 的图片有 4382 张
尺寸 (256, 256) 的图片有 21599 张
尺寸 (2048, 2048) 的图片有 81 张
尺寸 (512, 512) 的图片有 2100 张
尺寸 (3000, 3000) 的图片有 10 张
尺寸 (1024, 1024) 的图片有 200 张
尺寸 300~500左右，大概有300张
"""