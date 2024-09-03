import os
import xml.dom.minidom
from tqdm import tqdm


def read_xml2img(AnnoPath=r'D:\DeepLearning\3-SAR\SAR数据集\原label\RotatedAnnotations'):
    imagelist = os.listdir(AnnoPath)

    w_values = []
    h_values = []

    # 使用 tqdm 显示进度条
    for image in tqdm(imagelist, desc="Processing images"):
        image_pre, ext = os.path.splitext(image)
        xmlfile = os.path.join(AnnoPath, image_pre + '.xml')
        DomTree = xml.dom.minidom.parse(xmlfile)
        annotation = DomTree.documentElement
        objectlist = annotation.getElementsByTagName('object')

        for objects in objectlist:
            rotated_bndbox = objects.getElementsByTagName('robndbox')  # robndbox rotated_bndbox
            for box in rotated_bndbox:
                w1 = box.getElementsByTagName('w')
                w = float(w1[0].childNodes[0].data)
                w_values.append(w)

                h1 = box.getElementsByTagName('h')
                h = float(h1[0].childNodes[0].data)
                h_values.append(h)

    # 计算w和h的范围
    def get_ranges(values):
        if not values:
            return {}
        min_val, max_val = min(values), max(values)
        range_width = (max_val - min_val) / 20
        ranges = {}
        for i in range(21):
            lower = min_val + i * range_width
            upper = min_val + (i + 1) * range_width
            range_str = f'{lower:.2f}-{upper:.2f}'
            ranges[range_str] = sum(lower <= value < upper for value in values)
        return ranges

    w_ranges = get_ranges(w_values)
    h_ranges = get_ranges(h_values)

    # 打印统计结果
    print("w范围统计:")
    for range_str, count in w_ranges.items():
        print(f"{range_str} 的数量是 {count}")

    print("\nh范围统计:")
    for range_str, count in h_ranges.items():
        print(f"{range_str} 的数量是 {count}")


# 调用函数
read_xml2img()
"""
不考虑斜框，直接统计xml文件中的w、h，有个大致的印象：
可见如果split的话，300左右比较合适，不太会漏框。
w范围统计:
0.00-40.56 的数量是 63460
40.56-81.13 的数量是 28156
81.13-121.69 的数量是 4390
121.69-162.26 的数量是 1079
162.26-202.82 的数量是 424
202.82-243.39 的数量是 225
243.39-283.95 的数量是 118
283.95-324.52 的数量是 64
324.52-365.08 的数量是 27
365.08-405.65 的数量是 30
405.65-446.21 的数量是 13
446.21-486.78 的数量是 6
486.78-527.34 的数量是 1
527.34-567.91 的数量是 2
567.91-608.47 的数量是 2
608.47-649.04 的数量是 0
649.04-689.60 的数量是 0
689.60-730.17 的数量是 1
730.17-770.73 的数量是 0
770.73-811.30 的数量是 0
811.30-851.86 的数量是 1

h范围统计:
0.00-21.11 的数量是 86504
21.11-42.22 的数量是 8527
42.22-63.32 的数量是 1781
63.32-84.43 的数量是 526
84.43-105.54 的数量是 246
105.54-126.65 的数量是 120
126.65-147.75 的数量是 72
147.75-168.86 的数量是 47
168.86-189.97 的数量是 42
189.97-211.08 的数量是 32
211.08-232.18 的数量是 16
232.18-253.29 的数量是 20
253.29-274.40 的数量是 24
274.40-295.51 的数量是 10
295.51-316.61 的数量是 15
316.61-337.72 的数量是 6
337.72-358.83 的数量是 5
358.83-379.94 的数量是 2
379.94-401.04 的数量是 0
401.04-422.15 的数量是 3
422.15-443.26 的数量是 1
"""