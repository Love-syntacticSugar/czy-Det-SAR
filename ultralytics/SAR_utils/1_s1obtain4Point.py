from __future__ import division
import os
import xml.dom.minidom
import cv2
import numpy as np
from PIL import Image


def read_xml2img(ImgPath=r'D:\DeepLearning\3-SAR\SAR数据集\images\train',
                 AnnoPath=r'D:\DeepLearning\3-SAR\SAR数据集\RotatedAnnotations',
                 Savepath=r'D:\DeepLearning\3-SAR\SAR数据集',
                 use_int=True,
                 use_floor_0=True,
                 ):
    """
    1.该代码是官方提供的.
    我稍作修改:
    (1)添加参数use_int:读取坐标的时候是否转为int类型【官方说int是因为show的时候需要int类型】
    (2)添加参数use_floor_0:对于溢出的坐标是否取0

    2.为什么做这样的修改？
    有以下几个原因：
    1.官方的测试集肯定也是从xml格式转为4坐标格式，那么我就需要我预测的点坐标就需要和xml格式的结果越接近越好。这就需要训练集坐标越精确越好了
    所以我训练的时候用的坐标需要use_int=False，use_floor_0=False
    2.官方有几个逆天的处理你要注意：
    （1）将xywh的xml结果全部int处理
    （2）将溢出的坐标直接取0
    （3）将最终的4坐标又int处理
    我猜测它对test结果也做了相同的处理
    所以我的建议是：你自己的预测结果交一份精确标（不int处理也不归0处理），再交一份和官方上述处理一样的结果，看看哪个好
    """
    imagelist = os.listdir(AnnoPath)
    output_txt = Savepath + '\\groundtruth.txt' if use_int else Savepath + '\\groundtruth_not_use_int.txt'

    with open(output_txt, 'w') as f:
        for image in imagelist:
            image_pre, ext = os.path.splitext(image)
            imgfile = ImgPath + '\\' + image_pre + '.jpg'
            xmlfile = AnnoPath + '\\' + image_pre + '.xml'
            # im = cv2.imread(imgfile)
            DomTree = xml.dom.minidom.parse(xmlfile)
            annotation = DomTree.documentElement
            filenamelist = annotation.getElementsByTagName('filename')
            filename = filenamelist[0].childNodes[0].data
            objectlist = annotation.getElementsByTagName('object')

            # print(filename)
            for objects in objectlist:
                namelist = objects.getElementsByTagName('name')
                objectname = namelist[0].childNodes[0].data
                rotated_bndbox = objects.getElementsByTagName('robndbox')  # robndbox rotated_bndbox
                # print(rotated_bndbox)
                for box in rotated_bndbox:
                    cx1 = box.getElementsByTagName('cx')
                    cx = int(float(cx1[0].childNodes[0].data)) if use_int else float(
                        cx1[0].childNodes[0].data)  # need int when showin in images
                    # print('cx = ', cx)

                    cy1 = box.getElementsByTagName('cy')
                    cy = int(float(cy1[0].childNodes[0].data)) if use_int else float(cy1[0].childNodes[0].data)
                    # print('cy = ', cy)

                    # cv2.rectangle(im,(xmin,ymin),(xmax,ymax), (0, 255, 0), 2)
                    w1 = box.getElementsByTagName('w')
                    w = int(float(w1[0].childNodes[0].data)) if use_int else float(w1[0].childNodes[0].data)
                    # print('w = ', w)

                    h1 = box.getElementsByTagName('h')
                    h = int(float(h1[0].childNodes[0].data)) if use_int else float(h1[0].childNodes[0].data)
                    # print('h = ', h)

                    theta1 = box.getElementsByTagName('angle')
                    theta = float(theta1[0].childNodes[0].data)
                    # print('theta = ', theta)
                    # TODO 这个加90度就没看懂，这是因为官方用的工具是rolabelimg，而这个工具标的角度需要研究一下
                    #  好在官方已经研究好了：就是加90度。
                    #  我尝试自己研究但是没研究明白，得花很多的时间，所以暂时放置
                    # 以下有一些参考文章：
                    # https://blog.csdn.net/qq_34575070/article/details/115897950?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ECtr-1-115897950-blog-120591481.235%5Ev43%5Epc_blog_bottom_relevance_base4&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ECtr-1-115897950-blog-120591481.235%5Ev43%5Epc_blog_bottom_relevance_base4&utm_relevant_index=1
                    # https://blog.csdn.net/FL1623863129/article/details/120591481
                    # https://blog.csdn.net/qq_36563273/article/details/109814534
                    ang = theta + np.pi / 2
                    c, s = np.sin(ang), np.cos(ang)

                    # get 4 points
                    corners_original = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]],
                                                dtype=float)
                    R = np.asarray([[c, s], [-s, c]])
                    # 旋转角点
                    # TODO 这里的解析和以前OW研究的v8很像，但是有一点不同，具体可见相应的word。但思想应该是一样的，所以暂时不研究
                    rotated_corners = corners_original @ R.T  # 注意这里使用.T来获取R的转置
                    # 平移到中心点
                    translated_corners = rotated_corners + np.array([cx, cy])
                    # 分配角点坐标到变量中
                    # TODO 对结果四舍五入
                    if use_int:
                        x1, y1 = round(translated_corners[0, 0]), round(translated_corners[0, 1])
                        x2, y2 = round(translated_corners[1, 0]), round(translated_corners[1, 1])
                        x3, y3 = round(translated_corners[2, 0]), round(translated_corners[2, 1])
                        x4, y4 = round(translated_corners[3, 0]), round(translated_corners[3, 1])
                    else:
                        x1, y1 = translated_corners[0, 0], translated_corners[0, 1]
                        x2, y2 = translated_corners[1, 0], translated_corners[1, 1]
                        x3, y3 = translated_corners[2, 0], translated_corners[2, 1]
                        x4, y4 = translated_corners[3, 0], translated_corners[3, 1]
                    if x1 < 0 or x2 < 0 or x3 < 0 or x4 < 0 or y1 < 0 or y2 < 0 or y3 < 0 or y4 < 0:
                        print(xmlfile)
                    # 这里官方的对于溢出的坐标处理是：取0
                    if use_int and use_floor_0:
                        f.write(
                            f'{image_pre}.jpg {max(int(x1), 0)} {max(int(y1), 0)} {max(int(x2), 0)} {max(int(y2), 0)} {max(int(x3), 0)} {max(int(y3), 0)} {max(int(x4), 0)} {max(int(y4), 0)} \n')
                    elif not use_int and not use_floor_0:
                        f.write(
                            f'{image_pre}.jpg {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4} \n')
            # path = Savepath + '/' + image_pre + '.jpg'
            # cv2.imwrite(path, im)


# 官方的结果（没啥用，我想用更精确的标取训练）
# read_xml2img()

# 坐标更精准
read_xml2img(use_int=False, use_floor_0=False)
