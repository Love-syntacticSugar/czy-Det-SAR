# 1.将image split到xxx shape，做这件事情有两个契机：
#   （1）DOTA的启发,效果好自然就想着这次也用一用
#   （2）OW试过imgsz=256的和imgsz=640的，结果是640暴打256的。关于这件事非常值得去理解
#       为什么一开始OW要去试256的？因为OW发现7w张图像中，有5w是256的，本着不想失真的原则，才试了imgsz=256。但效果极差
#       是什么导致：将图像放大到640，明明失真了，结果却好到爆？
#       首先，256垃圾最重要的一个原因是：特征图太小，预测的框大大减少
#       其次，再谈图像失真的问题：imgsz=256会导致所有图像缩小到256，但是imgsz=640会导致图像放大到640
#            请你不要忘了，放大和缩小都会失真！！！都不是好东西！！！
#            那么问题来了：你觉得对于model来说，喜欢看放大后的图像还是缩小后的？你说你怎么知道？你当然知道！给你一张图片，让你找物体，你觉得是放大
#            后的图像容易找还是缩小后的图像容易找？答案显而易见：放大的。
#            从数学角度来看：放大后图像的失真是属于补充物体细节信息的失真（它是通过插值推断的物体信息，属于瞎编的信息，所以导致了失真。但是起码这个
#            物体原先的信息保留着没删除，是做加法而不是减法）但缩小的图像失真是真的把图像的信息给丢了，是做减法而不是加法
#            而model又不是GAN，不能凭空捏造出这些被你删掉的信息，所以自然喜欢看放大的图像
#            此外，本来这比赛小目标就多，非常多几十像素的目标存在，你这一缩小，不纯完蛋
#       所以，结论是:宁可放大也不要缩小
#       由此便可以顺理成章推出：对于SAR数据集中的超大图像，split就是不缩小的最佳办法
# 2.但无论如何，都是要split的，这件事情只有益处没有坏处，所以这里就是做这件事情
# 3.有件事你要知道：
# 你的SAR数据集，对于256*256的图像足足有5w张（比如1.jpg），如果设置了crop_size为640，那裁剪结果只有一张，且name为：
# 1__640__0___0.jpg，看名字似乎裁剪结果是640size的，实际上还是256，这个裁剪的图像和原图一模一样！只不过名字改了一下而已！
# 所以本次比赛的SAR数据集，裁剪的效果不会很明显，因为大部分的图像都是256*256的，根本就没裁剪
import sys

sys.path.insert(0, "/home/csx/disk/clg/code/Detection-SAR")
from ultralytics.data.split_dota import split_trainval, split_test

# 关于gap的选择：根据6_w_and_h_distribution.py，观察目标H和W的分布，选择300比较合适
crop_size = 1024
gap = 300
split_trainval(data_root="/home/csx/disk/clg/data/SAR",
               save_dir=f"/home/csx/disk/clg/data/SAR_split_with_{crop_size}",
               crop_size=crop_size,
               gap=gap)

split_test(data_root="/home/csx/disk/clg/data/SAR",
           save_dir=f"/home/csx/disk/clg/data/SAR_split_with_{crop_size}",
           crop_size=crop_size,
           gap=gap)
# TODO   label坐标保留了6位小数，原先是16位，因为我想了一下，不需要这么精准的，毕竟标的时候就存在误差，你存这么精准的干什么
#  （如果想改，请去crop_and_save函数中改）
