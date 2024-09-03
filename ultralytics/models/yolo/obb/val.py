# Ultralytics YOLO 🚀, AGPL-3.0 license
import os
from pathlib import Path

import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import OBBMetrics, batch_probiou
from ultralytics.utils.plotting import output_to_rotated_target, plot_images


class OBBValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on an Oriented Bounding Box (OBB) model.

    Example:
        ```python
        from ultralytics.models.yolo.obb import OBBValidator

        args = dict(model='yolov8n-obb.pt', data='dota8.yaml')
        validator = OBBValidator(args=args)
        validator(model=args['model'])
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize OBBValidator and set task to 'obb', metrics to OBBMetrics."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = "obb"
        self.metrics = OBBMetrics(save_dir=self.save_dir, plot=True, on_plot=self.on_plot)

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        super().init_metrics(model)
        val = self.data.get(self.args.split, "")  # validation path
        self.is_dota = isinstance(val, str) and "DOTA" in val  # is COCO

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            nc=self.nc,
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
            rotated=True,
        )

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Perform computation of the correct prediction matrix for a batch of detections and ground truth bounding boxes.

        Args:
            detections (torch.Tensor): A tensor of shape (N, 7) representing the detected bounding boxes and associated
                data. Each detection is represented as (x1, y1, x2, y2, conf, class, angle).
            gt_bboxes (torch.Tensor): A tensor of shape (M, 5) representing the ground truth bounding boxes. Each box is
                represented as (x1, y1, x2, y2, angle).
            gt_cls (torch.Tensor): A tensor of shape (M,) representing class labels for the ground truth bounding boxes.

        Returns:
            (torch.Tensor): The correct prediction matrix with shape (N, 10), which includes 10 IoU (Intersection over
                Union) levels for each detection, indicating the accuracy of predictions compared to the ground truth.

        Example:
            ```python
            detections = torch.rand(100, 7)  # 100 sample detections
            gt_bboxes = torch.rand(50, 5)  # 50 sample ground truth boxes
            gt_cls = torch.randint(0, 5, (50,))  # 50 ground truth class labels
            correct_matrix = OBBValidator._process_batch(detections, gt_bboxes, gt_cls)
            ```

        Note:
            This method relies on `batch_probiou` to calculate IoU between detections and ground truth bounding boxes.
        """
        iou = batch_probiou(gt_bboxes, torch.cat([detections[:, :4], detections[:, -1:]], dim=-1))
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def _prepare_batch(self, si, batch):
        """Prepares and returns a batch for OBB validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox[..., :4].mul_(torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]])  # target boxes
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad, xywh=True)  # native-space labels
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        """Prepares and returns a batch for OBB validation with scaled and padded bounding boxes."""
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"], xywh=True
        )  # native-space pred
        return predn

    def plot_predictions(self, batch, preds, ni, threaded=True):
        """Plots predicted bounding boxes on input images and saves the result."""
        plot_images(
            batch["img"],
            *output_to_rotated_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / "image" / "predict" / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
            threaded=threaded,
        )  # pred

    def pred_to_json(self, predn, filename):
        """Serialize YOLO predictions to COCO json format."""
        # TODO 待修改
        # 改动3：按照官方的方法对结果进行如下操作：
        #       （1）先讲xywh int(float(cx1[0]))
        #       (2)转为poly后对结果进行round，四舍五入
        #       （3）再对poly结果进行 max(int(x1), 0)
        stem = Path(filename).stem + Path(filename).suffix  # SAR需要文件名称加后缀
        image_id = int(stem) if stem.isnumeric() else stem
        rbox = torch.cat([predn[:, :4], predn[:, -1:]], dim=-1)
        poly = ops.xywhr2xyxyxyxy(rbox).view(-1, 8)
        for i, (r, b) in enumerate(zip(rbox.tolist(), poly.tolist())):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(predn[i, 5].item())],
                    "score": round(predn[i, 4].item(), 4),  # TODO 也可以改一改
                    # "rbox": [max(int(round(x)), 0) for x in r],
                    # "rbox": [round(x, 5) for x in r],x
                    # "rbox": [int(x) for x in r],
                    "rbox": [x for x in r],
                    # "poly": [max(int(round(x)), 0) for x in b],  # 76.9
                    # "poly": [round(x, 5) for x in b], # 77.2
                    # "poly": [int(x) for x in b],  # 77.9
                    "poly": [x for x in b],
                }
            )

    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        import numpy as np

        from ultralytics.engine.results import Results

        rboxes = torch.cat([predn[:, :4], predn[:, -1:]], dim=-1)
        # xywh, r, conf, cls
        obb = torch.cat([rboxes, predn[:, 4:6]], dim=-1)
        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            obb=obb,
        ).save_txt(file, save_conf=save_conf)

    def eval_json(self):
        """Evaluates YOLO output in JSON format and returns performance statistics."""
        if self.args.save_json and self.is_dota and len(self.jdict):
            import json
            import re
            from collections import defaultdict

            pred_json = self.save_dir / "predictions.json"  # predictions
            pred_txt = self.save_dir / "predictions_txt"  # predictions
            pred_txt.mkdir(parents=True, exist_ok=True)
            data = json.load(open(pred_json))
            # Save split results
            LOGGER.info(f"Saving predictions with DOTA format to {pred_txt}...")
            for d in data:
                image_id = d["image_id"]
                score = d["score"]
                classname = self.names[d["category_id"]].replace(" ", "-")
                p = d["poly"]

                with open(f'{pred_txt / f"Task1_{classname}"}.txt', "a") as f:
                    f.writelines(f"{image_id} {score} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n")
            # Save merged results, this could result slightly lower map than using official merging script,
            # because of the probiou calculation.
            pred_merged_txt = self.save_dir / "predictions_merged_txt"  # predictions
            pred_merged_txt.mkdir(parents=True, exist_ok=True)
            merged_results = defaultdict(list)
            LOGGER.info(f"Saving merged predictions with DOTA format to {pred_merged_txt}...")
            for d in data:
                image_id = d["image_id"].split("__")[0]
                pattern = re.compile(r"\d+___\d+")
                x, y = (int(c) for c in re.findall(pattern, d["image_id"])[0].split("___"))
                bbox, score, cls = d["rbox"], d["score"], d["category_id"]
                bbox[0] += x
                bbox[1] += y
                bbox.extend([score, cls])
                merged_results[image_id].append(bbox)
            for image_id, bbox in merged_results.items():
                bbox = torch.tensor(bbox)
                max_wh = torch.max(bbox[:, :2]).item() * 2
                c = bbox[:, 6:7] * max_wh  # classes
                scores = bbox[:, 5]  # scores
                b = bbox[:, :5].clone()
                b[:, :2] += c
                # 0.3 could get results close to the ones from official merging script, even slightly better.
                i = ops.nms_rotated(b, scores, 0.3)
                bbox = bbox[i]

                b = ops.xywhr2xyxyxyxy(bbox[:, :5]).view(-1, 8)
                for x in torch.cat([b, bbox[:, 5:7]], dim=-1).tolist():
                    classname = self.names[int(x[-1])].replace(" ", "-")
                    p = [round(i, 3) for i in x[:-2]]  # poly
                    score = round(x[-2], 3)

                    with open(f'{pred_merged_txt / f"Task1_{classname}"}.txt', "a") as f:
                        f.writelines(f"{image_id} {score} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n")
        if self.args.save_json and self.args.is_SAR and len(self.jdict):
            import json
            import re
            from collections import defaultdict

            pred_json = self.save_dir / "predictions.json"  # predictions

            # pred_txt_poly_int = self.save_dir / "int"  # 直接取整
            # pred_txt_poly_round = self.save_dir / "round"  # 保留5位小数
            # pred_txt_poly_round_int_max = self.save_dir / "round_int_max"  # 四舍五入后max
            # pred_txt_poly_round_0 = self.save_dir / "round"  # 四舍五入
            pred_txt_poly_int_max = self.save_dir / "int_max"  # 取整+max

            # pred_txt_poly_int.mkdir(parents=True, exist_ok=True)
            # pred_txt_poly_round.mkdir(parents=True, exist_ok=True)
            # pred_txt_poly_round_int_max.mkdir(parents=True, exist_ok=True)
            # pred_txt_poly_round_0.mkdir(parents=True, exist_ok=True)
            pred_txt_poly_int_max.mkdir(parents=True, exist_ok=True)

            data = json.load(open(pred_json))
            LOGGER.info(f"Saving predictions with DOTA format to {pred_txt_poly_int_max}...")
            for d in data:
                image_id = d["image_id"]
                score = d["score"]
                p = d["poly"]
                # with open(f'{pred_txt_poly_int / "submitted"}.txt', "a") as f:
                #     f.writelines(
                #         f"{image_id} {int(p[0])} {int(p[1])} {int(p[2])} {int(p[3])} {int(p[4])} {int(p[5])} {int(p[6])} {int(p[7])} {score}\n")

                # with open(f'{pred_txt_poly_round / "submitted"}.txt', "a") as f:
                #     f.writelines(
                #         f"{image_id} {round(p[0], 5)} {round(p[1], 5)} {round(p[2], 5)} {round(p[3], 5)} {round(p[4], 5)} {round(p[5], 5)} {round(p[6], 5)} {round(p[7], 5)} {score}\n")

                # with open(f'{pred_txt_poly_round_int_max / "submitted"}.txt', "a") as f:
                #     f.writelines(
                #         f"{image_id} {max(int(round(p[0])), 0)} {max(int(round(p[1])), 0)} {max(int(round(p[2])), 0)} {max(int(round(p[3])), 0)} {max(int(round(p[4])), 0)} {max(int(round(p[5])), 0)} {max(int(round(p[6])), 0)} {max(int(round(p[7])), 0)} {score}\n")

                # with open(f'{pred_txt_poly_round_0 / "submitted"}.txt', "a") as f:
                #     f.writelines(
                #         f"{image_id} {round(p[0])} {round(p[1])} {round(p[2])} {round(p[3])} {round(p[4])} {round(p[5])} {round(p[6])} {round(p[7])} {score}\n")
                with open(f'{pred_txt_poly_int_max / "submitted"}.txt', "a") as f:
                    f.writelines(
                        f"{image_id} {max(int(p[0]), 0)} {max(int(p[1]), 0)} {max(int(p[2]), 0)} {max(int(p[3]), 0)} {max(int(p[4]), 0)} {max(int(p[5]), 0)} {max(int(p[6]), 0)} {max(int(p[7]), 0)} {score}\n")
            os.remove(str(pred_json))
        if self.args.save_json and self.args.is_SAR_split and self.args.is_SAR and len(self.jdict):
            import json
            import re
            from collections import defaultdict

            pred_json = self.save_dir / "predictions.json"  # predictions
            pred_txt_poly_int_max = self.save_dir / "int_max"  # 取整+max
            pred_txt_poly_int_max.mkdir(parents=True, exist_ok=True)
            data = json.load(open(pred_json))

            merged_results = defaultdict(list)
            LOGGER.info(f"Saving merged predictions with DOTA format to {pred_txt_poly_int_max}...")
            for d in data:
                # 原先官方的d["image_id"]是没有后缀的，但是我在pred_to_json方法中加了.jpg的后缀。但无所谓，正则表达式匹配不到.jpg
                image_id = d["image_id"].split("__")[0] + ".jpg"
                pattern = re.compile(r"\d+___\d+")
                x, y = (int(c) for c in re.findall(pattern, d["image_id"])[0].split("___"))
                bbox, score, cls = d["rbox"], d["score"], d["category_id"]
                bbox[0] += x
                bbox[1] += y
                bbox.extend([score, cls])
                merged_results[image_id].append(bbox)
            for image_id, bbox in merged_results.items():
                bbox = torch.tensor(bbox)
                max_wh = torch.max(bbox[:, :2]).item() * 2
                c = bbox[:, 6:7] * max_wh  # classes
                scores = bbox[:, 5]  # scores
                b = bbox[:, :5].clone()
                b[:, :2] += c
                # 0.3 could get results close to the ones from official merging script, even slightly better.
                # TODO 可修改：在non_max_suppression方法那里是0.5，这里表示iou大于0.3就丢弃，更苛刻了。
                #  1.而你要知道：你有整整5w张ship由于imgsizs=256<crop_size，压根没做过裁剪，它们其实可以不用做这里的NMS的，
                #  就像以前那样直接拿来用但这里又做了一次NMS，而且比前向传播那里的non_max_suppression方法更严格用的是0.3.
                #  导致可能又有一些ship被丢掉了。你懂我的意思嘛？我的意思是可能会比之前的效果差，因为筛选更严格了。如果不想这么严格怎么办
                #  答：用0.5的阈值！
                #  2.但其实话又说回来了。这次比赛不比DOTA，我认为单张图像预测的结果中，重复的概率不高（又不是那种足球场和操场框重合），所以对于
                #  imgsizs=256的预测结果，0.3的阈值和0.5的阈值我认为是差不多的。所以我不认为要因为这一点将0.3改为0.5.
                #  3.此外，这里的NMS主要就是针对的gap=200那里重复的预测结果，这点你心里要有数
                #  4.后续也值得一试好吧，搞一个0.5版本的，反正10几分钟就预测出来了
                i = ops.nms_rotated(b, scores, 0.3)
                bbox = bbox[i]

                b = ops.xywhr2xyxyxyxy(bbox[:, :5]).view(-1, 8)
                for x in torch.cat([b, bbox[:, 5:7]], dim=-1).tolist():
                    p = [i for i in x[:-2]]  # poly
                    score = round(x[-2], 4)  # TODO 也可以改一改

                    with open(f'{pred_txt_poly_int_max / "submitted"}.txt', "a") as f:
                        f.writelines(
                            f"{image_id} {max(int(p[0]), 0)} {max(int(p[1]), 0)} {max(int(p[2]), 0)} {max(int(p[3]), 0)} {max(int(p[4]), 0)} {max(int(p[5]), 0)} {max(int(p[6]), 0)} {max(int(p[7]), 0)} {score}\n")
            os.remove(str(pred_json))
