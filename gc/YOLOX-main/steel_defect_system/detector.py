import sys
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from loguru import logger

class YOLOXDetector:
    def __init__(self, weight_path, yolox_root, exp_config=None, device='cuda', conf_thres=0.25, nms_thres=0.45):
        # 添加YOLOX路径
        yolox_root = os.path.abspath(yolox_root)
        if yolox_root not in sys.path:
            sys.path.insert(0, yolox_root)

        # 导入所需模块
        from yolox.exp import Exp as BaseExp
        from yolox.utils import postprocess
        from yolox.data.data_augment import ValTransform

        self.postprocess = postprocess
        self.ValTransform = ValTransform
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

        # 定义实验配置（与训练一致）
        class ExpNEUDET(BaseExp):
            def __init__(self):
                super().__init__()
                self.num_classes = 6
                self.depth = 0.33
                self.width = 0.50
                self.test_size = (640, 640)
                self.test_conf = conf_thres
                self.nmsthre = nms_thres
                self.fp16 = False
                self.legacy = False

        self.exp = ExpNEUDET()
        self.model = self.exp.get_model()
        self.model.to(self.device)
        ckpt = torch.load(weight_path, map_location=self.device)
        if 'model' in ckpt:
            self.model.load_state_dict(ckpt['model'])
        else:
            self.model.load_state_dict(ckpt)
        self.model.eval()
        print(f"[INFO] 检测器加载成功，设备: {self.device}")

        # 预处理器
        self.preproc = self.ValTransform(legacy=self.exp.legacy)

    def detect(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return None, []
        h, w = img.shape[:2]

        # 预处理
        ratio = min(self.exp.test_size[0] / h, self.exp.test_size[1] / w)
        img_resized, _ = self.preproc(img, None, self.exp.test_size)
        img_tensor = torch.from_numpy(img_resized).unsqueeze(0).float().to(self.device)

        # 推理
        with torch.no_grad():
            outputs = self.model(img_tensor)
            predictions = self.postprocess(outputs, self.exp.num_classes, self.conf_thres, self.nms_thres, class_agnostic=True)[0]

        if predictions is None:
            return img, []

        # 后处理：坐标映射、收集检测结果
        predictions = predictions.cpu()
        bboxes = predictions[:, 0:4]
        bboxes /= ratio
        scores = predictions[:, 4] * predictions[:, 5]
        cls_ids = predictions[:, 6].int()

        detections = []
        for i in range(len(bboxes)):
            if scores[i] < self.conf_thres:
                continue
            x1, y1, x2, y2 = bboxes[i].int().tolist()
            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(scores[i]),
                'class': self.class_names[cls_ids[i]],
                'class_id': int(cls_ids[i])
            })
        return img, detections

    def draw_boxes(self, img, detections):
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class']}: {det['confidence']:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

            font_scale = 0.8
            thickness = 3
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            label_y1 = max(y1 - text_height - baseline - 4, 0)

            cv2.rectangle(img, (x1, label_y1), (x1 + text_width + 8, label_y1 + text_height + baseline + 4),
                         (0, 0, 255), -1)
            cv2.putText(img, label, (x1 + 4, label_y1 + text_height + 2),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return img