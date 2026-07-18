#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEU-DET 钢材缺陷检测 YOLOX 训练脚本（YOLOX-S + COCO格式）
6 类缺陷：crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches
"""

import os
import sys
from pathlib import Path
import torch
import multiprocessing

# ------------------------------------------------------------
# 多进程兼容性（Windows 需要 spawn 方式）
# ------------------------------------------------------------
if sys.platform == "win32":
    multiprocessing.set_start_method("spawn", force=True)

# ------------------------------------------------------------
# 基础路径配置（自动定位 YOLOX 根目录）
# ------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent                # 当前脚本所在目录（YOLOX-main）
PROJECT_ROOT = SCRIPT_DIR.parent                  # 上一级目录（例如 /root/rivermind-data/gc）
YOLOX_ROOT = str(SCRIPT_DIR)                      # YOLOX 根目录
print(f"[INFO] YOLOX根目录: {YOLOX_ROOT}")
sys.path.insert(0, YOLOX_ROOT)                    # 确保能导入 yolox 模块

# 导入 YOLOX 核心组件
from yolox.exp import Exp
from yolox.core import Trainer, launch
from yolox.utils import configure_module

# ------------------------------------------------------------
# 数据集路径配置（NEU-DET 目录结构）
# ------------------------------------------------------------
DATASET_ROOT = PROJECT_ROOT / "NEU-DET"           # 数据集根目录
if not DATASET_ROOT.exists():
    raise FileNotFoundError(f"找不到 NEU-DET 数据集: {DATASET_ROOT}")

TRAIN_JSON = DATASET_ROOT / "annotations" / "train.json"   # 训练集 COCO 标注
VAL_JSON   = DATASET_ROOT / "annotations" / "val.json"     # 验证集 COCO 标注

assert TRAIN_JSON.exists(), f"训练标注不存在: {TRAIN_JSON}"
assert VAL_JSON.exists(), f"验证标注不存在: {VAL_JSON}"

# ------------------------------------------------------------
# 实验配置类（继承 YOLOX 的 Exp）
# ------------------------------------------------------------
class ExpNEUDET(Exp):
    def __init__(self):
        super().__init__()

        # ========== 数据集配置 ==========
        self.num_classes = 6                      # NEU-DET 有 6 类缺陷
        self.data_dir = str(DATASET_ROOT) + os.sep   # 末尾加分隔符，COCODataset 要求
        self.train_ann = str(TRAIN_JSON)          # 训练标注文件路径
        self.val_ann = str(VAL_JSON)              # 验证标注文件路径

        # ========== 模型结构：YOLOX-S ==========
        self.depth = 0.33      # 网络深度因子（YOLOX-S 默认值）
        self.width = 0.50      # 网络宽度因子
        self.input_size = (640, 640)   # 训练时输入图像尺寸（宽,高）
        self.test_size = (640, 640)    # 验证/测试时输入尺寸

        # ========== 训练超参数 ==========
        self.batch_size = 32               # 批次大小（NEU-DET 图片小，可设较大）
        self.data_num_workers = 4          # 数据加载子进程数
        self.max_epoch = 150               # 总训练轮数
        self.warmup_epochs = 5             # 学习率预热轮数（稳定初期训练）
        self.eval_interval = 5             # 每5个epoch评估一次验证集
        self.save_history_ckpt = True      # 保存训练过程中的中间权重

        # ========== 学习率调度（关键调整）==========
        # basic_lr_per_img 是每张图片的基础学习率，实际 lr = basic_lr_per_img * batch_size
        # 原默认值 0.00015625 → 实际 lr=0.005 (batch=32)
        # 调整为 0.000078125 → 实际 lr=0.0025，更适合 YOLOX-S + 640 输入，避免震荡
        self.basic_lr_per_img = 0.000078125
        self.min_lr_ratio = 0.02             # 余弦退火最终学习率为初始的 2% (0.0025*0.02=5e-5)
        self.warmup_lr = 0.0                 # 预热起始学习率
        self.scheduler = "yoloxwarmcos"      # 使用带预热的余弦退火调度器

        # ========== 在线数据增强（Mosaic、MixUp、旋转、剪切等）==========
        self.degrees = 10.0          # 随机旋转角度范围 ±10°
        self.shear = 5.0             # 剪切变换强度
        self.translate = 0.1         # 平移范围（相对图像尺寸）
        self.mosaic_scale = (0.1, 2.0)      # Mosaic 中图像的随机缩放范围
        self.mixup_scale = (0.5, 1.5)       # MixUp 中图像的随机缩放范围
        self.mosaic_prob = 1.0              # 使用 Mosaic 的概率（1.0 表示每个 batch 都使用）
        self.mixup_prob = 1.0               # 使用 MixUp 的概率
        self.enable_mixup = True            # 是否启用 MixUp
        self.flip_prob = 0.5                # 随机水平翻转概率
        self.hsv_prob = 1.0                 # HSV 色彩空间增强概率

        # ========== 预训练权重（可选）==========
        pretrained_path = Path(YOLOX_ROOT) / "pretrained" / "yolox_s.pth"
        if pretrained_path.exists():
            self.pretrained_weights = str(pretrained_path)
            print(f"[INFO] 使用预训练权重: {pretrained_path}")
        else:
            self.pretrained_weights = None
            print("[WARN] 未找到预训练权重，将从头训练")

    # ------------------------------------------------------------
    # 训练数据集构造（被 get_data_loader 调用）
    # ------------------------------------------------------------
    def get_dataset(self, cache=False, cache_type="ram"):
        from yolox.data import COCODataset, TrainTransform
        return COCODataset(
            data_dir=self.data_dir,               # 数据根目录（末尾有分隔符）
            json_file=self.train_ann,             # COCO 格式标注文件
            name="",                              # 关键：不添加子目录，直接使用 file_name 中的相对路径
            img_size=self.input_size,             # 输入尺寸
            preproc=TrainTransform(               # 基础预处理（翻转、HSV）
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type,
        )

    # ------------------------------------------------------------
    # 验证数据集构造（评估时使用）
    # ------------------------------------------------------------
    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        testdev = kwargs.get("testdev", False)
        legacy = kwargs.get("legacy", True)
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            name="",                              # 同样不添加子目录
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

# ------------------------------------------------------------
# 主训练函数（由 launch 调用）
# ------------------------------------------------------------
def main(exp, args):
    trainer = Trainer(exp, args)
    trainer.train()

# ------------------------------------------------------------
# 脚本入口
# ------------------------------------------------------------
if __name__ == "__main__":
    configure_module()                     # 修复多进程启动问题
    exp = ExpNEUDET()                      # 实例化实验配置

    # 检测可用的 GPU 数量
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    num_gpu = len(cuda_visible.split(",")) if torch.cuda.is_available() else 0
    if num_gpu == 0:
        print("[WARN] 未检测到 GPU，将使用 CPU 训练（速度极慢）")

    # 模拟命令行参数（因为直接运行脚本，无需在终端输入参数）
    args = type("Args", (), {
        "exp_file": None,
        "resume": False,
        "ckpt": None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "fp16": True,                      # 开启混合精度训练（节省显存、加速）
        "fuse": False,
        "trt": False,
        "dist_backend": "nccl",
        "dist_url": "env://",
        "experiment_name": "neudet_test",   # 实验名称（日志和模型保存目录）
        "batch_size": exp.batch_size,
        "num_workers": exp.data_num_workers,
        "cache": None,                     # 避免 cache_img 断言错误
        "debug": False,
        "conf": None,
        "nms": None,
        "tsize": None,
        "seed": None,
        "occupy": False,
        "opts": [],
        "local_rank": 0,
        "output_dir": exp.output_dir,
        "test": False,
        "logger": "tensorboard",
    })()

    # 启动分布式/单机训练
    launch(
        main,
        num_gpus_per_machine=num_gpu,      # 每台机器的 GPU 数量（关键参数名）
        num_machines=1,                    # 机器数量（单机）
        machine_rank=0,
        dist_url=args.dist_url,
        args=(exp, args),
    )