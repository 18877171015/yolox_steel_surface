import os
from pathlib import Path

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = 'your-secret-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(BASE_DIR, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static/uploads')
    RESULT_FOLDER = os.path.join(BASE_DIR, 'static/results')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'pt', 'pth', 'onnx'}
    MAX_CONTENT_LENGTH = 512 * 1024 * 1024

    # YOLOX 配置 - 自动定位
    # 假设当前文件位于 steel_defect_system/，上一级目录为 YOLOX-main
    YOLOX_ROOT = str(Path(BASE_DIR).parent)
    YOLOX_WEIGHT = os.path.join(BASE_DIR, 'model', 'latest_ckpt.pth')
    # 可选：如果训练时的实验配置文件存在，可以指定；否则使用默认配置
    YOLOX_EXP_CONFIG = None