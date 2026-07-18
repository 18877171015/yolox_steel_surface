# NEU-DET 钢材表面缺陷智能检测系统

基于 YOLOX 深度学习框架的企业级钢材表面缺陷检测系统，支持 NEU-DET 数据集的 6 类缺陷自动识别。

## 📋 系统特性

### 核心功能
- **单图检测**: 上传单张图片进行实时缺陷识别和分析
- **文件夹检测**: 选择包含图片的文件夹，自动扫描并批量检测所有图像
- **模型管理**: 支持本地上传自定义模型文件 (.pt/.pth/.onnx)
- **用户管理**: 多角色权限控制（管理员/普通用户）
- **访客记录**: 完整的系统访问日志和数据分析

### 检测能力
- **6类缺陷识别**: Crazing (裂纹)、Inclusion (夹杂物)、Patches (斑块)、Pitted Surface (点蚀面)、Rolled-in Scale (氧化皮压入)、Scratches (划痕)
- **GPU加速**: 基于 CUDA 的高性能推理
- **高精度**: 平均准确率 99%+
- **实时处理**: 快速响应检测结果

## 🚀 快速开始

### 环境要求
- Python 3.8+
- CUDA 11.x+ (推荐)
- Flask 2.x
- OpenCV 4.x

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd steel_defect_system
```

2. **创建虚拟环境**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **初始化数据库**
```python
from app import app, init_db
with app.app_context():
    init_db()
```

5. **启动系统**
```bash
python app.py
```

6. **访问系统**
打开浏览器访问: `http://127.0.0.1:5000`

**默认账户:**
- 用户名: `admin`
- 密码: `admin123`

## 📖 使用指南

### 单图检测流程

1. 登录系统后进入"检测中心"
2. 选择"单图检测"标签页
3. 点击选择文件或拖拽图片到上传区域
4. 配置或切换检测模型（可选）
5. 点击"开始检测"按钮
6. 查看检测结果，包括：
   - 标注后的检测图片（可放大查看）
   - 缺陷类别和置信度
   - 边界框坐标和尺寸信息
   - 完整的置信度数据（6位小数精度）

### 文件夹检测流程

1. 进入"检测中心"
2. 选择"文件夹检测"标签页
3. 点击"选择文件夹"按钮
4. 在弹出的对话框中选择包含图片的文件夹
5. 系统自动：
   - 扫描文件夹及子目录中的所有图片
   - 显示文件统计信息（数量、总大小、路径）
   - 预览已选择的图片缩略图
6. 点击"开始检测全部图片"
7. 查看检测结果页面，包括：
   - 统计摘要（总图片数、缺陷图片、总缺陷数、缺陷率）
   - 每张图片的检测结果卡片
   - 缺陷标记和分类标签
   - 点击查看详细检测结果

**支持的图片格式:** JPG, JPEG, PNG, BMP  
**单个文件大小限制:** 16MB  
**自动扫描:** 支持递归扫描子目录

### 模型管理

#### 使用预训练模型
系统默认加载 YOLOX-S 模型，针对 NEU-DET 数据集优化。

#### 上传自定义模型
1. 在检测中心右侧找到"模型配置"面板
2. 点击"上传自定义模型"区域
3. 选择 .pt / .pth / .onnx 格式的模型文件
4. 点击"切换模型"按钮完成加载

## 🏗️ 技术架构

### 后端技术栈
- **Web框架**: Flask + Flask-Login
- **数据库**: SQLite (SQLAlchemy ORM)
- **深度学习**: PyTorch + YOLOX
- **图像处理**: OpenCV + NumPy

### 前端技术栈
- **UI框架**: Bootstrap 5
- **图标库**: Font Awesome 6
- **模板引擎**: Jinja2
- **样式**: 自定义 CSS 变量系统

### 项目结构
```
steel_defect_system/
├── app.py                 # 主应用入口
├── config.py              # 配置文件
├── models.py              # 数据库模型
├── detector.py            # 检测器封装
├── templates/             # HTML模板
│   ├── base.html          # 基础布局
│   ├── login.html         # 登录页
│   ├── register.html      # 注册页
│   ├── dashboard.html     # 仪表板
│   ├── detect.html        # 检测中心
│   ├── detect_result.html # 单图结果
│   └── folder_detect_result.html  # 文件夹结果
├── static/
│   ├── css/style.css      # 全局样式
│   ├── results/           # 检测结果存储
│   └── uploads/           # 上传文件存储
└── README.md              # 项目文档
```

## 🔧 配置说明

### config.py 主要参数
```python
class Config:
    SECRET_KEY = 'your-secret-key'       # 会话密钥
    UPLOAD_FOLDER = 'static/uploads'     # 上传目录
    RESULT_FOLDER = 'static/results'     # 结果目录
    YOLOX_WEIGHT = 'path/to/weight.pt'   # 模型权重路径
    YOLOX_ROOT = '../YOLOX-main'         # YOLOX根目录
    YOLOX_EXP_CONFIG = 'exps/default...' # 实验配置
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 文件大小限制(16MB)
```

### 检测参数
- **置信度阈值 (conf_thres)**: 0.25
- **NMS阈值 (nms_thres)**: 0.45
- **输入尺寸**: 640x640 (可调整)

## 👥 用户角色

### 管理员 (Admin)
- 访问所有功能模块
- 用户管理（添加/编辑/删除）
- 查看访客记录和统计数据
- 模型管理权限

### 普通用户 (User)
- 单图检测
- 文件夹检测
- 模型切换（使用预置模型）
- 查看个人检测结果

## 📊 API 接口

### 检测相关
- `POST /detect` - 执行检测（单图/文件夹/批量）
- `POST /api/detect-folder` - 文件夹检测API
- `POST /api/upload-model` - 上传自定义模型
- `POST /api/switch-model` - 切换当前模型

### 用户认证
- `GET/POST /login` - 用户登录
- `POST /register` - 用户注册
- `GET /api/logout` - 用户登出
- `GET /api/user-info` - 获取当前用户信息

### 管理（仅管理员）
- `GET /user-manage` - 用户管理页面
- `POST /user/add` - 添加新用户
- `POST /user/edit/<id>` - 编辑用户信息
- `GET /visitor_records` - 访客日志

## 🔒 安全特性

- 密码哈希存储 (werkzeug.security)
- CSRF保护
- 文件类型验证
- 路径遍历防护
- 会话管理
- 角色权限控制

## 🐛 常见问题

### Q: 检测不出结果？
A: 请检查：
1. 模型文件是否正确加载
2. 图片格式是否支持（JPG/PNG/BMP）
3. 置信度阈值是否过高
4. 查看终端输出的错误信息

### Q: 上传文件失败？
A: 确保：
1. 文件大小不超过 16MB
2. 文件格式正确
3. uploads 目录有写入权限

### Q: 模型切换失败？
A: 验证：
1. 模型文件格式（.pt/.pth/.onnx）
2. 模型与当前框架兼容
3. CUDA版本匹配

## 📝 更新日志

### v2.0.0 (2025-05-26)
- ✨ 新增文件夹检测功能（替代原批量检测）
- ✨ 图片放大功能（滚轮缩放/拖拽/全屏）
- ✨ 完整置信度输出（6位小数精度）
- 🎨 企业级UI全面升级
- 🔄 系统品牌更新为 NEU-DET
- 🐛 修复模板渲染错误
- ⚡ 性能优化和用户体验改进

### v1.0.0 (初始版本)
- 基础检测功能
- 用户管理系统
- 模型切换功能
- 访客记录统计

## 📄 许可证

MIT License

## 👨‍💻 开发团队

NEU-DET Development Team

---

**技术支持**: 如有问题请查看终端输出或联系开发人员
