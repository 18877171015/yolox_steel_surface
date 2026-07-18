import os
import zipfile
import cv2
import urllib.parse
import time
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from config import Config
from models import db, User, VisitorRecord
from detector import YOLOXDetector
from report_generator import ReportGenerator
from functools import wraps
import datetime

# 创建 Flask 应用实例
app = Flask(__name__)
# 从 config.py 加载配置
app.config.from_object(Config)
# 初始化数据库对象，绑定到应用
db.init_app(app)
# 初始化 Flask-Login 管理器
login_manager = LoginManager()
login_manager.init_app(app)
# 设置登录视图（未登录时跳转到的端点）
login_manager.login_view = 'login'

# 全局检测器变量，用于懒加载
detector = None

def get_detector():
    """获取或创建 YOLOX 检测器单例"""
    global detector
    if detector is None:
        detector = YOLOXDetector(
            weight_path=Config.YOLOX_WEIGHT,
            yolox_root=Config.YOLOX_ROOT,
            exp_config=Config.YOLOX_EXP_CONFIG,
            conf_thres=0.25,
            nms_thres=0.45
        )
    return detector

@login_manager.user_loader
def load_user(user_id):
    """根据用户 ID 加载用户对象（Flask-Login 必需）"""
    return User.query.get(int(user_id))

def init_db():
    """初始化数据库：创建所有表，并创建默认管理员账户（如果不存在）"""
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username='admin').first():
            admin = User(username='admin', password=generate_password_hash('admin123'), role='admin')
            db.session.add(admin)
            db.session.commit()

def record_visitor(f):
    """装饰器：记录访客信息到 VisitorRecord 表"""
    @wraps(f)
    def decorated(*args, **kwargs):
        ip = request.remote_addr                     # 客户端 IP
        endpoint = request.endpoint                  # 访问的端点名称
        method = request.method                      # HTTP 方法
        ua = request.headers.get('User-Agent', '')   # 用户代理字符串
        user_id = current_user.id if current_user.is_authenticated else None  # 已登录用户的 ID
        record = VisitorRecord(ip=ip, endpoint=endpoint, method=method, user_agent=ua, user_id=user_id)
        db.session.add(record)
        db.session.commit()
        return f(*args, **kwargs)
    return decorated

# 创建上传文件存储目录和检测结果目录（如果不存在）
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    """根路径重定向到登录页面"""
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """用户登录路由"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)                         # 登录用户
            return redirect(url_for('dashboard'))
        flash('用户名或密码错误', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """用户注册路由"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('用户名已存在', 'warning')
        else:
            hashed = generate_password_hash(password)
            new_user = User(username=username, password=hashed, role='user')
            db.session.add(new_user)
            db.session.commit()
            flash('注册成功，请登录', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    """用户登出路由"""
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
@record_visitor
def dashboard():
    """仪表板主页"""
    return render_template('dashboard.html', user=current_user)

@app.route('/detect', methods=['GET', 'POST'])
@login_required
@record_visitor
def detect():
    """检测中心：支持单张图片上传、文件夹上传、批量图片上传"""
    if request.method == 'POST':
        detector = get_detector()
        # 处理单张图片上传（name="file"）
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            if file.filename == '':
                flash('未选择文件', 'warning')
                return redirect(request.url)
            original_filename = file.filename
            safe_name = secure_filename(original_filename)        # 安全化文件名
            if not safe_name:
                safe_name = f"image_{int(time.time())}.jpg"
            name, ext = os.path.splitext(safe_name)
            unique_name = f"{name}_{int(time.time()*1000)}{ext}"   # 添加时间戳避免重名
            filepath = os.path.join(Config.UPLOAD_FOLDER, unique_name)
            file.save(filepath)                                    # 保存原始图片
            img, detections = detector.detect(filepath)            # 执行检测
            if img is None:
                flash('无法读取图片', 'danger')
                return redirect(request.url)
            result_img = detector.draw_boxes(img, detections)      # 绘制检测框
            result_filename = f"res_{unique_name}"
            result_path = os.path.join(Config.RESULT_FOLDER, result_filename)
            cv2.imwrite(result_path, result_img)                   # 保存结果图片
            return render_template('detect_result.html',
                                   image_url=url_for('static', filename=f'results/{result_filename}'),
                                   detections=detections,
                                   original_filename=original_filename)
        # 处理文件夹或批量文件上传（name="files"）
        elif 'files' in request.files:
            files = request.files.getlist('files')
            if not files or files[0].filename == '':
                flash('未选择文件', 'warning')
                return redirect(request.url)

            # 判断是否为文件夹上传（浏览器会添加 webkitRelativePath 属性）
            is_folder_upload = any(hasattr(f, 'webkitRelativePath') and f.webkitRelativePath for f in files)

            if is_folder_upload:
                # 文件夹上传：显示结果页面
                return handle_folder_detection(detector, files)
            else:
                # 普通多文件上传：打包为 ZIP 下载
                zip_path = os.path.join(Config.RESULT_FOLDER, 'batch_results.zip')
                with zipfile.ZipFile(zip_path, 'w') as zf:
                    for file in files:
                        if file.filename == '':
                            continue
                        safe_name = secure_filename(file.filename)
                        if not safe_name:
                            safe_name = f"batch_{int(time.time())}.jpg"
                        name, ext = os.path.splitext(safe_name)
                        unique_name = f"{name}_{int(time.time()*1000)}{ext}"
                        filepath = os.path.join(Config.UPLOAD_FOLDER, unique_name)
                        file.save(filepath)
                        img, detections = detector.detect(filepath)
                        if img is not None:
                            result_img = detector.draw_boxes(img, detections)
                            out_path = os.path.join(Config.RESULT_FOLDER, unique_name)
                            cv2.imwrite(out_path, result_img)
                            zf.write(out_path, unique_name)        # 将结果图片加入 ZIP 文件
                return send_file(zip_path, as_attachment=True, download_name='detection_results.zip')
    return render_template('detect.html')

def handle_folder_detection(detector, files):
    """处理文件夹上传：检测每个文件，生成结果页面"""
    results = []
    processed_count = 0

    for file in files:
        # 只处理图片文件
        if not file.filename or not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue

        try:
            # 提取纯文件名（去除可能包含的子目录路径）
            original_filename = os.path.basename(file.filename)
            safe_name = secure_filename(original_filename)
            if not safe_name:
                safe_name = f"image_{int(time.time())}.jpg"
            name, ext = os.path.splitext(safe_name)
            unique_filename = f"{name}_{int(time.time()*1000)}{ext}"
            filepath = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
            filepath = os.path.abspath(filepath).replace('\\', '/')   # 统一斜杠
            file.save(filepath)                                      # 保存原始图片
            print(f"[DEBUG] 保存文件: {filepath}")

            # 调用检测器
            img, detections = detector.detect(filepath)
            processed_count += 1

            if img is not None and len(detections) > 0:
                # 有缺陷：绘制结果图并保存
                result_img = detector.draw_boxes(img, detections)
                result_filename = f"detected_{unique_filename}"
                result_path = os.path.join(Config.RESULT_FOLDER, result_filename)
                cv2.imwrite(result_path, result_img)

                results.append({
                    'filename': original_filename,                      # 显示用原始文件名
                    'relative_path': getattr(file, 'webkitRelativePath', original_filename),  # 文件夹相对路径
                    'original_path': filepath,                          # 绝对路径，供详情页使用
                    'detection_count': len(detections),
                    'result_image': url_for('static', filename=f'results/{result_filename}'),
                    'detections': detections,
                    'has_defects': True
                })
            else:
                # 无缺陷
                results.append({
                    'filename': original_filename,
                    'relative_path': getattr(file, 'webkitRelativePath', original_filename),
                    'original_path': filepath,
                    'detection_count': 0,
                    'result_image': None,
                    'detections': [],
                    'has_defects': False
                })

        except Exception as e:
            print(f"[ERROR] 处理文件 {file.filename} 时出错: {str(e)}")
            results.append({
                'filename': os.path.basename(file.filename),
                'relative_path': getattr(file, 'webkitRelativePath', file.filename),
                'error': str(e),
                'has_defects': False
            })

    # 统计信息
    defect_count = sum(1 for r in results if r.get('has_defects'))
    total_detections = sum(r.get('detection_count', 0) for r in results)

    summary = {
        'total_images': processed_count,
        'defect_images': defect_count,
        'total_detections': total_detections,
        'defect_rate': (defect_count / processed_count * 100) if processed_count > 0 else 0
    }

    # 获取文件夹名称（从第一个文件的 webkitRelativePath 中提取）
    folder_name = files[0].webkitRelativePath.split('/')[0] if hasattr(files[0], 'webkitRelativePath') else 'uploaded_folder'
    return render_template('folder_detect_result.html',
                           results=results,
                           summary=summary,
                           folder_name=folder_name)

@app.route('/api/detect-folder', methods=['POST'])
@login_required
def api_detect_folder():
    """API 端点：处理文件夹检测请求（返回 JSON）"""
    if 'files' not in request.files:
        return {'success': False, 'error': '未接收到文件'}, 400
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return {'success': False, 'error': '未选择文件夹'}, 400
    detector = get_detector()
    return handle_folder_detection(detector, files)

@app.route('/api/upload-model', methods=['POST'])
@login_required
def api_upload_model():
    """API 端点：上传自定义模型权重文件"""
    global detector
    if 'model_file' not in request.files:
        return {'success': False, 'error': '未选择模型文件'}, 400
    file = request.files['model_file']
    if not file.filename:
        return {'success': False, 'error': '文件名为空'}, 400
    allowed_extensions = {'.pt', '.pth', '.onnx'}   # 支持的扩展名
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        return {'success': False, 'error': f'不支持的格式: {ext}，仅支持 .pt/.pth/.onnx'}, 400
    try:
        model_path = os.path.join(Config.UPLOAD_FOLDER, f"custom_{file.filename}")
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        file.save(model_path)
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            if file_size > 100:   # 文件大小至少 100 字节，防止无效文件
                if detector is not None:
                    del detector
                    detector = None
                size_mb = file_size / (1024 * 1024)
                return {
                    'success': True,
                    'message': f'模型上传成功: {file.filename}',
                    'filename': file.filename,
                    'path': model_path,
                    'size': f'{size_mb:.2f} MB',
                    'size_bytes': file_size
                }
            else:
                os.remove(model_path)
                return {'success': False, 'error': f'模型文件过小 ({file_size} bytes)，可能是损坏的文件'}, 400
        else:
            return {'success': False, 'error': '文件保存失败'}, 500
    except Exception as e:
        print(f"[ERROR] 模型上传失败: {str(e)}")
        return {'success': False, 'error': f'上传失败: {str(e)}'}, 500

@app.route('/api/export-single-report', methods=['POST'])
@login_required
def api_export_single_report():
    """导出单张图片检测报告（PDF或Excel）"""
    try:
        data = request.get_json()
        if not data:
            return {'success': False, 'error': '请求数据为空'}, 400
        
        image_url = data.get('image_url', '')
        detections = data.get('detections', [])
        original_filename = data.get('original_filename', 'image.jpg')
        report_type = data.get('report_type', 'pdf')  # 'pdf' or 'excel'
        
        generator = ReportGenerator()
        
        if image_url:
            result_image_path = os.path.join(Config.RESULT_FOLDER, os.path.basename(image_url))
        else:
            result_image_path = None
        
        if report_type == 'excel':
            file_path, filename = generator.generate_single_excel_report(detections, original_filename)
            download_name = f"NEU-DET_Excel_{original_filename.rsplit('.', 1)[0]}.xlsx"
        else:
            file_path, filename = generator.generate_single_image_report(
                image_url, detections, original_filename, result_image_path
            )
            download_name = f"NEU-DET_Report_{original_filename.rsplit('.', 1)[0]}.pdf"
        
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name=download_name)
        else:
            return {'success': False, 'error': '报告生成失败'}, 500
            
    except Exception as e:
        print(f"[ERROR] 导出单图报告失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': f'导出失败: {str(e)}'}, 500

@app.route('/api/export-batch-report', methods=['POST'])
@login_required
def api_export_batch_report():
    """导出文件夹批量检测报告（PDF或Excel）"""
    try:
        data = request.get_json()
        if not data:
            return {'success': False, 'error': '请求数据为空'}, 400
        
        results = data.get('results', [])
        summary = data.get('summary', {})
        folder_name = data.get('folder_name', 'batch_results')
        report_type = data.get('report_type', 'pdf')  # 'pdf' or 'excel'
        
        generator = ReportGenerator()
        
        if report_type == 'excel':
            file_path, filename = generator.generate_batch_excel_report(results, summary, folder_name)
            download_name = f"NEU-DET_Batch_Excel_{folder_name}.xlsx"
        else:
            file_path, filename = generator.generate_batch_pdf_report(results, summary, folder_name)
            download_name = f"NEU-DET_Batch_Report_{folder_name}.pdf"
        
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name=download_name)
        else:
            return {'success': False, 'error': '报告生成失败'}, 500
            
    except Exception as e:
        print(f"[ERROR] 导出批量报告失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': f'导出失败: {str(e)}'}, 500

@app.route('/user-manage')
@login_required
@record_visitor
def user_manage():
    """用户管理页面（仅管理员可见）"""
    if current_user.role != 'admin':
        flash('无权限访问', 'danger')
        return redirect(url_for('dashboard'))
    users = User.query.all()
    return render_template('user_manage.html', users=users)

@app.route('/user/add', methods=['POST'])
@login_required
def add_user():
    """添加用户（仅管理员）"""
    if current_user.role != 'admin':
        return 'Forbidden', 403
    username = request.form['username']
    password = request.form['password']
    role = request.form.get('role', 'user')
    if User.query.filter_by(username=username).first():
        flash('用户名已存在', 'warning')
    else:
        new_user = User(username=username, password=generate_password_hash(password), role=role)
        db.session.add(new_user)
        db.session.commit()
        flash('添加成功', 'success')
    return redirect(url_for('user_manage'))

@app.route('/user/edit/<int:user_id>', methods=['POST'])
@login_required
def edit_user(user_id):
    """编辑用户信息（仅管理员）"""
    if current_user.role != 'admin':
        return 'Forbidden', 403
    user = User.query.get_or_404(user_id)
    user.username = request.form.get('username', user.username)
    if request.form.get('password'):
        user.password = generate_password_hash(request.form['password'])
    user.role = request.form.get('role', user.role)
    db.session.commit()
    flash('更新成功', 'success')
    return redirect(url_for('user_manage'))

@app.route('/user/delete/<int:user_id>')
@login_required
def delete_user(user_id):
    """删除用户（仅管理员，不能删除默认管理员）"""
    if current_user.role != 'admin':
        return 'Forbidden', 403
    user = User.query.get_or_404(user_id)
    if user.username == 'admin':
        flash('不能删除管理员', 'danger')
    else:
        db.session.delete(user)
        db.session.commit()
        flash('删除成功', 'success')
    return redirect(url_for('user_manage'))

@app.route('/visitor-records')
@login_required
@record_visitor
def visitor_records():
    """访客记录页面（仅管理员）"""
    if current_user.role != 'admin':
        flash('无权限访问', 'danger')
        return redirect(url_for('dashboard'))
    # 查询最近 200 条记录，按时间倒序
    records = VisitorRecord.query.order_by(VisitorRecord.timestamp.desc()).limit(200).all()
    return render_template('visitor_records.html', records=records)

@app.route('/detect/detail')
@login_required
def detect_detail():
    """单张图片详情页：根据原始图片路径重新检测并显示结果"""
    img_path = request.args.get('path')
    filename = request.args.get('name', '图片')

    if not img_path:
        flash('未提供图片路径', 'danger')
        return redirect(url_for('detect'))

    # URL 解码并统一斜杠分隔符
    img_path = urllib.parse.unquote(img_path).replace('\\', '/')
    print(f"[DETAIL] 请求图片: {img_path}")

    # 如果路径不存在，尝试从上传目录中取文件名匹配
    if not os.path.exists(img_path):
        base_name = os.path.basename(img_path)
        fallback = os.path.join(Config.UPLOAD_FOLDER, base_name)
        if os.path.exists(fallback):
            img_path = fallback
        else:
            flash(f'图片不存在: {filename}', 'danger')
            return redirect(url_for('detect'))

    detector = get_detector()
    img, detections = detector.detect(img_path)
    if img is None:
        flash('无法检测图片，请检查图片格式', 'danger')
        return redirect(url_for('detect'))

    # 绘制结果并保存
    result_img = detector.draw_boxes(img, detections)
    result_filename = f"detail_{os.path.basename(img_path)}"
    result_path = os.path.join(Config.RESULT_FOLDER, result_filename)
    cv2.imwrite(result_path, result_img)

    return render_template('detect_result.html',
                           image_url=url_for('static', filename=f'results/{result_filename}'),
                           detections=detections,
                           original_filename=filename)

if __name__ == '__main__':
    init_db()               # 初始化数据库
    app.run(debug=True, host='0.0.0.0', port=8080)   # 启动 Flask 开发服务器