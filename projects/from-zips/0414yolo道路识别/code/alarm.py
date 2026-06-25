from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets
import os
import sys
from pathlib import Path
import numpy as np
import time
import argparse
import platform
import torch
import sqlite3
import random
import string
from scipy.spatial import distance
import traceback
import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import masks2segments, process_mask, process_mask_native
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox


# 初始化数据库
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    ''')
    conn.commit()
    conn.close()


# 生成随机验证码
def generate_captcha(length=4):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))


class LoginWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.captcha_text = generate_captcha()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("登录")
        self.setFixedSize(400, 300)
        layout = QVBoxLayout()

        # 标题
        title_label = QLabel("车道线检测系统登录")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # 登录表单
        form_layout = QFormLayout()

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("请输入用户名")
        form_layout.addRow("用户名:", self.username_input)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("请输入密码")
        self.password_input.setEchoMode(QLineEdit.Password)
        form_layout.addRow("密码:", self.password_input)

        # 验证码
        captcha_layout = QHBoxLayout()
        self.captcha_label = QLabel(self.captcha_text)
        self.captcha_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.captcha_label.setStyleSheet("background-color: #f0f0f0; padding: 5px;")

        self.captcha_input = QLineEdit()
        self.captcha_input.setPlaceholderText("请输入验证码")

        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_captcha)

        captcha_layout.addWidget(QLabel("验证码:"))
        captcha_layout.addWidget(self.captcha_label)
        captcha_layout.addWidget(refresh_btn)
        layout.addLayout(form_layout)
        layout.addLayout(captcha_layout)
        layout.addWidget(self.captcha_input)

        # 按钮
        btn_layout = QHBoxLayout()

        login_btn = QPushButton("登录")
        login_btn.clicked.connect(self.login)

        register_btn = QPushButton("注册")
        register_btn.clicked.connect(self.go_to_register)

        btn_layout.addWidget(login_btn)
        btn_layout.addWidget(register_btn)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def refresh_captcha(self):
        self.captcha_text = generate_captcha()
        self.captcha_label.setText(self.captcha_text)

    def login(self):
        username = self.username_input.text()
        password = self.password_input.text()
        captcha = self.captcha_input.text()

        # 验证码校验
        if captcha.upper() != self.captcha_text.upper():
            QMessageBox.warning(self, "错误", "验证码错误，请重新输入")
            return

        # 数据库验证
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            QMessageBox.information(self, "成功", "登录成功")

            # 创建全局主窗口（这样可以防止被垃圾回收）
            global MainWindow
            MainWindow = QtWidgets.QMainWindow()
            global ui
            ui = Ui_MainWindow()
            ui.setupUi(MainWindow)
            MainWindow.show()
            self.close()
        else:
            QMessageBox.warning(self, "错误", "用户名或密码错误")

    def go_to_register(self):
        self.register_window = RegisterWindow(self)
        self.register_window.show()
        self.hide()


class RegisterWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.captcha_text = generate_captcha()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("注册")
        self.setFixedSize(400, 350)
        layout = QVBoxLayout()

        # 标题
        title_label = QLabel("注册新账号")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # 注册表单
        form_layout = QFormLayout()

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("请输入用户名")
        form_layout.addRow("用户名:", self.username_input)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("请输入密码")
        self.password_input.setEchoMode(QLineEdit.Password)
        form_layout.addRow("密码:", self.password_input)

        self.confirm_input = QLineEdit()
        self.confirm_input.setPlaceholderText("请再次输入密码")
        self.confirm_input.setEchoMode(QLineEdit.Password)
        form_layout.addRow("确认密码:", self.confirm_input)

        # 验证码
        captcha_layout = QHBoxLayout()
        self.captcha_label = QLabel(self.captcha_text)
        self.captcha_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.captcha_label.setStyleSheet("background-color: #f0f0f0; padding: 5px;")

        self.captcha_input = QLineEdit()
        self.captcha_input.setPlaceholderText("请输入验证码")

        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_captcha)

        captcha_layout.addWidget(QLabel("验证码:"))
        captcha_layout.addWidget(self.captcha_label)
        captcha_layout.addWidget(refresh_btn)
        layout.addLayout(form_layout)
        layout.addLayout(captcha_layout)
        layout.addWidget(self.captcha_input)

        # 按钮
        btn_layout = QHBoxLayout()

        back_btn = QPushButton("返回登录")
        back_btn.clicked.connect(self.go_to_login)

        register_btn = QPushButton("注册")
        register_btn.clicked.connect(self.register)

        btn_layout.addWidget(back_btn)
        btn_layout.addWidget(register_btn)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def refresh_captcha(self):
        self.captcha_text = generate_captcha()
        self.captcha_label.setText(self.captcha_text)

    def register(self):
        username = self.username_input.text()
        password = self.password_input.text()
        confirm = self.confirm_input.text()
        captcha = self.captcha_input.text()

        if not username or not password:
            QMessageBox.warning(self, "错误", "用户名和密码不能为空")
            return

        if password != confirm:
            QMessageBox.warning(self, "错误", "两次输入的密码不一致")
            return

        if captcha.upper() != self.captcha_text.upper():
            QMessageBox.warning(self, "错误", "验证码错误，请重新输入")
            return

        # 保存到数据库
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            QMessageBox.information(self, "成功", "注册成功，请登录")
            self.go_to_login()
        except sqlite3.IntegrityError:
            QMessageBox.warning(self, "错误", "用户名已存在")
        finally:
            conn.close()

    def go_to_login(self):
        self.parent.show()
        self.close()


class LaneVehicleDetector:
    def __init__(self):
        self.lane_color = [0, 255, 0]  # 绿色表示车道线
        self.warning_color = [0, 0, 255]  # 红色表示警告

        # 车辆检测参数
        self.vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
        self.safe_distance = 400  # 像素距离
        self.lane_change_threshold = 400  # 横向移动阈值(像素)
        self.frame_window = 5  # 判断变道的帧数窗口

        self.prev_positions = {}
        self.warning_messages = []
        self.warning_boxes = []

    def calculate_lane_deviation(self, vehicle_box, lane_masks):
        """计算车辆与车道线的偏移距离"""
        if len(lane_masks) == 0:
            return 0, False

        x1, y1, x2, y2 = map(int, vehicle_box)

        vehicle_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        vehicle_bottom = ((x1 + x2) // 2, y2)
        min_dist = float('inf')
        for mask in lane_masks:
            if isinstance(mask, torch.Tensor):
                mask_np = mask.squeeze().cpu().numpy()
            else:
                mask_np = mask

            mask_np = (mask_np > 0).astype(np.uint8)

            # 获取车道线轮廓
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # 计算到所有轮廓点的最小距离
                for cnt in contours:
                    for point in cnt:
                        # 使用车辆底部中心点计算距离
                        dist = distance.euclidean(vehicle_bottom, point[0])
                        if dist < min_dist:
                            min_dist = dist

        is_deviating = min_dist > self.safe_distance
        return min_dist, is_deviating

    def detect_lane_changes(self, track_id, current_position, current_frame):
        if track_id in self.prev_positions:
            prev_pos, prev_frame = self.prev_positions[track_id]
            x_diff = abs(current_position[0] - prev_pos[0])
            if x_diff > self.lane_change_threshold and (current_frame - prev_frame) < self.frame_window:
                return True

        self.prev_positions[track_id] = (current_position, current_frame)
        return False

    def process_detections(self, detections, lane_masks, frame_count):
        self.warning_messages = []
        self.warning_boxes = []

        for det in detections:
            if len(det) == 3:
                label, box, conf = det
            elif len(det) == 4:
                label, box, conf, _ = det
            else:
                continue

            if not isinstance(box, (list, tuple)) or len(box) < 4:
                continue

            try:
                box = list(map(int, box[:4]))
                dist, is_deviating = self.calculate_lane_deviation(box, lane_masks)

                x1, y1, x2, y2 = box
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                is_lane_changing = self.detect_lane_changes(id(box), center, frame_count)

                if is_deviating or is_lane_changing:
                    warning_type = "Change Lanes" if is_lane_changing else "Lanes Deviation"
                    warning_msg = f"{warning_type}: {label} (conf: {conf:.2f})"
                    self.warning_messages.append(warning_msg)
                    self.warning_boxes.append((box, warning_type))

            except (ValueError, TypeError) as e:
                print(f"处理检测框时出错: {e}, 检测数据: {det}")
                continue

        return self.warning_boxes


def load_model(
        weights='',  # model.pt path(s)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    return model, stride, names, pt


# 加载两个模型
def load_models():
    root_path = os.path.dirname(os.path.abspath(__file__))
    lane_path = os.path.join(root_path, 'best.pt')
    vehicle_path = os.path.join(root_path, 'yolov5s.pt')

    lane_model, lane_stride, lane_names, lane_pt = load_model(
        weights=lane_path
    )

    # 车辆检测模型
    vehicle_model, vehicle_stride, vehicle_names, vehicle_pt = load_model(
        weights=vehicle_path
    )

    return lane_model, lane_stride, lane_names, lane_pt, vehicle_model, vehicle_stride, vehicle_names, vehicle_pt


def run(lane_model, vehicle_model, img, lane_stride, vehicle_stride, lane_pt, vehicle_pt, frame_count=0,
        imgsz=(640, 640), conf_thres=0.25, iou_thres=0.45,
        max_det=1000, device='', classes=None, agnostic_nms=False,
        augment=False, half=False, retina_masks=True):
    lane_im = letterbox(img, imgsz, stride=lane_stride, auto=lane_pt)[0]
    lane_im = lane_im.transpose((2, 0, 1))[::-1]
    lane_im = np.ascontiguousarray(lane_im)

    vehicle_im = letterbox(img, imgsz, stride=vehicle_stride, auto=vehicle_pt)[0]
    vehicle_im = vehicle_im.transpose((2, 0, 1))[::-1]
    vehicle_im = np.ascontiguousarray(vehicle_im)

    device = select_device(device)
    lane_im = torch.from_numpy(lane_im).to(device)
    lane_im = lane_im.half() if half else lane_im.float()
    lane_im /= 255

    vehicle_im = torch.from_numpy(vehicle_im).to(device)
    vehicle_im = vehicle_im.half() if half else vehicle_im.float()
    vehicle_im /= 255

    if len(lane_im.shape) == 3:
        lane_im = lane_im[None]

    if len(vehicle_im.shape) == 3:
        vehicle_im = vehicle_im[None]

    with torch.no_grad():
        lane_pred, lane_proto = lane_model(lane_im, augment=augment)[:2]
        vehicle_pred = vehicle_model(vehicle_im, augment=augment)[0]

    lane_pred = non_max_suppression(lane_pred, conf_thres, iou_thres, classes,
                                    agnostic_nms, max_det=max_det, nm=32)
    vehicle_pred = non_max_suppression(vehicle_pred, conf_thres, iou_thres, classes,
                                       agnostic_nms, max_det=max_det, nm=32)

    detector = LaneVehicleDetector()

    lane_masks = []
    vehicle_detections = []

    for i, det in enumerate(lane_pred):
        if len(det):
            det[:, :4] = scale_boxes(lane_im.shape[2:], det[:, :4], img.shape).round()
            masks = process_mask_native(lane_proto[i], det[:, 6:], det[:, :4], img.shape[:2])
            segments = [
                scale_segments(lane_im.shape[2:], x, img.shape, normalize=True)
                for x in reversed(masks2segments(masks))]

            # 分类处理车道线
            for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                label = lane_model.names[int(cls)]
                lane_masks.append(masks[j])

    for i, det in enumerate(vehicle_pred):
        if len(det):
            det[:, :4] = scale_boxes(vehicle_im.shape[2:], det[:, :4], img.shape).round()

            for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                label = vehicle_model.names[int(cls)]
                if label in ['car', 'truck', 'bus', 'motorcycle']:
                    vehicle_detections.append([
                        label,
                        [int(x) for x in xyxy],
                        float(conf)
                    ])

    warning_boxes = detector.process_detections(vehicle_detections, lane_masks, frame_count)
    annotator = Annotator(img.copy(), line_width=2, example=str(lane_model.names))

    for mask in lane_masks:
        mask_np = mask.squeeze().cpu().numpy() if isinstance(mask, torch.Tensor) else mask
        mask_rgb = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
        mask_rgb[mask_np > 0] = detector.lane_color
        img = cv2.addWeighted(img, 1, mask_rgb, 0.3, 0)

    for box, warning_type in warning_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), detector.warning_color, 3)

        label = f"{warning_type}!"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(img, (x1, y1 - 30), (x1 + tw, y1), detector.warning_color, -1)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        cv2.circle(img, center, 5, (255, 0, 0), -1)

    status = "Normal" if not warning_boxes else "Warning"
    status_color = (0, 255, 0) if status == "Normal" else (0, 0, 255)

    cv2.putText(img, f"Status: {status}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

    if warning_boxes:
        cv2.putText(img, f"Warning Num: {len(warning_boxes)}", (img.shape[1] - 200, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return vehicle_detections, lane_masks, detector.warning_messages, img


def det_yolov5v7(info1):
    global lane_model, vehicle_model, lane_stride, vehicle_stride, lane_pt, vehicle_pt, ui, frame_count

    # 初始化帧计数器（用于变道检测）
    frame_count = getattr(det_yolov5v7, 'frame_count', 0) + 1
    det_yolov5v7.frame_count = frame_count

    # 图片处理
    if info1.lower().endswith(('jpg', 'png', 'jpeg', 'bmp')):
        try:
            # 读取图片
            img = cv2.imread(info1)
            if img is None:
                ui.printf("[错误] 无法读取图片文件")
                return

            # 执行检测
            start_time = time.time()
            vehicles, lanes, warnings, vis_img = run(
                lane_model, vehicle_model, img, lane_stride, vehicle_stride, lane_pt, vehicle_pt, frame_count
            )
            cost_time = time.time() - start_time

            # 更新UI
            ui.textBrowser.clear()
            ui.printf(f"图片分析完成 (耗时: {cost_time:.2f}s)")
            ui.printf(f"检测到 {len(vehicles)} 辆车辆和 {len(lanes)} 条车道线")

            # 显示警告信息
            if warnings:
                for warn in warnings:
                    ui.printf(f"[!] {warn}")
                ui.label3.setStyleSheet("color: red; font-weight: bold;")
                ui.label3.setText("⚠️ 检测到危险驾驶行为")
            else:
                ui.label3.setStyleSheet("color: green;")
                ui.label3.setText("✅ 行车状态正常")

            # 显示结果图片
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            ui.showimg(vis_img)

            # 保存结果
            result_path = f"./result/{Path(info1).name}"
            cv2.imwrite(result_path, vis_img)
            ui.printf(f"结果已保存至: {result_path}")

        except Exception as e:
            ui.printf(f"[错误] 图片处理失败: {str(e)}")
            traceback.print_exc()

    # 视频处理
    elif info1.lower().endswith(('mp4', 'avi', 'mov')):
        cap = None
        try:
            # 打开视频
            cap = cv2.VideoCapture(info1)
            if not cap.isOpened():
                ui.printf("[ui错误] 无法打开视频文件")
                return

            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ui.printf(f"开始分析视频: {Path(info1).name}")
            ui.printf(f"视频信息: {fps:.1f} FPS, 共 {total_frames} 帧")

            # 创建视频保存路径
            result_video = f"./result/{Path(info1).stem}_output.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(result_video, fourcc, fps,
                                  (int(cap.get(3)), int(cap.get(4))))

            # 逐帧处理
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 执行检测
                vehicles, lanes, warnings, vis_frame = run(
                    lane_model, vehicle_model, frame, lane_stride, vehicle_stride, lane_pt, vehicle_pt, frame_count
                )

                # 更新UI
                ui.textBrowser.clear()
                ui.printf(f"处理进度: {frame_count}/{total_frames} 帧")
                ui.printf(f"当前帧检测: {len(vehicles)} 辆车辆")

                # 实时显示警告
                if warnings:
                    for warn in warnings:
                        ui.printf(f"[帧 {frame_count}] {warn}")

                # 调用 update_warning_display 函数
                ui.update_warning_display(warnings)

                # 显示处理帧
                ui.showimg(vis_frame)
                out.write(vis_frame)

                frame_count += 1
                QApplication.processEvents()

            # 完成处理
            out.release()
            ui.printf(f"视频分析完成，结果已保存至: {result_video}")
            ui.label3.setText("🎬 视频处理完成")

        except Exception as e:
            ui.printf(f"[错误] 视频处理失败: {str(e)}")
            traceback.print_exc()
        finally:
            if cap is not None:
                cap.release()

    else:
        ui.printf("[错误] 不支持的文件格式，请使用jpg/png/mp4/avi")


class Thread_1(QThread):  # 线程1
    def __init__(self, info1):
        super().__init__()
        self.info1 = info1
        self.run2(self.info1)

    def run2(self, info1):
        result = []
        result = det_yolov5v7(info1)


class Ui_MainWindow(object):
    def __init__(self):
        self.stop_flag = False

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 960)
        MainWindow.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1e3c72, stop:1 #2a5298);
            }
            QPushButton {
                background: rgba(255,142,0,0.9);
                border-radius: 10px;
                padding: 8px 16px;
                color: white;
                font: bold 16px '微软雅黑';
                min-width: 120px;
            }
            QPushButton:hover {
                background: rgba(255,142,0,1);
            }
            QTextBrowser {
                background: rgba(255,255,255,0.8);
                border-radius: 5px;
                font: 14px 'Consolas';
                padding: 10px;
            }
        """)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(168, 60, 600, 60))
        self.label.setAutoFillBackground(False)
        self.label.setStyleSheet("""
            font-size:32px;
            font-weight:bold;
            font-family:SimHei;
            color: white;
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
        """)
        self.label.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(40, 150, 900, 760))
        self.label_2.setStyleSheet("background:rgba(255,255,255,0.3);")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")

        self.label3 = QtWidgets.QLabel(self.centralwidget)
        self.label3.setGeometry(QtCore.QRect(450, 920, 870, 39))
        self.label3.setAutoFillBackground(False)
        self.label3.setStyleSheet(
            "font-size:16px;font-weight:bold;font-family:SimHei;color:white;background:rgba(255,255,255,0);")
        self.label3.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label3.setAlignment(QtCore.Qt.AlignCenter)
        self.label3.setObjectName("label")

        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(950, 150, 300, 580))
        self.textBrowser.setStyleSheet("background:rgba(255,255,255,0.5);")
        self.textBrowser.setObjectName("textBrowser")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(950, 740, 150, 40))
        self.pushButton.setStyleSheet("background:rgba(255,142,0,1);border-radius:10px;padding:2px 4px;")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(1110, 740, 150, 40))
        self.pushButton_2.setStyleSheet("background:rgba(255,142,0,1);border-radius:10px;padding:2px 4px;")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(950, 790, 150, 40))
        self.pushButton_3.setStyleSheet("background:rgba(255,142,0,1);border-radius:10px;padding:2px 4px;")
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(1110, 790, 150, 40))
        self.pushButton_4.setStyleSheet("background:rgba(255,142,0,1);border-radius:10px;padding:2px 4px;")
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(1030, 840, 150, 40))
        self.pushButton_5.setStyleSheet("background:rgba(255,142,0,1);border-radius:10px;padding:2px 4px;")
        self.pushButton_5.setObjectName("pushButton_5")

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "基于深度学习的车道线检测系统"))
        self.label.setText(_translate("MainWindow", "基于深度学习的车道线检测系统"))
        self.label_2.setText(_translate("MainWindow", "请添加对象"))
        self.label3.setText(_translate("MainWindow", ""))
        self.pushButton.setText(_translate("MainWindow", "选择对象"))
        self.pushButton_2.setText(_translate("MainWindow", "开始识别"))
        self.pushButton_3.setText(_translate("MainWindow", "实时检测"))
        self.pushButton_4.setText(_translate("MainWindow", "停止检测"))
        self.pushButton_5.setText(_translate("MainWindow", "退出系统"))

        # 点击文本框绑定槽事件
        self.pushButton.clicked.connect(self.openfile)
        self.pushButton_2.clicked.connect(self.click_1)
        self.pushButton_3.clicked.connect(self.handleCalc2)
        self.pushButton_4.clicked.connect(self.stop_detection)
        self.pushButton_5.clicked.connect(self.handleCalc3)
        self.printf('')

    def handleCalc2(self):
        global lane_model, vehicle_model, lane_stride, vehicle_stride, lane_pt, vehicle_pt, ui, frame_count
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            ui.printf("[错误] 无法打开摄像头")
            return

        # 创建视频保存路径
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_video = f"./result/webcam_output_{timestamp}.mp4"
        fps = 30  # 假设默认30fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(result_video, fourcc, fps,
                              (int(capture.get(3)), int(capture.get(4))))

        self.stop_flag = False
        while not self.stop_flag:
            ret, image = capture.read()
            if not ret or image is None:
                ui.printf("[错误] 无法读取摄像头帧")
                break
            try:
                frame_count = getattr(self, 'frame_count', 0) + 1
                self.frame_count = frame_count

                vehicles, lanes, warnings, vis_image = run(
                    lane_model, vehicle_model, image, lane_stride, vehicle_stride, lane_pt, vehicle_pt, frame_count
                )

                # 更新UI
                ui.textBrowser.clear()
                ui.printf(f"实时检测: 帧 {frame_count}")
                ui.printf(f"检测到 {len(vehicles)} 辆车辆和 {len(lanes)} 条车道线")

                # 显示警告信息
                if warnings:
                    for warn in warnings:
                        ui.printf(f"[!] {warn}")
                    ui.label3.setStyleSheet("color: red; font-weight: bold;")
                    ui.label3.setText("⚠️ 检测到危险驾驶行为")
                else:
                    ui.label3.setStyleSheet("color: green;")
                    ui.label3.setText("✅ 行车状态正常")

                # 显示结果
                vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
                ui.showimg(vis_image)
                out.write(cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

                QApplication.processEvents()

            except Exception as e:
                ui.printf(f"[错误] 摄像头处理失败: {str(e)}")
                traceback.print_exc()

        capture.release()
        out.release()
        ui.printf(f"摄像头检测停止，结果已保存至: {result_video}")

    def stop_detection(self):
        self.stop_flag = True
        ui.printf("用户手动停止检测")

    def openfile(self):
        global sname, filepath
        fname = QFileDialog()
        fname.setAcceptMode(QFileDialog.AcceptOpen)
        fname, _ = fname.getOpenFileName()
        if fname == '':
            return
        filepath = os.path.normpath(fname)
        sname = filepath.split(os.sep)
        ui.printf("当前选择的文件路径是：%s" % filepath)

    def handleCalc3(self):
        os._exit(0)

    def printf(self, text):
        self.textBrowser.append(text)
        self.cursor = self.textBrowser.textCursor()
        self.textBrowser.moveCursor(self.cursor.End)
        QtWidgets.QApplication.processEvents()

    def showimg(self, img):
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _image = QtGui.QImage(img2[:], img2.shape[1], img2.shape[0], img2.shape[1] * 3,
                              QtGui.QImage.Format_RGB888)
        n_width = _image.width()
        n_height = _image.height()
        if n_width / 500 >= n_height / 400:
            ratio = n_width / 900
        else:
            ratio = n_height / 900
        new_width = int(n_width / ratio)
        new_height = int(n_height / ratio)
        new_img = _image.scaled(new_width, new_height, Qt.KeepAspectRatio)
        self.label_2.setPixmap(QPixmap.fromImage(new_img))

    def click_1(self):
        global filepath
        try:
            self.thread_1.quit()
        except:
            pass
        self.thread_1 = Thread_1(filepath)
        self.thread_1.wait()
        self.thread_1.start()

    def update_warning_display(self, warnings):
        """更新警告显示"""
        self.textBrowser.clear()
        for warning in warnings:
            self.printf(warning)

        # 如果有警告，闪烁提示
        if warnings:
            self.label3.setStyleSheet("color: red; font-weight: bold;")
            self.label3.setText("⚠️ 检测到危险驾驶行为!")
        else:
            self.label3.setStyleSheet("color: green;")
            self.label3.setText("✅ 行车状态正常")


if __name__ == "__main__":
    try:
        # 创建全局变量
        global MainWindow, ui

        # 创建应用
        app = QApplication(sys.argv)

        # 初始化数据库
        init_db()

        # 加载模型
        global lane_model, lane_stride, lane_names, lane_pt, vehicle_model, vehicle_stride, vehicle_names, vehicle_pt
        print("正在加载模型...")
        lane_model, lane_stride, lane_names, lane_pt, vehicle_model, vehicle_stride, vehicle_names, vehicle_pt = load_models()
        print("模型加载完成")

        # 创建登录窗口并显示
        login_window = LoginWindow()
        login_window.setWindowTitle("车道线检测系统登录")
        login_window.show()
        print("登录窗口已显示")

        # 确保窗口处理完成
        QApplication.processEvents()

        # 进入事件循环
        sys.exit(app.exec_())
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback

        traceback.print_exc()