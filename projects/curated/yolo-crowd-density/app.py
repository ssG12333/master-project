import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QFileDialog, QDialog, QLineEdit, QFormLayout)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPalette, QBrush, QFont
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

class LoginWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("登录")
        self.setFixedSize(300, 200)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        form_layout = QFormLayout()

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("请输入用户名")
        form_layout.addRow("用户名:", self.username_input)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("请输入密码")
        self.password_input.setEchoMode(QLineEdit.Password)
        form_layout.addRow("密码:", self.password_input)

        layout.addLayout(form_layout)

        self.login_button = QPushButton("登录")
        self.login_button.clicked.connect(self.check_credentials)
        layout.addWidget(self.login_button)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: red;")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

        self.setStyleSheet("""
            QDialog {
                background-color: rgba(255, 255, 255, 200);
                border-radius: 10px;
            }
            QLineEdit {
                padding: 8px;
                font-size: 14px;
                border-radius: 5px;
                border: 1px solid #ccc;
            }
            QPushButton {
                padding: 10px;
                font-size: 14px;
                background-color: #007BFF;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)

    def check_credentials(self):
        username = self.username_input.text()
        password = self.password_input.text()

        if username == "admin" and password == "password":
            self.accept()
        else:
            self.status_label.setText("用户名或密码错误")

class CrowdDensityMonitoringSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        try:
            self.model = YOLO("best.pt")
        except Exception as e:
            print(f"加载 YOLO 模型失败: {e}")
            sys.exit(1)
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.results_history = []
        self.person_count = 0
        self.font = None
        font_paths = [
            "SimHei.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/System/Library/Fonts/PingFang.ttc"
        ]
        for path in font_paths:
            try:
                self.font = ImageFont.truetype(path, 20)
                print(f"成功加载字体: {path}")
                break
            except Exception as e:
                print(f"加载字体 {path} 失败: {e}")
                continue
        if not self.font:
            try:
                self.font = ImageFont.truetype("NotoSansCJK-Regular.ttc", 20)
                print("成功加载备用字体: NotoSansCJK-Regular.ttc")
            except Exception as e:
                print(f"加载备用字体失败: {e}")
                self.font = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("人群密度监测系统")
        self.setFixedSize(1920, 1000)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)

        title_label = QLabel("人群密度监测系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 28, QFont.Bold))
        title_label.setStyleSheet("""
            color: white;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #007BFF, stop:1 #00BFFF);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        """)
        main_layout.addWidget(title_label)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)
        main_layout.addLayout(content_layout)

        display_layout = QHBoxLayout()
        display_layout.setSpacing(15)
        content_layout.addLayout(display_layout)

        original_layout = QVBoxLayout()
        original_title = QLabel("原图")
        original_title.setAlignment(Qt.AlignCenter)
        original_title.setFixedSize(640, 40)
        original_title.setStyleSheet("""
            font-size: 16px;
            color: white;
            background-color: rgba(0, 0, 0, 200);
            padding: 10px;
            border-radius: 8px;
        """)
        original_layout.addWidget(original_title)
        original_layout.addSpacing(10)
        self.original_window = QLabel()
        self.original_window.setFixedSize(640, 480)
        self.original_window.setStyleSheet("""
            border: 2px solid #333;
            background-color: #1a1a1a;
            border-radius: 8px;
        """)
        self.original_window.setAlignment(Qt.AlignCenter)
        original_layout.addWidget(self.original_window)
        display_layout.addLayout(original_layout)

        detection_layout = QVBoxLayout()
        detection_title = QLabel("检测情况")
        detection_title.setAlignment(Qt.AlignCenter)
        detection_title.setFixedSize(640, 40)
        detection_title.setStyleSheet("""
            font-size: 16px;
            color: white;
            background-color: rgba(0, 0, 0, 200);
            padding: 10px;
            border-radius: 8px;
        """)
        detection_layout.addWidget(detection_title)
        detection_layout.addSpacing(10)
        self.detect_window = QLabel()
        self.detect_window.setFixedSize(640, 480)
        self.detect_window.setStyleSheet("""
            border: 2px solid #333;
            background-color: #1a1a1a;
            border-radius: 8px;
        """)
        self.detect_window.setAlignment(Qt.AlignCenter)
        detection_layout.addWidget(self.detect_window)
        display_layout.addLayout(detection_layout)

        right_layout = QVBoxLayout()
        right_layout.setAlignment(Qt.AlignTop)
        content_layout.addLayout(right_layout)

        self.person_count_label = QLabel("检测到人数: 0")
        self.person_count_label.setFixedSize(300, 80)
        self.person_count_label.setAlignment(Qt.AlignCenter)
        self.person_count_label.setStyleSheet("""
            font-size: 20px;
            color: white;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #28a745, stop:1 #20c997);
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        """)
        right_layout.addWidget(self.person_count_label)
        right_layout.addSpacing(20)

        # 添加三个文本输入窗口
        input_form_layout = QFormLayout()
        self.area_input = QLineEdit()
        self.area_input.setPlaceholderText("请输入区域面积 (平方米)")
        self.area_input.setFixedSize(300, 40)
        self.area_input.setStyleSheet("""
            padding: 8px;
            font-size: 14px;
            border-radius: 5px;
            border: 1px solid #ccc;
            background-color: white;
        """)
        input_form_layout.addRow("区域面积:", self.area_input)

        self.density_input = QLineEdit()
        self.density_input.setPlaceholderText("请输入当前密度 (人/平方米)")
        self.density_input.setFixedSize(300, 40)
        self.density_input.setStyleSheet("""
            padding: 8px;
            font-size: 14px;
            border-radius: 5px;
            border: 1px solid #ccc;
            background-color: white;
        """)
        input_form_layout.addRow("当前密度:", self.density_input)

        self.population_input = QLineEdit()
        self.population_input.setPlaceholderText("请输入当前人数")
        self.population_input.setFixedSize(300, 40)
        self.population_input.setStyleSheet("""
            padding: 8px;
            font-size: 14px;
            border-radius: 5px;
            border: 1px solid #ccc;
            background-color: white;
        """)
        input_form_layout.addRow("当前人数:", self.population_input)

        right_layout.addLayout(input_form_layout)
        right_layout.addStretch()

        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        main_layout.addSpacing(30)
        main_layout.addLayout(button_layout)

        buttons = [
            ("选择图片", self.select_image),
            ("选择视频", self.select_video),
            ("启动摄像头", self.start_camera),
            ("关闭摄像头", self.stop_camera)
        ]
        for text, func in buttons:
            btn = QPushButton(text)
            btn.setFixedSize(160, 50)
            btn.clicked.connect(func)
            btn.setStyleSheet("""
                background-color: #007BFF;
                color: white;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
            """)
            btn.setStyleSheet(btn.styleSheet() + """
                QPushButton:hover {
                    background-color: #0056b3;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
                }
                QPushButton:pressed {
                    background-color: #003d80;
                }
            """)
            button_layout.addWidget(btn)

        button_layout.addStretch()

        try:
            background = QPixmap("background.jpg")
            palette = QPalette()
            palette.setBrush(QPalette.Background, QBrush(background.scaled(self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)))
            self.setPalette(palette)
        except Exception as e:
            print(f"加载背景图片失败: {e}")
            self.setStyleSheet("background-color: #f0f2f5;")

    def update_person_count(self, count):
        self.person_count = count
        self.person_count_label.setText(f"检测到人数: {self.person_count}")
        # 更新当前人数输入框
        self.population_input.setText(str(self.person_count))
        # 计算并更新密度
        try:
            area = float(self.area_input.text()) if self.area_input.text() else 0
            if area > 0:
                density = self.person_count / area
                self.density_input.setText(f"{density:.2f}")
            else:
                self.density_input.setText("0.00")
        except ValueError:
            self.density_input.setText("0.00")

    def select_image(self):
        self.stop_camera()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            img = cv2.imread(file_name)
            if img is not None:
                self.process_frame(img)
            else:
                print(f"无法加载图片: {file_name}")

    def select_video(self):
        self.stop_camera()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi)")
        if file_name:
            self.cap = cv2.VideoCapture(file_name)
            if self.cap.isOpened():
                self.timer.start(30)
            else:
                print(f"无法打开视频: {file_name}")

    def start_camera(self):
        self.stop_camera()
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.timer.start(30)
        else:
            print("无法打开摄像头")

    def stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.timer.stop()
        self.original_window.clear()
        self.detect_window.clear()
        self.update_person_count(0)
        self.results_history.clear()

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.process_frame(frame)
            else:
                print("无法读取视频帧")

    def process_frame(self, frame):
        try:
            if frame.shape[0] > 1080 or frame.shape[1] > 1920:
                frame = cv2.resize(frame, (1280, 720))
            original_frame = frame.copy()
            results = self.model(frame)
            frame_results = []
            boxes = []

            for result in results:
                boxes.extend(result.boxes)
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0].item()
                    cls = int(box.cls[0])
                    state = self.model.names[cls]
                    coords = (x1, y1, x2, y2)
                    frame_results.append((state, confidence, coords))

                    color = (0, 255, 0)
                    state_desc = state

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{state_desc} {confidence:.2f}"
                    if self.font:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(frame_rgb)
                        draw = ImageDraw.Draw(pil_img)
                        text_bbox = draw.textbbox((0, 0), label, font=self.font)
                        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                        draw.rectangle((x1, y1 - text_h - 10, x1 + text_w, y1), fill=color)
                        draw.text((x1, y1 - text_h - 5), label, font=self.font, fill=(255, 255, 255))
                        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    else:
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            print(f"检测到 {len(boxes)} 个边界框")
            self.update_person_count(len(boxes))
            self.results_history.extend(frame_results)
            if len(self.results_history) > 100:
                self.results_history = self.results_history[-100:]

            target_w, target_h = 640, 480
            h, w = original_frame.shape[:2]
            aspect_ratio = w / h
            target_aspect_ratio = target_w / target_h

            if aspect_ratio > target_aspect_ratio:
                new_w = target_w
                new_h = int(target_w / aspect_ratio)
            else:
                new_h = target_h
                new_w = int(target_h * aspect_ratio)

            resized_original = cv2.resize(original_frame, (new_w, new_h))
            resized_detect = cv2.resize(frame, (new_w, new_h))

            original_canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            detect_canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            original_canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_original
            detect_canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_detect

            original_canvas = cv2.cvtColor(original_canvas, cv2.COLOR_BGR2RGB)
            h, w, ch = original_canvas.shape
            bytes_per_line = ch * w
            q_image = QImage(original_canvas.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.original_window.setPixmap(QPixmap.fromImage(q_image))

            detect_canvas = cv2.cvtColor(detect_canvas, cv2.COLOR_BGR2RGB)
            h, w, ch = detect_canvas.shape
            bytes_per_line = ch * w
            q_image = QImage(detect_canvas.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.detect_window.setPixmap(QPixmap.fromImage(q_image))

        except Exception as e:
            print(f"处理帧失败: {e}")

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    login = LoginWindow()
    if login.exec_() == QDialog.Accepted:
        window = CrowdDensityMonitoringSystem()
        window.show()
        sys.exit(app.exec_())
    else:
        sys.exit()