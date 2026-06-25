import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QFileDialog, QDialog, QLineEdit, QFormLayout)
from PyQt5.QtCore import Qt
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

class TuberculosisDetectionSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.label_map = {
            'Pneumonia Bacteria': '细菌性肺炎',
            'Pneumonia Virus': '病毒性肺炎',
            'Sick': '患病',
            'healthy': '健康',
            'tuberculosis': '肺结核'
        }
        try:
            self.model = YOLO("best.pt")
        except Exception as e:
            print(f"加载 YOLO 模型失败: {e}")
            sys.exit(1)
        self.results_history = []
        self.disease_detected = "无"
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
        self.setWindowTitle("Chest X-ray诊断系统")
        self.setFixedSize(1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)

        title_label = QLabel("Chest X-ray诊断系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("SimHei", 24, QFont.Bold))
        title_label.setStyleSheet("""
            color: white;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #007BFF, stop:1 #00BFFF);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        """)
        main_layout.addWidget(title_label)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)
        main_layout.addLayout(content_layout)

        detection_layout = QVBoxLayout()
        detection_title = QLabel("检测结果")
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
        content_layout.addLayout(detection_layout)

        right_layout = QVBoxLayout()
        right_layout.setAlignment(Qt.AlignTop)
        content_layout.addLayout(right_layout)

        self.disease_label = QLabel("检测到疾病: 无")
        self.disease_label.setFixedSize(300, 80)
        self.disease_label.setAlignment(Qt.AlignCenter)
        self.disease_label.setStyleSheet("""
            font-size: 20px;
            color: white;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #dc3545, stop:1 #ff6b6b);
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        """)
        right_layout.addWidget(self.disease_label)
        right_layout.addSpacing(20)

        input_form_layout = QFormLayout()
        self.confidence_input = QLineEdit()
        self.confidence_input.setPlaceholderText("检测置信度")
        self.confidence_input.setFixedSize(300, 40)
        self.confidence_input.setReadOnly(True)
        self.confidence_input.setStyleSheet("""
            padding: 8px;
            font-size: 14px;
            border-radius: 5px;
            border: 1px solid #ccc;
            background-color: #f0f0f0;
        """)
        input_form_layout.addRow("置信度:", self.confidence_input)

        right_layout.addLayout(input_form_layout)
        right_layout.addStretch()

        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        main_layout.addSpacing(20)
        main_layout.addLayout(button_layout)

        select_btn = QPushButton("选择图片")
        select_btn.setFixedSize(160, 50)
        select_btn.clicked.connect(self.select_image)
        select_btn.setStyleSheet("""
            background-color: #007BFF;
            color: white;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            padding: 10px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        """)
        select_btn.setStyleSheet(select_btn.styleSheet() + """
            QPushButton:hover {
                background-color: #0056b3;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            }
            QPushButton:pressed {
                background-color: #003d80;
            }
        """)
        button_layout.addWidget(select_btn)
        button_layout.addStretch()

        try:
            background = QPixmap("background.jpg")
            palette = QPalette()
            palette.setBrush(QPalette.Background, QBrush(background.scaled(self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)))
            self.setPalette(palette)
        except Exception as e:
            print(f"加载背景图片失败: {e}")
            self.setStyleSheet("background-color: #f0f2f5;")

    def update_disease_status(self, disease, confidence):
        self.disease_detected = self.label_map.get(disease, "无")
        self.disease_label.setText(f"检测到疾病: {self.disease_detected}")
        self.confidence_input.setText(f"{confidence:.2f}" if confidence else "0.00")

    def select_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片 (*.png *.jpg *.jpeg)")
        if file_name:
            img = cv2.imread(file_name)
            if img is not None:
                self.process_frame(img)
            else:
                print(f"无法加载图片: {file_name}")

    def process_frame(self, frame):
        try:
            if frame.shape[0] > 1080 or frame.shape[1] > 1920:
                frame = cv2.resize(frame, (1280, 720))
            results = self.model(frame)
            frame_results = []
            boxes = []

            disease_detected = "None"
            max_confidence = 0.0

            for result in results:
                boxes.extend(result.boxes)
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0].item()
                    cls = int(box.cls[0])
                    disease = self.model.names[cls]
                    coords = (x1, y1, x2, y2)
                    frame_results.append((disease, confidence, coords))

                    if confidence > max_confidence:
                        max_confidence = confidence
                        disease_detected = disease

                    color = (255, 0, 0) if disease.lower() == "tuberculosis" else (0, 255, 0)
                    label = f"{self.label_map.get(disease, disease)} {confidence:.2f}"
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
            self.update_disease_status(disease_detected, max_confidence)
            self.results_history.extend(frame_results)
            if len(self.results_history) > 100:
                self.results_history = self.results_history[-100:]

            target_w, target_h = 640, 480
            h, w = frame.shape[:2]
            aspect_ratio = w / h
            target_aspect_ratio = target_w / target_h

            if aspect_ratio > target_aspect_ratio:
                new_w = target_w
                new_h = int(target_w / aspect_ratio)
            else:
                new_h = target_h
                new_w = int(target_h * aspect_ratio)

            resized_detect = cv2.resize(frame, (new_w, new_h))
            detect_canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            detect_canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_detect

            detect_canvas = cv2.cvtColor(detect_canvas, cv2.COLOR_BGR2RGB)
            h, w, ch = detect_canvas.shape
            bytes_per_line = ch * w
            q_image = QImage(detect_canvas.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.detect_window.setPixmap(QPixmap.fromImage(q_image))

        except Exception as e:
            print(f"处理帧失败: {e}")

    def closeEvent(self, event):
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    login = LoginWindow()
    if login.exec_() == QDialog.Accepted:
        window = TuberculosisDetectionSystem()
        window.show()
        sys.exit(app.exec_())
    else:
        sys.exit()