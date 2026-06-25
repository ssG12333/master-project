import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QTextEdit, QTableWidget, QTableWidgetItem,
                             QFileDialog, QComboBox, QDialog, QLineEdit, QFormLayout)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPalette, QBrush
from ultralytics import YOLO
import uuid


class LoginWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("登录")
        self.setFixedSize(300, 200)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        form_layout = QFormLayout()

        # 用户名输入
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("请输入用户名")
        form_layout.addRow("用户名:", self.username_input)

        # 密码输入
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("请输入密码")
        self.password_input.setEchoMode(QLineEdit.Password)
        form_layout.addRow("密码:", self.password_input)

        layout.addLayout(form_layout)

        # 登录按钮
        self.login_button = QPushButton("登录")
        self.login_button.clicked.connect(self.check_credentials)
        layout.addWidget(self.login_button)

        # 状态标签
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: red;")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

        # 设置窗口样式
        self.setStyleSheet("""
            QDialog {
                background-color: rgba(255, 255, 255, 200);
            }
            QLineEdit {
                padding: 5px;
                font-size: 14px;
            }
            QPushButton {
                padding: 10px;
                font-size: 14px;
                background-color: rgba(0, 123, 255, 200);
                color: white;
            }
            QPushButton:hover {
                background-color: rgba(0, 86, 179, 200);
            }
        """)

    def check_credentials(self):
        username = self.username_input.text()
        password = self.password_input.text()

        # 硬编码的凭据（仅用于演示）
        if username == "admin" and password == "password":
            self.accept()  # 关闭对话框并返回成功
        else:
            self.status_label.setText("用户名或密码错误")


class HelmetDetectionSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = YOLO("best.pt")  # 加载YOLO模型
        self.cap = None  # 摄像头或视频捕获
        self.timer = QTimer()  # 定时器用于帧更新
        self.timer.timeout.connect(self.update_frame)
        self.results_history = []  # 存储检测结果
        self.selected_target = None  # 当前选择的目标
        self.initUI()

    def initUI(self):
        # 设置主窗口
        self.setWindowTitle("安全帽检测预警系统")
        self.setFixedSize(1280, 720)

        # 主控件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 标题
        title_label = QLabel("安全帽检测预警系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px; color: white; background-color: rgba(0, 0, 0, 150);")
        main_layout.addWidget(title_label)

        # 主内容区域
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)

        # 检测窗口（左侧）
        self.detect_window = QLabel()
        self.detect_window.setFixedSize(640, 480)
        self.detect_window.setStyleSheet("border: 2px solid black; background-color: #000000;")
        self.detect_window.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(self.detect_window)

        # 预警信息区域（右侧）
        right_layout = QVBoxLayout()
        content_layout.addLayout(right_layout)

        # 预警信息
        self.alert_info = QTextEdit()
        self.alert_info.setFixedSize(320, 240)
        self.alert_info.setReadOnly(True)
        self.alert_info.setStyleSheet("font-size: 14px; background-color: rgba(255, 255, 255, 200);")
        self.update_alert_info(0, 0, 0)
        right_layout.addWidget(self.alert_info)

        # 选择目标区域
        target_select_layout = QHBoxLayout()
        right_layout.addLayout(target_select_layout)

        # 选择目标标签
        select_label = QLabel("选择目标序号:")
        select_label.setStyleSheet("font-size: 14px; color: white; background-color: rgba(0, 0, 0, 150);")
        target_select_layout.addWidget(select_label)

        # 选择目标下拉菜单
        self.target_combo = QComboBox()
        self.target_combo.setFixedSize(150, 40)
        self.target_combo.addItem("无")
        self.target_combo.setStyleSheet("background-color: rgba(255, 255, 255, 200);")
        self.target_combo.currentIndexChanged.connect(self.select_target)
        target_select_layout.addWidget(self.target_combo)

        # 目标信息窗口
        self.target_info = QTextEdit()
        self.target_info.setFixedSize(320, 160)
        self.target_info.setReadOnly(True)
        self.target_info.setStyleSheet("font-size: 14px; background-color: rgba(255, 255, 255, 200);")
        right_layout.addWidget(self.target_info)

        # 底部按钮区域
        button_layout = QHBoxLayout()
        main_layout.addLayout(button_layout)

        # 底部按钮
        buttons = [
            ("选择图片", self.select_image),
            ("选择视频", self.select_video),
            ("启动摄像头", self.start_camera),
            ("关闭摄像头", self.stop_camera)
        ]
        for text, func in buttons:
            btn = QPushButton(text)
            btn.setFixedSize(150, 40)
            btn.clicked.connect(func)
            btn.setStyleSheet("background-color: rgba(255, 255, 255, 200);")
            button_layout.addWidget(btn)

        # 信息分析按钮
        analysis_btn = QPushButton("查看信息分析")
        analysis_btn.setFixedSize(150, 40)
        analysis_btn.clicked.connect(self.show_analysis)
        analysis_btn.setStyleSheet("background-color: rgba(255, 255, 255, 200);")
        button_layout.addWidget(analysis_btn)

        # 信息分析窗口（初始隐藏）
        self.analysis_window = QTableWidget()
        self.analysis_window.setColumnCount(4)
        self.analysis_window.setHorizontalHeaderLabels(["序号", "是否带安全帽", "置信度", "坐标位置"])
        self.analysis_window.setFixedSize(960, 200)
        self.analysis_window.setColumnWidth(0, 100)
        self.analysis_window.setColumnWidth(1, 200)
        self.analysis_window.setColumnWidth(2, 200)
        self.analysis_window.setColumnWidth(3, 460)
        self.analysis_window.setStyleSheet("background-color: rgba(255, 255, 255, 200);")
        self.analysis_window.hide()
        main_layout.addWidget(self.analysis_window)

        # 设置背景图片
        background = QPixmap("background.jpg")  # 假设图片名为background.jpg
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(background.scaled(self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)))
        self.setPalette(palette)

    def update_alert_info(self, total, with_helmet, without_helmet):
        text = f"总人数: {total}\n戴头盔人数: {with_helmet}\n没戴头盔人数: {without_helmet}"
        self.alert_info.setText(text)

    def update_target_info(self, coords, has_helmet, confidence):
        text = f"目标坐标: {coords}\n是否佩戴安全帽: {'是' if has_helmet else '否'}\n置信度: {confidence:.2f}"
        self.target_info.setText(text)

    def update_analysis_table(self):
        self.analysis_window.setRowCount(len(self.results_history))
        for i, (has_helmet, confidence, coords) in enumerate(self.results_history):
            self.analysis_window.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.analysis_window.setItem(i, 1, QTableWidgetItem("是" if has_helmet else "否"))
            self.analysis_window.setItem(i, 2, QTableWidgetItem(f"{confidence:.2f}"))
            self.analysis_window.setItem(i, 3, QTableWidgetItem(str(coords)))

    def update_target_combo(self):
        self.target_combo.blockSignals(True)
        current_index = self.target_combo.currentIndex()
        self.target_combo.clear()
        self.target_combo.addItem("无")
        for i in range(len(self.results_history)):
            self.target_combo.addItem(str(i + 1))
        if current_index != -1 and current_index <= len(self.results_history):
            self.target_combo.setCurrentIndex(current_index)
        self.target_combo.blockSignals(False)

    def select_image(self):
        self.stop_camera()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            img = cv2.imread(file_name)
            if img is not None:
                self.process_frame(img)

    def select_video(self):
        self.stop_camera()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi)")
        if file_name:
            self.cap = cv2.VideoCapture(file_name)
            self.timer.start(30)

    def start_camera(self):
        self.stop_camera()
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.timer.start(30)

    def stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.timer.stop()
        self.detect_window.clear()
        self.update_alert_info(0, 0, 0)
        self.target_info.clear()
        self.results_history.clear()
        self.update_target_combo()

    def select_target(self):
        selected_index = self.target_combo.currentIndex()
        if selected_index == 0:
            self.selected_target = None
            self.target_info.clear()
        else:
            target_index = selected_index - 1
            if 0 <= target_index < len(self.results_history):
                self.selected_target = self.results_history[target_index]
                has_helmet, confidence, coords = self.selected_target
                self.update_target_info(coords, has_helmet, confidence)

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.process_frame(frame)

    def process_frame(self, frame):
        results = self.model(frame)
        total, with_helmet, without_helmet = 0, 0, 0
        frame_results = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                total += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0].item()
                cls = int(box.cls[0])
                has_helmet = cls == 0
                coords = (x1, y1, x2, y2)
                frame_results.append((has_helmet, confidence, coords))

                if has_helmet:
                    with_helmet += 1
                    color = (0, 255, 0)
                else:
                    without_helmet += 1
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{'Helmet' if has_helmet else 'No Helmet'} {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        self.update_alert_info(total, with_helmet, without_helmet)
        self.results_history.extend(frame_results)
        if len(self.results_history) > 100:
            self.results_history = self.results_history[-100:]

        self.update_target_combo()

        if self.selected_target:
            selected_index = self.target_combo.currentIndex()
            if selected_index > 0 and (selected_index - 1) < len(self.results_history):
                self.selected_target = self.results_history[selected_index - 1]
                self.update_target_info(*self.selected_target)
            else:
                self.selected_target = None
                self.target_info.clear()

        self.update_analysis_table()

        frame = cv2.resize(frame, (640, 480))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.detect_window.setPixmap(QPixmap.fromImage(q_image))

    def show_analysis(self):
        self.analysis_window.setVisible(not self.analysis_window.isVisible())

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 显示登录窗口
    login = LoginWindow()
    if login.exec_() == QDialog.Accepted:
        # 登录成功，显示主窗口
        window = HelmetDetectionSystem()
        window.show()
        sys.exit(app.exec_())
    else:
        # 登录失败，退出应用
        sys.exit()