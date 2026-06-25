import sys
import cv2
import json
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QInputDialog
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint

class Annotator(QWidget):
    def __init__(self, image_path):
        super().__init__()
        self.setWindowTitle("图像标注工具")

        self.image_path = image_path
        self.original_img = cv2.imread(image_path)
        self.current_img = self.original_img.copy()
        self.drawing = False
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.rects = []  # [{'id': 'xxx', 'bbox': [xmin, ymin, xmax, ymax]}]

        # 图像显示控件
        self.label = QLabel(self)
        self.update_display()

        # 按钮
        self.btn_undo = QPushButton("撤回上一步")
        self.btn_save = QPushButton("保存标注")
        self.btn_undo.clicked.connect(self.undo)
        self.btn_save.clicked.connect(self.save)

        hbox = QHBoxLayout()
        hbox.addWidget(self.btn_undo)
        hbox.addWidget(self.btn_save)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addLayout(hbox)
        self.setLayout(layout)

        # 鼠标事件绑定
        self.label.mousePressEvent = self.mouse_press
        self.label.mouseMoveEvent = self.mouse_move
        self.label.mouseReleaseEvent = self.mouse_release

    def update_display(self):
        img = self.current_img.copy()
        for region in self.rects:
            xmin, ymin, xmax, ymax = region["bbox"]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(img, region["id"], (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        height, width, channel = img.shape
        bytes_per_line = 3 * width
        qimg = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(qimg))

    def mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = event.pos()

    def mouse_move(self, event):
        if self.drawing:
            self.end_point = event.pos()
            self.preview()

    def mouse_release(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            x1, y1 = self.start_point.x(), self.start_point.y()
            x2, y2 = self.end_point.x(), self.end_point.y()
            xmin, xmax = sorted([x1, x2])
            ymin, ymax = sorted([y1, y2])

            # 弹出编号输入框
            text, ok = QInputDialog.getText(self, "输入编号", "请输入编号：")
            if ok and text:
                self.rects.append({
                    "id": text,
                    "bbox": [xmin, ymin, xmax, ymax]
                })
                self.update_display()

    def preview(self):
        temp_img = self.current_img.copy()
        painter = QPainter()
        height, width, _ = temp_img.shape
        qimg = QImage(temp_img.data, width, height, 3 * width, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimg)
        painter.begin(pixmap)
        pen = QPen(Qt.red, 1, Qt.SolidLine)
        painter.setPen(pen)
        painter.drawRect(self.start_point.x(), self.start_point.y(),
                         self.end_point.x() - self.start_point.x(),
                         self.end_point.y() - self.start_point.y())
        painter.end()
        self.label.setPixmap(pixmap)

    def undo(self):
        if self.rects:
            removed = self.rects.pop()
            print(f"↩️ 已撤销：{removed['id']}")
            self.update_display()

    def save(self):
        json_path, _ = QFileDialog.getSaveFileName(self, "保存标注为 JSON", "regions.json", "JSON 文件 (*.json)")
        if json_path:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self.rects, f, indent=2)
            print(f"✅ 已保存：{json_path}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    annotator = Annotator("test.png")  # 替换为你的图像路径
    annotator.resize(800, 600)
    annotator.show()
    sys.exit(app.exec_())
