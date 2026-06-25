import sys
import json
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt


class AnnotViewer(QWidget):
    def __init__(self, image_path, json_path):
        super().__init__()
        self.setWindowTitle("标注查看器 - 点击生成编号框")

        self.image_path = image_path
        self.annotations = self.load_annotations(json_path)
        self.original_img = cv2.imread(image_path)
        self.scale = 1.0  # 缩放因子
        self.shown_ids = []  # 已点击过的编号 ID

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.update_display()
        self.label.mousePressEvent = self.mouse_click
        self.label.wheelEvent = self.wheel_event

    def load_annotations(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def update_display(self):
        # 根据缩放因子调整图像大小
        img_resized = cv2.resize(self.original_img,
                                 (int(self.original_img.shape[1] * self.scale),
                                  int(self.original_img.shape[0] * self.scale)))

        for region in self.annotations:
            if region["id"] in self.shown_ids:
                xmin, ymin, xmax, ymax = region["bbox"]
                # 根据缩放因子调整框的位置和大小
                xmin, ymin, xmax, ymax = [int(x * self.scale) for x in (xmin, ymin, xmax, ymax)]
                # 绘制矩形框
                cv2.rectangle(img_resized, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                # 在框内标注编号
                text_x = xmin + 5  # 在框内左上角
                text_y = ymin + (ymax - ymin) // 2  # 在框内垂直居中
                cv2.putText(img_resized, region["id"], (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        h, w, _ = img_resized.shape
        qimg = QImage(img_resized.data, w, h, 3 * w, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(qimg))

    def mouse_click(self, event):
        x = int(event.pos().x() / self.scale)
        y = int(event.pos().y() / self.scale)
        for region in self.annotations:
            xmin, ymin, xmax, ymax = region["bbox"]
            if xmin <= x <= xmax and ymin <= y <= ymax:
                if region["id"] not in self.shown_ids:
                    self.shown_ids.append(region["id"])
                    print(f"🎯 显示编号：{region['id']}")
                self.update_display()
                return
        print("⚠️ 未命中任何框")

    def wheel_event(self, event):
        angle = event.angleDelta().y()
        factor = 1.25 if angle > 0 else 0.8
        self.scale = max(0.2, min(5.0, self.scale * factor))
        print(f"🔍 缩放至 {self.scale:.2f}x")
        self.update_display()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AnnotViewer("test.png", "regions.json")  # 替换成你的图像和json
    window.resize(1000, 800)
    window.show()
    sys.exit(app.exec_())
