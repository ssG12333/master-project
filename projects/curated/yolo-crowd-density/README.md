# YOLO 密集人群检测与密度监测系统

基于 YOLOv8 (Ultralytics) 和 PyQt5 的桌面端密集人群检测与密度监测系统。支持图片、视频文件和摄像头实时流三种输入模式，实现人群目标检测、人数统计、密度计算等功能，适用于人流监控和安全预警场景。

**原始目录**: `密集人群yolo/`

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 目标检测模型 | YOLOv8 (Ultralytics) |
| 图形界面 | PyQt5 (QMainWindow, QDialog) |
| 图像处理 | OpenCV (cv2), PIL (ImageDraw) |
| 数值计算 | NumPy |
| 训练框架 | Ultralytics YOLO (PyTorch) |

---

## 应用程序架构

### 类结构 (app.py)

#### `LoginWindow(QDialog)`

登录验证弹窗，固定尺寸 `300x200`。

| 方法 | 功能 |
|------|------|
| `initUI()` | 构建用户名/密码输入表单 + 登录按钮 |
| `check_credentials()` | 校验凭据 (硬编码 `admin` / `password`)，成功调用 `self.accept()` |

凭据通过 `QLineEdit` 输入，密码使用 `EchoMode.Password` 掩码显示。

#### `CrowdDensityMonitoringSystem(QMainWindow)`

主检测窗口，固定尺寸 `1920x1000`。

```python
class CrowdDensityMonitoringSystem(QMainWindow):
    def __init__(self):
        self.model = YOLO("best.pt")          # 加载训练好的 YOLOv8 权重
        self.cap = None                       # cv2.VideoCapture 实例
        self.timer = QTimer()                 # 定时驱动视频帧读取
        self.timer.timeout.connect(self.update_frame)
        self.results_history = []             # 检测结果历史（上限 100 条）
        self.person_count = 0
        # 多路径中文字体回退加载策略
        self.font = None                      # PIL.ImageFont 用于中文标签
```

**界面布局 (`initUI`)**:

- 顶部标题栏: "人群密度监测系统" (Arial 28pt Bold, 渐变蓝背景)
- 中间区域 (水平分栏):
  - 左栏: **原图** 显示面板 (640x480)
  - 中栏: **检测情况** 显示面板 (640x480)，带检测框标注
  - 右栏: 人数统计标签 + 三个输入框（区域面积、当前密度、当前人数）
- 底部按钮组: 选择图片、选择视频、启动摄像头、关闭摄像头
- 背景图片支持 (自动加载 `background.jpg`)

### 检测流水线

```
用户输入 (图片/视频/摄像头)
        |
        v
  select_image() / select_video() / start_camera()
        |
        v
  process_frame(frame)      ← 核心检测方法
        |
        +---> self.model(frame)          ← YOLOv8 推理
        +---> 遍历 result.boxes          ← 提取 xyxy, confidence, class
        +---> cv2.rectangle 绘制检测框   ← OpenCV 绘制
        +---> ImageDraw.text 中文标签    ← PIL 字体回退机制
        +---> self.update_person_count() ← 更新计数 & 密度计算
        +---> QImage → QPixmap 渲染到界面
```

### 关键方法详解

#### `process_frame(self, frame)`

```python
def process_frame(self, frame):
    # 1. 大图缩放: 若 >1080p 则缩放到 1280x720
    if frame.shape[0] > 1080 or frame.shape[1] > 1920:
        frame = cv2.resize(frame, (1280, 720))

    # 2. YOLO 推理
    results = self.model(frame)
    boxes = []
    for result in results:
        boxes.extend(result.boxes)
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            cls = int(box.cls[0])
            state = self.model.names[cls]          # 类别名称

            # 3. 绘制检测框 + 置信度标签
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # 若中文字体可用则用 PIL.ImageDraw 绘制
            # 否则用 cv2.putText (仅支持 ASCII)

    # 4. 更新人数统计
    self.update_person_count(len(boxes))

    # 5. 保持宽高比缩放到 640x480 canvas 并渲染
```

#### `update_person_count(self, count)`

```python
def update_person_count(self, count):
    self.person_count = count
    self.person_count_label.setText(f"检测到人数: {self.person_count}")
    self.population_input.setText(str(self.person_count))
    # 密度 = 人数 / 面积
    if area > 0:
        density = self.person_count / area
        self.density_input.setText(f"{density:.2f}")
```

#### 输入源切换

| 方法 | 功能 | 实现细节 |
|------|------|----------|
| `select_image()` | 选择单张图片 | `QFileDialog.getOpenFileName` 过滤 `*.png *.jpg *.jpeg` → `cv2.imread` |
| `select_video()` | 选择视频文件 | `QFileDialog.getOpenFileName` 过滤 `*.mp4 *.avi` → `cv2.VideoCapture` → `timer.start(30)` |
| `start_camera()` | 启动摄像头 | `cv2.VideoCapture(0)` → `timer.start(30)` |
| `stop_camera()` | 停止输入 | `cap.release()`, `timer.stop()`, 清空界面和结果历史 |

`QTimer` 每 30ms 触发一次 `update_frame()`，实现视频流实时处理（约 33 FPS）。

---

## 模型与训练 (demo.py)

```python
from ultralytics import YOLO

# 从 YAML 配置和预训练权重加载模型
model = YOLO('yolov8.yaml').load('yolov8n.pt')

# 启动训练
model.train(
    data=r'data/v3/data.yaml',
    workers=0,
    device='cuda',
    amp=False
)
```

训练数据采用 Roboflow YOLO 格式，数据集名称为 `people_counterv0 - v1 2023-05-18 6-22pm`。

---

## 依赖 (requirements.txt)

```
# 核心
hydra-core>=1.2.0
matplotlib>=3.3.0
numpy>=1.22.2
opencv-python>=4.6.0
pillow>=7.1.2
pyyaml>=5.3.1
requests>=2.23.0
scipy>=1.4.1
torch>=1.8.0
torchvision>=0.9.0
tqdm>=4.64.0

# 日志
tensorboard>=2.13.0

# 绘图
pandas>=1.1.4
seaborn>=0.11.0

# 工具
psutil
py-cpuinfo
thop>=0.1.1
GitPython>=3.1.24
```

---

## 项目文件结构

```
yolo-crowd-density/
├── app.py                 # PyQt5 主程序 (LoginWindow + CrowdDensityMonitoringSystem)
├── demo.py                # 训练脚本示例
├── setup.py               # Ultralytics 包配置（AGPL-3.0）
├── requirements.txt       # Python 依赖
├── best.pt                # 训练好的 YOLOv8 权重 (需单独下载)
├── background.jpg         # 界面背景图
└── README.md
```

---

## 知识点提炼

- **YOLOv8 推理**: 使用 `ultralytics.YOLO` 加载 `.pt` 权重，`model(frame)` 返回 `Results` 对象，通过 `result.boxes.xyxy/conf/cls` 提取检测信息
- **中文文本渲染**: 多路径回退加载中文字体 (`SimHei.ttf` → `NotoSansCJK` → `PingFang.ttc`)，通过 PIL `ImageDraw.text()` 绘制，OpenCV 不支持原生中文
- **视频流处理**: `QTimer` + `cv2.VideoCapture` 驱动帧读取循环，`timer.start(30)` 设定约 33 FPS
- **密度计算**: `密度 = 人数 / 区域面积`，通过 `update_person_count()` 联动更新界面
- **等比例缩放**: 保持原始宽高比缩放到固定 canvas (640x480)，避免图像拉伸变形
