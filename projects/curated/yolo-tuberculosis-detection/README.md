# YOLO 肺结核与胸片疾病检测系统

基于 YOLOv8 (Ultralytics) 和 PyQt5 的胸片 (Chest X-ray) 疾病检测桌面应用。系统可识别肺结核、细菌性肺炎、病毒性肺炎等疾病类型，提供中文标签映射和检测结果可视化。

**原始目录**: `肺结核yolo/`

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 目标检测模型 | YOLOv8 / YOLOv5+CBAM (双架构探索) |
| 图形界面 | PyQt5 (QMainWindow, QDialog) |
| 图像处理 | OpenCV, PIL (ImageDraw) |
| 医疗影像 | 胸部 X 光片 (Chest X-ray) |
| 训练框架 | Ultralytics YOLO (PyTorch) |

---

## 应用程序架构 (app.py)

### `LoginWindow(QDialog)`

与人群检测项目相同的登录验证弹窗，固定尺寸 `300x200`，凭据硬编码为 `admin` / `password`。

### `TuberculosisDetectionSystem(QMainWindow)`

主检测窗口，固定尺寸 `1200x800`。

#### 检测标签映射

```python
self.label_map = {
    'Pneumonia Bacteria': '细菌性肺炎',
    'Pneumonia Virus':    '病毒性肺炎',
    'Sick':               '患病',
    'healthy':            '健康',
    'tuberculosis':       '肺结核'
}
```

#### 模型初始化

```python
self.model = YOLO("best.pt")  # 加载训练好的胸片检测权重
```

支持 5 个检测类别:
| 原始标签 | 中文显示 | 颜色 |
|----------|----------|------|
| `tuberculosis` | 肺结核 | 蓝色 (255, 0, 0) |
| `Pneumonia Bacteria` | 细菌性肺炎 | 绿色 (0, 255, 0) |
| `Pneumonia Virus` | 病毒性肺炎 | 绿色 (0, 255, 0) |
| `Sick` | 患病 | 绿色 (0, 255, 0) |
| `healthy` | 健康 | 绿色 (0, 255, 0) |

#### 界面布局 (`initUI`)

- 顶部标题栏: "Chest X-ray诊断系统" (SimHei 24pt Bold)
- 中间区域:
  - 左栏: **检测结果** 显示面板 (640x480)，标注检测框
  - 右栏: 疾病类型状态标签 + 置信度只读输入框
- 底部: "选择图片" 按钮

#### 核心方法

##### `process_frame(self, frame)`

```python
def process_frame(self, frame):
    results = self.model(frame)
    disease_detected = "None"
    max_confidence = 0.0

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            cls = int(box.cls[0])
            disease = self.model.names[cls]

            # 取置信度最高的疾病作为诊断结果
            if confidence > max_confidence:
                max_confidence = confidence
                disease_detected = disease

            # 肺结核用蓝色框，其他用绿色框
            color = (255, 0, 0) if disease.lower() == "tuberculosis" else (0, 255, 0)
            label = f"{self.label_map.get(disease, disease)} {confidence:.2f}"

            # 使用 PIL ImageDraw 绘制中文标签
```

##### `update_disease_status(self, disease, confidence)`

```python
def update_disease_status(self, disease, confidence):
    self.disease_detected = self.label_map.get(disease, "无")
    self.disease_label.setText(f"检测到疾病: {self.disease_detected}")
    self.confidence_input.setText(f"{confidence:.2f}" if confidence else "0.00")
```

##### `select_image(self)`

```python
file_name, _ = QFileDialog.getOpenFileName(
    self, "选择图片", "", "图片 (*.png *.jpg *.jpeg)")
```

仅支持单张图片输入（无视频/摄像头模式）。

#### 应用入口

```python
if __name__ == '__main__':
    app = QApplication(sys.argv)
    login = LoginWindow()
    if login.exec_() == QDialog.Accepted:
        window = TuberculosisDetectionSystem()
        window.show()
        sys.exit(app.exec_())
```

---

## 模型架构 (model-configs)

### YOLOv8 配置 (`yolov8.yaml`)

标准 Ultralytics YOLOv8 检测模型，P3-P5 输出。

| 参数 | 值 |
|------|-----|
| `nc` (类别数) | 80 |
| Backbone | 9 层: Conv → Conv → C2f×3 → Conv → C2f×6 → Conv → C2f×6 → Conv → C2f×3 → SPPF |
| Head | FPN+PAN 结构，3 个 Detect 头 (P3/8-small, P4/16-medium, P5/32-large) |
| 缩放变体 | n/s/m/l/x (depth: 0.33~1.00, width: 0.25~1.25) |

训练命令 (demo.py):

```python
from ultralytics import YOLO
model = YOLO('yolov8l_fusion_transformerx3_hsi.yaml')
model.train(data=r'data/v1/data.yaml', workers=0, device='cuda', amp=False)
```

使用了自定义融合 Transformer 的 YOLOv8 变体配置 `yolov8l_fusion_transformerx3_hsi.yaml`。

### YOLOv5 + CBAM 配置 (`add(CBAM).yaml`)

该配置为 YOLOv5 v6.0 架构，在 Head 末端引入 CBAM 注意力模块。

| 参数 | 值 |
|------|-----|
| `nc` (类别数) | 80 |
| `depth_multiple` | 0.33 |
| `width_multiple` | 0.50 |
| Anchors | P3: [10,13, 16,30, 33,23]; P4: [30,61, 62,45, 59,119]; P5: [116,90, 156,198, 373,326] |

**CBAM 插入位置**: Head 的 P5/32-large 支路末尾，第 24 层:

```yaml
# YOLOv5 v6.0 head (局部)
head:
  [[-1, 1, Conv, [512, 1, 1]],
   ...                     # Upsample + Concat + C3 (P3/P4/P5 支路)
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)
   [-1, 1, CBAM, [1024]],       # 24 ← CBAM 插入至此
   [[17, 20, 24], 1, Detect, [nc, anchors]],  # Detect 引用 24
  ]
```

CBAM 模块通过 `parse_model()` 的 `eval(m)` 字符串解析从 `models/common.py` 导入，实现即插即用。

---

## CBAM 注意力机制 (Convolutional Block Attention Module)

CBAM 通过串联通道注意力和空间注意力，对特征图进行自适应精炼。

```
输入特征 x
    │
    ▼
┌─────────────────────────┐
│  ChannelAttentionModule │  ← 通道维度加权
│  ├─ AdaptiveAvgPool2d   │
│  ├─ AdaptiveMaxPool2d   │
│  └─ shared_MLP (Sigmoid)│
└─────────┬───────────────┘
          ▼
   out = channel_attention(x) * x
          │
          ▼
┌─────────────────────────┐
│  SpatialAttentionModule │  ← 空间维度加权
│  ├─ mean(dim=1)         │
│  ├─ max(dim=1)          │
│  └─ Conv2d(2→1, k=7)   │
│     → Sigmoid           │
└─────────┬───────────────┘
          ▼
   out = spatial_attention(out) * out
```

```python
class ChannelAttentionModule(nn.Module):
    def __init__(self, c1, reduction=16):
        mid_channel = c1 // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Linear(c1, mid_channel),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(mid_channel, c1))
        self.act = nn.Sigmoid()

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        self.conv2d = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.act = nn.Sigmoid()
```

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

# 日志 & 绘图
tensorboard>=2.13.0
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
yolo-tuberculosis-detection/
├── app.py                         # PyQt5 主程序
├── demo.py                        # 训练脚本示例
├── setup.py                       # Ultralytics 包配置
├── requirements.txt               # Python 依赖
├── best.pt                        # 训练好的权重 (需单独下载)
├── model-configs/
│   ├── yolov8.yaml                # 标准 YOLOv8 配置 (80类, P3-P5)
│   └── add(CBAM).yaml             # YOLOv5 v6.0 + CBAM 注意力配置
├── background.jpg                 # 界面背景图
└── README.md
```

---

## 知识点提炼

- **医学影像目标检测**: 针对胸部 X 光片进行疾病检测，涵盖 5 类标签（肺结核、细菌性肺炎、病毒性肺炎、患病、健康）
- **双架构探索**: 项目同时使用了 YOLOv8 (Ultralytics) 和 YOLOv5+CBAM 两种架构进行实验，分别对应 `yolov8.yaml` 和 `add(CBAM).yaml`
- **CBAM 注意力**: 在 YOLOv5 P5/32-large 检测支路末端插入 CBAM 模块，通过通道注意力 + 空间注意力双维度加权提升检测精度
- **中文医学标签**: 通过 `label_map` 字典将英文检测结果映射为中文显示（"tuberculosis" → "肺结核"）
- **疾病优先级高亮**: 肺结核检测框使用蓝色 (BGR 255,0,0) 区别于其他疾病的绿色，提供视觉显著性提示
- **置信度取最大值**: 在多检测框中选取置信度最高的疾病类别作为最终诊断结果
