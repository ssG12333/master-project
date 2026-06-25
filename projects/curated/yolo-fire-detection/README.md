# FIRE-YOLOv5 火灾检测与边缘部署系统

基于 YOLOv5 的火灾目标检测系统，集成 CBAM/SE 注意力模块、TensorRT 推理加速、串口通信和 BiFPN 特征金字塔改进，适配边缘设备 (Jetson 等) 的实时推理与外部设备联动。

**原始目录**: `FIRE-YOLOV5-master/`

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 基础模型 | YOLOv5 (Ultralytics, GPL-3.0) |
| 注意力机制 | CBAM (Channel+ Spatial Attention), SE (Squeeze-and-Excitation) |
| 推理加速 | TensorRT + pycuda |
| 串口通信 | pyserial (UART, 115200 baud) |
| 特征金字塔 | BiFPN (加权特征融合) |
| 训练框架 | PyTorch (SGD/Adam, AMP, DDP) |

---

## 检测目标类别

在 TensorRT 推理脚本中定义了 4 个检测类别:

| 索引 | 类别 | 说明 |
|------|------|------|
| 0 | `fire` | 火焰/火灾 |
| 1 | `smoke` | 烟雾 |
| 2 | `w` | 未知/预留 |
| 3 | `c` | 未知/预留 |

---

## 注意力模块集成

### 1. CBAM (Convolutional Block Attention Module)

文件: `add(CBAM)common.py`

CBAM 由通道注意力和空间注意力串联组成，插入在 YOLOv5 Head 的 P5/32-large 检测支路末端。

```python
class ChannelAttentionModule(nn.Module):
    def __init__(self, c1, reduction=16):
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Linear(c1, c1 // 16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(c1 // 16, c1))
        self.act = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(...))
        maxout = self.shared_MLP(self.max_pool(x).view(...))
        return self.act(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        self.conv2d = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.act = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        return self.act(self.conv2d(out))

class CBAM(nn.Module):
    def __init__(self, c1, c2):
        self.channel_attention = ChannelAttentionModule(c1)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x    # 通道加权
        out = self.spatial_attention(out) * out # 空间加权
        return out
```

**CBAM 在 YOLOv5 中的插入位置** (参考 `model-configs/add(CBAM).yaml`):

```yaml
# ... 标准 YOLOv5 Head ...
   [-1, 3, C3, [1024, False]],  # P5/32-large 支路
   [-1, 1, CBAM, [1024]],       # ← CBAM 插入在 Detect 之前
   [[17, 20, 24], 1, Detect, [nc, anchors]]
```

### 2. SE (Squeeze-and-Excitation)

文件: `add(SE)common.py`

SE 模块以 `seC3` (SE 增强的 CSP Bottleneck) 和 `seBottleneck` 的形式集成，替换标准的 `C3` 模块。

```python
class seBottleneck(nn.Module):
    # 标准 Bottleneck + SE 通道注意力
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c1 // 4, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // 4, c1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.cv1(x)                              # 1x1 卷积降维
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)               # 全局平均池化
        y = self.l2(self.relu(self.l1(y)))            # FC → ReLU → FC
        y = self.sig(y).view(b, c, 1, 1)             # Sigmoid 激活权重
        x = x * y.expand_as(x)                       # 通道加权
        return x + self.cv2(x) if self.add else ...

class seC3(nn.Module):
    # CSP Bottleneck with 3 convolutions + SE
    def __init__(self, c1, c2, n=1, ...):
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*[seBottleneck(...) for _ in range(n)])
```

**SE 模块注册** (`add(SE)Yolo.py`):

```python
if m in [Conv, ..., C3, seC3]:  # seC3 加入 parse_model 模块列表
    if c2 != no:
        c2 = make_divisible(c2 * gw, 8)
    if m in [BottleneckCSP, seC3]:
        args.insert(2, n)  # number of repeats
```

---

## BiFPN 特征金字塔 (yolov5-bifpn.yaml)

在标准 YOLOv5 PANet Head 基础上，修改 P4 层的特征融合方式。

```yaml
# 标准 PANet:
[[-1, 14], 1, Concat, [1]]    # 仅融合 Head P4 (layer 14)

# BiFPN 改进:
[[-1, 14, 6], 1, Concat, [1]] # 融合 Head P4 (14) + Backbone P4 (6)
```

`[[-1, 14, 6], 1, Concat, [1]]` 的含义:

| 参数 | 说明 |
|------|------|
| `-1` | 上一层输出（前一个 C3 模块的输出） |
| `14` | Head 中第 14 层（上一级 Upsample 后的特征） |
| `6`  | Backbone P4/16 层原始特征（第 6 层 C3 输出） |
| `Concat` | 三路特征在通道维度拼接 |
| `[1]` | 拼接维度为 1 (通道维) |

这种跨层跳跃连接使得 P4 检测层同时获得高层语义和底层细节信息，类似于 EfficientDet 中 BiFPN 的加权特征融合思路。

---

## TensorRT 推理加速

文件: `Accelerate the engine.py`

使用 TensorRT Python API + pycuda 实现硬件加速推理，专为 NVIDIA Jetson 平台优化。

### `YoLov5TRT` 类

```python
class YoLov5TRT(object):
    def __init__(self, engine_file_path):
        self.ctx = cuda.Device(0).make_context()       # CUDA 上下文
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        # 分配 host/device 缓冲区
        for binding in engine:
            size = trt.volume(shape) * engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
```

### 推理流水线 (`infer` 方法)

```python
def infer(self):
    categories = ["fire", "smoke", "w", "c"]
    cap = cv2.VideoCapture(0)              # 摄像头捕获

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)         # 水平翻转

        # 预处理: BGR→RGB → resize+letterbox → normalize → NCHW
        input_image, origin_h, origin_w = self.preprocess_image(frame)

        # CUDA 推理
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async(batch_size=1, bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)

        # 后处理: NMS → 坐标转换 → 绘制
        result_boxes, result_scores, result_classid = self.post_process(output[0:6001], ...)

        # 串口发送目标坐标
        for j in range(len(result_boxes)):
            plot_one_box(box, frame, label=f"{categories[int(result_classid[j])]}:{result_scores[j]:.2f}")
```

### 预处理 (`preprocess_image`)

```python
def preprocess_image(self, raw_bgr_image):
    # 1. BGR → RGB
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    # 2. 等比例缩放 + letterbox (padding 128)
    image = cv2.copyMakeBorder(image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    # 3. 归一化 [0,1]
    image /= 255.0
    # 4. HWC → NCHW
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
```

### 后处理流水线

```
output (原始 TensorRT 输出: [num, cx,cy,w,h,conf,cls_id, ...])
    │
    ▼
post_process():
    ├─ num = int(output[0])                              ← 检测框数量
    ├─ pred = reshape(output[1:], (-1, 6))[:num]         ← 解析为 [x,y,w,h,conf,cls]
    ├─ xywh2xyxy()                                       ← 坐标格式转换
    ├─ non_max_suppression(conf_thres=0.5, nms_thres=0)  ← NMS 去重
    └─ return result_boxes, result_scores, result_classid
```

---

## 串口通信

通过 UART 将检测到的目标中心坐标发送给外部 MCU/执行器。

### 硬件配置

| 参数 | 值 |
|------|-----|
| 端口 | `/dev/ttyTHS1` (Jetson TX1/TX2 串口) |
| 波特率 | 115200 |
| 数据位 | 8 |
| 校验位 | NONE |
| 停止位 | 1 |

### 通信协议 (`plot_one_box` 函数)

```python
uart.write(bytearray([255, 255]))       # 起始标识 FF FF
for watertank in xy_c:                  # xy_c = [center_x, center_y]
    x1 = watertank % 256                # 低字节
    x2 = watertank // 256               # 高字节
    uart.write(bytearray([x2]))
    uart.write(bytearray([x1]))
uart.write(bytearray([251]))            # 结束标识 FB
```

协议格式: `FF FF` + `[center_x_H][center_x_L][center_y_H][center_y_L]` × N 个目标 + `FB`

### 备用串口实现 (`serial communication.py`)

使用标准 YOLOv5 `detect.py` 架构的串口版本，支持额外的帧计数器:

```python
uart.write(bytearray([255, 255]))       # 起始标识
uart.write(bytearray([counter]))        # 当前帧目标计数
for watertank in xy8xy:                 # 中心坐标
    x1 = watertank % 256
    x2 = watertank // 256
    uart.write(bytearray([x2]))
    uart.write(bytearray([x1]))
uart.write(bytearray([251]))            # 结束标识
```

---

## 训练脚本 (train.py)

完整 YOLOv5 训练流水线，支持:

| 特性 | 参数 |
|------|------|
| 优化器 | SGD (默认) / Adam |
| 学习率调度 | Cosine Annealing / Linear |
| 混合精度 | AMP (Automatic Mixed Precision) |
| 多卡训练 | DDP (DistributedDataParallel) |
| EMA | 指数移动平均模型 |
| 超参数进化 | 遗传算法搜索最优超参 |
| 早停 | EarlyStopping (patience 参数) |
| 数据增强 | Mosaic, MixUp, Copy-Paste, HSV 扰动等 |

```bash
python train.py --data fire_data.yaml --cfg models/yolov5s.yaml --weights '' --batch-size 16 --epochs 300
```

---

## 项目文件结构

```
yolo-fire-detection/
├── Accelerate the engine.py        # TensorRT 推理 + 串口通信主程序
├── serial communication.py         # 标准 YOLOv5 推理 + 串口通信备选
├── add(CBAM)common.py              # CBAM 注意力模块 (ChannelAttention + SpatialAttention)
├── add(SE)common.py                # SE 注意力模块 (seC3, seBottleneck)
├── add(SE)Yolo.py                  # SE 模块在 parse_model 中的注册
├── train.py                        # YOLOv5 训练脚本 (含超参进化)
├── models/
│   ├── common.py                   # YOLOv5 标准模块 (Conv, Bottleneck, C3, SPPF...)
│   ├── yolo.py                     # Model 类 + Detect 层 + parse_model
│   └── hub/
│       ├── yolov5-bifpn.yaml       # BiFPN 特征金字塔改进配置
│       ├── yolov5-fpn.yaml         # 标准 FPN 配置
│       ├── yolov5-panet.yaml       # 标准 PANet 配置
│       ├── yolov5-p2.yaml          # P2 额外小目标检测层
│       ├── yolov5-p6.yaml          # P6 额外大目标检测层
│       ├── yolov5-p7.yaml          # P7 额外大目标检测层
│       ├── yolov3.yaml             # YOLOv3 对比配置
│       ├── yolov3-spp.yaml         # YOLOv3 SPP 对比配置
│       ├── yolov3-tiny.yaml        # YOLOv3 Tiny 对比配置
│       └── anchors.yaml            # 锚框配置参考
└── README.md
```

---

## 知识点提炼

- **CBAM 注意力**: 通道注意力 (avg_pool + max_pool → shared MLP) 与空间注意力 (通道维 mean + max → Conv2d 7x7) 串联，插入 YOLOv5 Head P5 支路
- **SE 注意力**: 以 `seC3` / `seBottleneck` 形式替换标准 `C3`，通过全局平均池化 + FC → ReLU → FC → Sigmoid 生成通道权重
- **BiFPN 改进**: P4 层三路融合 `[[-1, 14, 6]]`，结合 Backbone 原始特征和 Head 上采样特征，提升多尺度表达
- **TensorRT 部署**: 使用 `trt.Runtime` 反序列化 `.engine` 文件，pycuda 管理 host/device 内存，异步 CUDA stream 执行
- **预处理适配**: 等比例 letterbox 缩放 + 128 padding + [0,1] 归一化 + NCHW 排列，匹配 TensorRT 输入要求
- **串口协议**: `FF FF` 起始 → 坐标 (高字节+低字节) → `FB` 结束，115200 8N1，适配 Jetson `/dev/ttyTHS1`
- **NMS 实现**: 自定义 `non_max_suppression()` 使用 `bbox_iou` 和 `label_match` 双重过滤，支持 `conf_thres` 和 `nms_thres` 配置
