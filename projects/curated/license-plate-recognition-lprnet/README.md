# LPRNet 车牌识别系统

## 项目简介

基于 PyTorch 的 LPRNet (License Plate Recognition Network) 端到端车牌字符识别。无需字符分割，直接从车牌图像输出字符序列，支持中文车牌字符集（31 省级缩写 + 字母数字）和 CTC 解码。

## 代码架构

### LPRNet 骨干 (`model/LPRNet.py`)

```python
class small_basic_block(nn.Module):
    """参数高效的卷积块 — 非对称卷积替代 3×3
    Conv1×1 → ReLU → Conv(3,1) → ReLU → Conv(1,3) → ReLU → Conv1×1
    参数量减少 ~33%，感受野等价于 3×3
    """
    def __init__(self, ch_in, ch_out):
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out//4, kernel_size=1), nn.ReLU(),
            nn.Conv2d(ch_out//4, ch_out//4, kernel_size=(3,1), padding=(1,0)),
            nn.ReLU(),
            nn.Conv2d(ch_out//4, ch_out//4, kernel_size=(1,3), padding=(0,1)),
            nn.ReLU(),
            nn.Conv2d(ch_out//4, ch_out, kernel_size=1),
        )

class LPRNet(nn.Module):
    """
    参数:
      lpr_max_len:  最大字符长度 (如 8)
      phase:        'train' / 'test'
      class_num:    字符类别数 (约 68 = 31省级 + 10数字 + 26字母 + 1 blank)
      dropout_rate: Dropout 比率

    Backbone (23 层):
      Conv(3→64, 3×3) → BN → ReLU → MaxPool(1,3,3)
      → small_basic_block(64→128) → BN → ReLU → MaxPool(1,3,3, stride=2)
      → small_basic_block(64→256) → BN → ReLU
      → small_basic_block(256→256) → BN → ReLU → MaxPool(1,3,3, stride=4)
      → Dropout → Conv(256, 1×4) → BN → ReLU
      → Dropout → Conv(256→class_num, 13×1) → BN → ReLU

    跳跃连接收集: layers [2, 6, 13, 22] 出口特征
      → 自适应 AvgPool → L2 归一化 (f / mean(f²))
      → torch.cat 多尺度拼接 → 1×1 Conv 融合
      → torch.mean(dim=2) → logits (B, class_num, seq_len)
    """

    def forward(self, x):
        keep_features = []
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]:
                keep_features.append(x)
        # 多尺度特征 → AvgPool → L2 norm → cat → container → mean → logits
```

### CTC 解码

```python
# 训练: torch.nn.CTCLoss(logits, targets, input_lengths, target_lengths)
# 推理: Greedy Decode
#   argmax → 合并连续重复 → 去除 blank (class 0)
#   中文字符集: 皖沪津渝冀晋...藏川陕甘青... (约 31 省级缩写)
#   字母数字: 0-9, A-Z (除去 I/O 防混淆)
```

### 数据加载 (`data/load_data.py`)

```python
# 输入: 车牌图像 (RGB, 94×24 等比例) → Resize → ToTensor → Normalize
# 标签: 字符序列 (如 "皖A12345")
# DataLoader: batch_size 可配置, shuffle=True
```

### 训练与测试

```python
# train_LPRNet.py:
#   optimizer: Adam(lr=0.001)
#   loss: torch.nn.CTCLoss(blank=0, reduction='mean')
#   scheduler: ReduceLROnPlateau

# test_LPRNet.py:
#   指标: per-char accuracy, sequence accuracy
```

## 技术栈

| 类别 | 技术 |
|------|------|
| 模型 | LPRNet (轻量 CNN, 无 RNN), small_basic_block |
| 卷积优化 | 非对称卷积 (3×1 + 1×3) |
| 损失函数 | CTC Loss (Connectionist Temporal Classification) |
| 解码 | Greedy CTC Decode |
| 特征融合 | 4 层跳跃连接 + L2 归一化 + 1×1 Conv |
| 框架 | PyTorch |

## 运行方式

```bash
pip install torch numpy opencv-python pillow
python train_LPRNet.py    # 训练
python test_LPRNet.py     # 测试
```

## 关键文件

| 文件 | 说明 |
|------|------|
| `model/LPRNet.py` | LPRNet 模型 (small_basic_block + LPRNet + forward) |
| `data/load_data.py` | 数据加载与预处理 |
| `train_LPRNet.py` | 训练入口 (Adam + CTCLoss) |
| `test_LPRNet.py` | 测试与评估 |
