# 医学图像二分类与肺部注意力模型

## 项目简介

基于 Swin Transformer 的胸部 X 光片疾病分类。创新性地引入**肺部区域空间注意力**（抑制非肺背景）和**病灶通道注意力**（建模 5 类病灶响应模式），结合 DeepLabV3 肺部掩膜生成和 Albumentations 医学图像增强管线。

## 代码架构

### 双注意力机制 (`model.py`)

```python
class LungAttentionModule(nn.Module):
    """肺部区域空间注意力
    Forward: x(B,C,H,W) + lung_mask(B,1,H,W)
      → F.interpolate 对齐 mask 与特征图分辨率
      → Conv2d(C→1, 3×3) → Sigmoid → 空间权重
      → output = x * lung_mask * attention
      → 抑制非肺部区域 (背景、骨骼、设备标识)
    """

class LesionChannelAttention(nn.Module):
    """病灶类型通道注意力
    Args: in_channels, lesion_types=5 (正常/结节/浸润/空洞/积液)
    Forward: GAP → FC(C→C//2) → ReLU → FC(C//2→5) → Sigmoid
      → x * channel_weights (广播至空间维度)
    """
```

### Swin Transformer 主干

```python
from torchvision.models import swin_t  # Swin-Tiny, ImageNet 预训练
# 输入: 224×224 RGB 胸部 X 光片
# 特征输出: 多尺度特征图 → 分类头
# 损失: CrossEntropyLoss / BCEWithLogitsLoss (多标签)
```

### 数据预处理 (`preprocessing.py`)

```python
# Albumentations 医学增强:
#   Resize(224, 224)
#   RandomBrightnessContrast(p=0.5)
#   RandomGamma(p=0.3)
#   CLAHE(clip_limit=2.0, p=0.5)       # 对比度受限自适应直方图均衡
#   Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

# 肺部掩膜: DeepLabV3 语义分割 → 二值 mask
```

### 数据集划分 (`divide.py`)

```python
# NIH Chest X-ray 数据集 (BBox_List_2017.csv)
# 14 种疾病多标签分类
# 患者级分层采样 → Train/Val/Test (避免交叉泄漏)
```

### 评估 (`evaluate_and_plot.py`)

```python
# 指标: Accuracy, AUC-ROC (macro/micro), Precision, Recall, F1
# 曲线: ROC per-class, PR curve, Confusion Matrix heatmap
```

## 技术栈

| 类别 | 技术 |
|------|------|
| 骨干网络 | Swin Transformer (torchvision swin_t) |
| 注意力 | LungAttentionModule (空间) + LesionChannelAttention (通道) |
| 分割 | DeepLabV3 (肺部区域提取) |
| 数据增强 | Albumentations (CLAHE, Gamma, Brightness) |
| 数据集 | NIH Chest X-ray (112,120 张, 14 种疾病) |

## 运行方式

```bash
pip install torch torchvision albumentations numpy pandas matplotlib scikit-learn
python preprocessing.py        # 图像增强 + 肺部掩膜生成
python divide.py               # 患者级分层数据划分
python model.py                # Swin + 双注意力训练
python evaluate_and_plot.py    # 评估报告 + ROC/PR 曲线
```

## 关键文件

| 文件 | 说明 |
|------|------|
| `model.py` | Swin Transformer + LungAttention + LesionChannelAttention |
| `preprocessing.py` | Albumentations 增强 + DeepLabV3 掩膜生成 |
| `divide.py` | 患者级 Train/Val/Test 划分 |
| `evaluate_and_plot.py` | AUC-ROC, PR, Confusion Matrix |
| `BBox_List_2017.csv` | NIH 数据集标注文件 |
