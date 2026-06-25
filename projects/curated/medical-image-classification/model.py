import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_t
from typing import Optional, Tuple


class LungAttentionModule(nn.Module):
    """肺部区域空间注意力模块"""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, lung_mask: torch.Tensor) -> torch.Tensor:
        # 确保 lung_mask 具有正确的维度
        if lung_mask.dim() == 2:  # 如果 lung_mask 是 (N, 1) 形状
            lung_mask = lung_mask.unsqueeze(-1).unsqueeze(-1)  # 增加两个维度变成 (N, 1, 1, 1)
        """
        Args:
            x: 输入特征图 (B, C, H, W)
            lung_mask: 肺部掩膜 (B, 1, H, W)
        """
        print("\n===== 空间注意力模块 =====")
        print(f"输入特征图形状: {x.shape}")
        print(f"原始肺部掩膜形状: {lung_mask.shape}")

        # 调整掩膜尺寸匹配特征图
        lung_mask = F.interpolate(lung_mask, size=x.shape[2:], mode="bilinear")
        print(f"调整后肺部掩膜形状: {lung_mask.shape}")

        # 生成空间注意力权重
        attn = self.conv(x)
        attn = self.sigmoid(attn)
        print(f"空间注意力权重形状: {attn.shape}")

        # 融合肺部掩膜与注意力
        x = x * lung_mask * attn  # 抑制非肺部区域
        print(f"输出特征图形状: {x.shape}")

        return x * lung_mask


class LesionChannelAttention(nn.Module):
    """病灶类型通道注意力模块"""

    def __init__(self, in_channels: int, lesion_types: int = 5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, lesion_types),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征图 (B, C, H, W)
        Returns:
            带通道注意力的特征图 (B, C, H, W)
        """
        print("\n===== 通道注意力模块 =====")
        print(f"输入特征图形状: {x.shape}")

        # 全局平均池化获取图像级特征 (B, C)
        global_features = x.mean(dim=(2, 3))
        print(f"全局平均池化后形状: {global_features.shape}")

        # 病灶类型编码 (B, lesion_types)
        lesion_emb = self.fc(global_features)
        print(f"病灶类型编码形状: {lesion_emb.shape}")

        # 扩展维度以匹配特征图通道
        channel_attn = lesion_emb.unsqueeze(-1).unsqueeze(-1)  # (B, lesion_types, 1, 1)
        channel_attn = channel_attn.expand(-1, -1, x.shape[2], x.shape[3])  # (B, lesion_types, H, W)

        # 简化实现：对lesion_types个通道取平均
        channel_attn = channel_attn.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        x = x * channel_attn
        print(f"输出特征图形状: {x.shape}")

        return x


class TBTransformerModel(nn.Module):
    """基于预训练Swin Transformer的肺结核诊断模型"""

    def __init__(
            self,
            num_classes: int = 2,  # 分类任务类别数（二分类为2）
            use_attention: bool = True,
            pretrained: bool = True,
            dropout_rate: float = 0.5  # Dropout率
    ):
        super().__init__()
        self.use_attention = use_attention

        print("\n===== 模型初始化 =====")
        # 加载预训练的Swin-Tiny模型（输出维度768）
        self.backbone = swin_t(weights="DEFAULT" if pretrained else None)
        # 移除原分类头
        self.backbone.head = nn.Identity()
        self.feature_dim = 768
        print(f"使用预训练Swin-Tiny模型，特征维度: {self.feature_dim}")

        # 可选的空间注意力模块（作用于原始图像）
        self.lung_attn = LungAttentionModule(3) if use_attention else None

        # 可选的通道注意力模块（作用于特征图）
        self.lesion_attn = LesionChannelAttention(self.feature_dim) if use_attention else None

        # 分类头
        self.classification_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        print(f"分类头结构: {self.feature_dim}→512→{num_classes}，Dropout率: {dropout_rate}")

    def forward(
            self,
            x: torch.Tensor,  # 输入图像 (B, 3, 224, 224)
            lung_mask: Optional[torch.Tensor] = None  # 肺部掩膜 (B, 1, 224, 224)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: 输入图像
            lung_mask: 肺部掩膜（来自数据预处理的_segment_lungs函数）
        Returns:
            classification_logits: 分类任务输出 (B, num_classes)
            detection_output: 检测任务输出（未实现时为None）
        """
        print("\n===== 前向传播开始 =====")
        print(f"输入图像形状: {x.shape}")

        # ------------------- 空间注意力（作用于原始图像） -------------------
        if self.use_attention and lung_mask is not None:
            x = self.lung_attn(x, lung_mask)

        # ------------------- 特征提取（Swin Transformer） -------------------
        features = self.backbone(x)
        print(f"Swin Transformer输出特征形状: {features.shape}")

        # ------------------- 通道注意力（作用于特征图） -------------------
        if self.use_attention and self.lesion_attn is not None:
            # 调整特征维度以匹配注意力模块
            features_reshaped = features.unsqueeze(-1).unsqueeze(-1)  # (B, 768) → (B, 768, 1, 1)
            features_reshaped = self.lesion_attn(features_reshaped)
            features = features_reshaped.squeeze(-1).squeeze(-1)  # (B, 768, 1, 1) → (B, 768)
            print(f"应用通道注意力后特征形状: {features.shape}")

        # ------------------- 分类任务输出 -------------------
        classification_logits = self.classification_head(features)
        print(f"分类logits形状: {classification_logits.shape}")

        print("===== 前向传播结束 =====")
        return classification_logits, None


if __name__ == "__main__":
    # 测试模型结构
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型（使用预训练Swin Transformer + 注意力机制）
    model = TBTransformerModel(
        num_classes=2,
        use_attention=True,
        pretrained=True,
        dropout_rate=0.5
    ).to(device)

    # 模拟输入数据
    x = torch.randn(2, 3, 224, 224).to(device)  # 输入图像
    lung_mask = torch.randn(2, 1, 224, 224).to(device)  # 肺部掩膜

    # 前向传播
    with torch.no_grad():
        logits, _ = model(x, lung_mask)

    print("\n模型测试完成，输出形状符合预期")