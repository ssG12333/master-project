# 通用时间序列预测框架

## 项目简介

本项目是一个时间序列预测与实验框架，集成 9 种深度学习模型和统一的数据加载模块，支持长周期/短周期预测、异常检测、分类和插补任务。基于 PyTorch 构建，核心代码位于 `forecast/` 目录。

## 技术栈

- **框架**: Python 3.8+, PyTorch 2.x
- **数据处理**: NumPy, pandas, scikit-learn (StandardScaler)
- **时间序列**: statsmodels, sktime
- **特殊依赖**: reformer_pytorch (LSH attention), mamba_ssm (Mamba 模型)
- **可视化**: matplotlib, tqdm

## 模型目录 (`models/`)

所有模型均通过 `models/__init__.py` 统一导出，在 `Exp_Basic.model_dict` 中注册。

| 模型类 | 文件名 | 说明 | 关键模块 |
|--------|--------|------|----------|
| `Autoformer` | `Autoformer.py` | 基于自相关机制的序列级连接，复杂度 O(LlogL) | `AutoCorrelation`, `series_decomp`, `DataEmbedding_wo_pos` |
| `Transformer` | `Transformer.py` | 经典 Transformer 架构，复杂度 O(L^2) | `FullAttention`, `DataEmbedding`, `Encoder/Decoder` |
| `DLinear` | `DLinear.py` | 将序列分解为趋势+季节分量后分别线性映射 | `series_decomp`, `nn.Linear(seq_len, pred_len)` |
| `GraphPatchTST` | `GraphPatchTST.py` | 将序列划分为 patch + 图结构建模 | 图注意力 + patch 嵌入 |
| `GRU` | `GRU.py` | 单层/多层 GRU 循环网络 | `nn.GRU(enc_in, d_model, e_layers, batch_first)` + `nn.Linear(d_model, enc_in)` |
| `CNN1D` | `CNN1D.py` | 一维卷积时间序列模型 | `nn.Conv1d(seq_len, enc_in, kernel_size=3)` + `nn.Linear(enc_in, pred_len)` |
| `wo_global_embedding` | `wo_global_embedding.py` | AGCRN 消融变体：去掉全局节点嵌入 | 删除 `node_embeddings` |
| `wo_graph_learning` | `wo_graph_learning.py` | AGCRN 消融变体：去掉图学习模块 | 删除自适应图生成 |
| `Mamba` (条件导入) | `Mamba.py` | 状态空间模型，需安装 `mamba_ssm` | S6 选择性扫描 |

### 核心模型架构细节

**Autoformer** (`Autoformer.py`):
```python
class Model(nn.Module):
    def __init__(self, configs):
        # 序列分解: moving_avg kernel_size
        self.decomp = series_decomp(kernel_size=configs.moving_avg)
        # 编码器嵌入: 无位置编码
        self.enc_embedding = DataEmbedding_wo_pos(enc_in, d_model, embed, freq, dropout)
        # 编码器: N 层 EncoderLayer(AutoCorrelationLayer)
        # 解码器: AutoCorrelationLayer + Cross-correlation + FeedForward
```

**GRU** (`GRU.py`):
```python
class Model(nn.Module):
    def __init__(self, configs):
        self.gru = nn.GRU(input_size=configs.enc_in,
                          hidden_size=configs.d_model,
                          num_layers=configs.e_layers,
                          batch_first=True)
        self.projection = nn.Linear(configs.d_model, configs.enc_in)
    # forward 返回 dec_out[:, -self.pred_len:, :]
```

**DLinear** (`DLinear.py`):
```python
class Model(nn.Module):
    def __init__(self, configs, individual=False):
        self.decompsition = series_decomp(configs.moving_avg)
        # individual=True: 每通道独立 Linear(seq_len, pred_len)
        # individual=False: 共享 Linear(seq_len, pred_len)
        # 权重初始化为 1/seq_len
        self.Linear_Seasonal.weight = nn.Parameter((1/seq_len) * torch.ones([pred_len, seq_len]))
```

## 实验框架 (`exp/`)

继承体系：`Exp_Basic` (基类) -> `Exp_Long_Term_Forecast`

### Exp_Basic (`exp_basic.py`)

```python
class Exp_Basic(object):
    def __init__(self, args):
        self.model_dict = {
            "Autoformer": Autoformer, "Transformer": Transformer,
            "DLinear": DLinear, "GraphPatchTST": GraphPatchTST,
            "GRU": GRU, "CNN1D": CNN1D,
            "wo_global_embedding": wo_global_embedding,
            "wo_graph_learning": wo_graph_learning,
        }
        # Mamba 条件导入
        if args.model == "Mamba":
            from models import Mamba
            self.model_dict[Mamba] = Mamba
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
```

- `_acquire_device()`: 支持单 GPU (`cuda:0`), 多 GPU (`DataParallel`), 和 CPU
- `_build_model()`: 由子类实现

### Exp_Long_Term_Forecast (`exp_long_term_forecasting.py`)

继承 `Exp_Basic`，实现完整训练/验证/测试流程：

```python
class Exp_Long_Term_Forecast(Exp_Basic):
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model)
        return model

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()
```

**训练流程** (`train()` 方法):
1. 读取 train/val/test DataLoader (`_get_data`)
2. 初始化 `EarlyStopping(patience=self.args.patience)`
3. 逐 epoch 训练，支持 AMP (`torch.cuda.amp.autocast` + `GradScaler`)
4. 每 epoch 后在验证/测试集上评估 (`vali()`)
5. 学习率调度 (`adjust_learning_rate`): 支持 type1, type2, cosine 策略
6. 早停时保存最佳 checkpoint

**测试流程** (`test()` 方法):
1. 加载 checkpoint
2. 逐 batch 预测，支持反归一化 (`inverse_transform`)
3. 输出预测可视化 PDF
4. 计算指标: MAE, MSE, RMSE, MAPE, MSPE (`utils.metrics.metric`)
5. 可选 DTW 距离计算 (`utils.dtw_metric.accelerated_dtw`)
6. 保存 `pred.npy`, `true.npy`, `metrics.npy`

### 评估指标 (`utils/metrics.py`)

```python
def metric(pred, true):
    mae = MAE(pred, true)       # 平均绝对误差
    mse = MSE(pred, true)       # 均方误差
    rmse = RMSE(pred, true)     # 均方根误差
    mape = MAPE(pred, true)     # 平均绝对百分比误差
    mspe = MSPE(pred, true)     # 均方百分比误差
    r2 = R2(pred, true)         # 决定系数 (单独提供)
    return mae, mse, rmse, mape, mspe
```

## 数据提供层 (`data_provider/`)

### data_factory.py

`data_provider(args, flag)` 函数根据 `args.data` 选择对应的 Dataset 类：

```python
data_dict = {
    'ETTh1': Dataset_ETT_hour, 'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute, 'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom, 'm4': Dataset_M4,
    'PSM': PSMSegLoader, 'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader, 'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader, 'UEA': UEAloader
}
```

根据 `args.task_name` 分支处理:
- `anomaly_detection`: 传入 `win_size=args.seq_len`
- `classification`: 使用 `collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)`
- 默认（预测任务）: 传入 `size=[seq_len, label_len, pred_len]`, `features`, `target`, `timeenc`, `freq`, `seasonal_patterns`

### data_loader.py

基类 `Dataset_Group(Dataset)` 处理:
- **滑动窗口**: `seq_len`(输入长度), `label_len`(解码器起始标记长度), `pred_len`(预测长度)
- **特征模式**: `features='S'`(单变量), `'M'`(多变量), `'MS'`(多特征单目标)
- **归一化**: `StandardScaler` 基于训练集拟合
- **时间编码**: `time_features()` 生成 hour/week/month 等周期性特征
- **数据增强**: `run_augmentation_single` (可选)

**支持的数据集**:
- **ETT** (Electricity Transformer Temperature): ETT-h1, ETT-h2 (小时级), ETT-m1, ETT-m2 (分钟级)
- **Custom**: 通用 CSV 格式
- **M4**: M4 竞赛数据集
- **异常检测**: PSM, MSL, SMAP, SMD, SWAT
- **分类**: UEA 多元时间序列分类

## 网络层目录 (`layers/`)

| 文件 | 核心类 | 功能 |
|------|--------|------|
| `Embed.py` | `PositionalEmbedding`, `TokenEmbedding`, `FixedEmbedding`, `TimeFeatureEmbedding`, `DataEmbedding`, `DataEmbedding_wo_pos` | 将原始序列嵌入为 d_model 维度，支持位置编码、时间特征编码 |
| `AutoCorrelation.py` | `AutoCorrelation`, `AutoCorrelationLayer` | 自相关机制：周期发现 + 时延聚合，可替换 Self-Attention |
| `SelfAttention_Family.py` | `FullAttention`, `ProbAttention`, `DSAttention`(去平稳化注意力), `AttentionLayer`, `ReformerAttention` | 多种注意力机制 |
| `Autoformer_EncDec.py` | `Encoder`, `Decoder`, `EncoderLayer`, `DecoderLayer`, `series_decomp`, `moving_avg`, `my_Layernorm` | Autoformer 编解码器 |
| `Transformer_EncDec.py` | `Encoder`, `Decoder`, `EncoderLayer`, `DecoderLayer`, `ConvLayer` | 标准 Transformer 编解码器 |
| `Conv_Blocks.py` | `Inception_Block_V1` | Inception 卷积模块 |
| `FourierCorrelation.py` | `FourierBlock`, `FourierCrossAttention` | 傅里叶域相关性 |
| `MultiWaveletCorrelation.py` | `WaveletTransform`, `MultiWaveletBlock` | 多小波相关性 |
| `Crossformer_EncDec.py` | Crossformer 专用编解码结构 | 跨维度注意力 |
| `ETSformer_EncDec.py` | ETSformer 指数平滑编解码 | 增长/季节/趋势分解 |
| `Pyraformer_EncDec.py` | Pyraformer 金字塔注意力模块 | 多尺度金字塔注意力 |
| `StandardNorm.py` | `Normalize`(可逆实例归一化) | RevIN 可逆归一化 |

### Embedding 层详解 (`layers/Embed.py`)

```python
class PositionalEmbedding(nn.Module):   # 正弦位置编码, max_len=5000
class TokenEmbedding(nn.Module):        # 1D Conv(kernel=3) 映射到 d_model
class FixedEmbedding(nn.Module):        # 固定频率编码 (用于时间特征)
class TimeFeatureEmbedding(nn.Module):  # 时间特征 -> Linear(d_model)
class DataEmbedding(nn.Module):         # TokenEmbed + PositionalEmbed + TimeFeatureEmbed
class DataEmbedding_wo_pos(nn.Module):  # TokenEmbed + TimeFeatureEmbed (无位置编码)
```

## 工具模块 (`utils/`)

| 文件 | 功能 |
|------|------|
| `tools.py` | `EarlyStopping(patience, verbose, delta)` + `adjust_learning_rate(optimizer, epoch, args)` (type1/type2/cosine) |
| `metrics.py` | MAE, MSE, RMSE, MAPE, MSPE, RSE, CORR, R2 |
| `masking.py` | `TriangularCausalMask`, `ProbMask` |
| `timefeatures.py` | `time_features()` 时间特征提取 |
| `augmentation.py` | `run_augmentation`, `run_augmentation_single` 数据增强 |
| `losses.py` | 自定义损失函数 |
| `dtw_metric.py` | 加速 DTW 距离计算 |

## 关键超参数配置

超参数通过 `args` 对象传入，典型配置：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `seq_len` | int | 96 | 输入序列长度 |
| `label_len` | int | 48 | 解码器起始标记长度 |
| `pred_len` | int | 24 | 预测序列长度 |
| `enc_in` | int | 7 | 编码器输入通道数 |
| `dec_in` | int | 7 | 解码器输入通道数 |
| `d_model` | int | 512 | 模型隐藏维度 |
| `d_ff` | int | 2048 | 前馈网络维度 |
| `n_heads` | int | 8 | 注意力头数 |
| `e_layers` | int | 2 | 编码器层数 |
| `d_layers` | int | 1 | 解码器层数 |
| `dropout` | float | 0.1 | Dropout 比率 |
| `learning_rate` | float | 0.001 | Adam 学习率 |
| `batch_size` | int | 32 | 批次大小 |
| `train_epochs` | int | 10 | 最大训练轮数 |
| `patience` | int | 3 | 早停耐心值 |
| `moving_avg` | int | 25 | 移动平均核大小 (Autoformer/DLinear) |
| `features` | str | 'M' | 特征模式: M/S/MS |
| `freq` | str | 'h' | 时间频率 |
| `embed` | str | 'timeF' | 时间特征编码方式 |
| `output_attention` | bool | False | 是否输出注意力权重 |
| `use_amp` | bool | False | 是否使用混合精度训练 |
| `lradj` | str | 'type1' | 学习率调整策略 |

## 工作链路

1. **数据读取**: `data_provider(args, flag)` 选择数据集类并返回 `Dataset` + `DataLoader`
2. **窗口构造**: `Dataset_Group.__init__` 设置 `seq_len` / `label_len` / `pred_len`
3. **归一化**: `StandardScaler` 基于训练集拟合，对 val/test 变换
4. **时间编码**: `time_features()` 生成周期性特征
5. **模型选择**: `Exp_Basic.model_dict[args.model]` 实例化模型
6. **训练**: `Exp_Long_Term_Forecast.train()` 执行早停、学习率调度、AMP
7. **评估**: `metric()` 计算 MAE/MSE/RMSE/MAPE/MSPE + 可选 DTW
8. **结果保存**: 预测值、真实值、指标保存为 `.npy` 文件，可视化输出为 `.pdf`

## 可扩展性

- **新增模型**: 在 `models/` 下创建文件，实现 `class Model(nn.Module)`，在 `Exp_Basic.model_dict` 中注册
- **新增数据集**: 在 `data_loader.py` 中继承 `Dataset_Group`，在 `data_factory.data_dict` 中注册
- **新增任务**: 在 `Exp_Long_Term_Forecast` 中添加对应的 `_get_data` 分支和 loss 函数
