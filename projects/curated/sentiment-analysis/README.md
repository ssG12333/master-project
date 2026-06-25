# BiLSTM 情感分析模型

## 项目简介

基于 PyTorch 实现 IMDB 评论情感二分类 (positive/negative)，包含完整的文本分词、词表构建、序列填充、BiLSTM 分类建模和超参数消融实验。项目展示 NLP 文本分类的标准工程管线。

## 技术栈

- **核心框架**: Python 3.8+, PyTorch 2.x
- **NLP 工具**: NLTK (`word_tokenize`)
- **数据处理**: pandas, NumPy, collections.Counter
- **可视化**: matplotlib, tqdm
- **序列化**: pickle (保存训练指标)

---

## 模型架构: `BiLSTMClassifier`

```python
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.3):
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim,
                              num_layers=num_layers,
                              bidirectional=True,
                              batch_first=True,
                              dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
```

### 网络结构

```
Input (batch, max_len=300)   -- token indices (int)
  -> Embedding(vocab_size, 256, padding_idx=0)
     Output: (batch, 300, 256)
  -> BiLSTM(256 -> 512, num_layers, bidirectional)
     Output: (batch, 300, 1024)
      取最后时间步: (batch, 1024)
  -> Dropout(0.3)
  -> Linear(1024 -> 1)
  -> Sigmoid
     Output: (batch,)  -- [0,1] 表示 positive 概率
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vocab_size` | ~10001 | 词表大小 (max_words + 1, 含 `<PAD>`) |
| `embed_dim` | 256 | 词嵌入维度 |
| `hidden_dim` | 512 | LSTM 隐藏状态维度 |
| `num_layers` | 1 或 2 | BiLSTM 层数 (实验变量) |
| `dropout` | 0.3 | Dropout 比率 (num_layers>1 时层间 dropout) |
| `max_len` | 300 | 序列最大长度 (截断/填充) |

---

## 数据处理管线

### 1. 数据格式

输入 CSV 文件需包含两列:

| 列名 | 类型 | 说明 |
|------|------|------|
| `review` | str | 评论文本 |
| `sentiment` | str | `"positive"` 或 `"negative"` |

### 2. 词表构建 (`load_csv_data`)

```python
def load_csv_data(csv_path, max_words=10000, max_len=300):
```

1. 读取 CSV，提取 `review` 文本列
2. 将 `sentiment` 映射为二值标签: `positive -> 1`, `negative -> 0`
3. 使用 NLTK `word_tokenize` 对所有评论文本分词
4. 统计词频，取前 `max_words - 1` 个高频词构建词表
5. `<PAD>` 索引为 0，未知词 (OOV) 也映射为 0

### 3. 数据集类 (`CustomIMDBDataset`)

```python
class CustomIMDBDataset(Dataset):
    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        tokens = word_tokenize(review.lower())
        indices = [self.vocab.get(token, 0) for token in tokens]
        # 截断: indices[:max_len]
        # 填充: indices + [0] * (max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), torch.tensor(labels, dtype=torch.float)
```

- **分词**: `nltk.word_tokenize` + lowercase
- **OOV 处理**: 未登录词映射为 `0` (与 `<PAD>` 同索引)
- **序列长度**: 固定 `max_len=300`。超过截断，不足用 `0` 填充
- **80/20 划分**: 前 80% 训练，后 20% 验证

---

## 训练配置

### 超参数

| 参数 | 可取值 | 说明 |
|------|--------|------|
| `learning_rate` | 0.005, 0.0005 | Adam 学习率 (实验变量) |
| `num_layers` | 1, 2 | BiLSTM 层数 (实验变量) |
| `batch_size` | 64, 128 | 批次大小 (实验变量) |
| `num_epochs` | 20 | 最大训练轮次 |
| `optimizer` | Adam | 梯度优化器 |
| `criterion` | BCELoss | 二分类交叉熵损失 |
| `scheduler` | ReduceLROnPlateau(mode='min', factor=0.5, patience=2) | 验证 loss 不降时学习率减半 |

### 随机种子

```python
torch.manual_seed(42)
np.random.seed(42)
```

### 训练循环 (`train_model`)

每个 epoch:
1. **训练**: 遍历 train_loader, 计算 BCELoss, 反向传播, Adam 更新
2. **验证**: 遍历 val_loader, 计算 loss 和准确率 (阈值 0.5)
3. **调度**: `scheduler.step(val_loss)` 动态调整学习率
4. **记录**: 保存 train_losses, val_losses, val_accuracies 到 `.pkl` 文件

---

## 消融实验 (`main`)

遍历 3 组超参数的笛卡尔积进行对比:

```python
learning_rates = [0.005, 0.0005]
num_layers_list = [1, 2]
batch_sizes = [64, 128]
# 共 2 x 2 x 2 = 8 组实验
```

每组实验结果保存为 `metrics_lr{lr}_layers{layers}_batch{batch_size}.pkl`

### 评估指标

- **准确率**: `correct / total` (验证集和测试集)
- **损失曲线**: 训练/验证 loss 随 epoch 变化
- **输出**: 每组实验的最终测试准确率汇总打印

### 可视化 (`plot_metrics`)

```
Figure (12, 4)
  Subplot 1: Train Loss + Validation Loss 曲线
  Subplot 2: Validation Accuracy (%) 曲线
保存为: {title}.png
```

---

## 工作链路

1. **数据加载**: `load_csv_data(csv_path)` -> `train_dataset`, `val_dataset`, `vocab`
2. **词表构建**: 统计高频词 -> 建立 `token -> idx` 映射 (上限 10000)
3. **序列化**: `word_tokenize` + lowercase -> 转换为 idx 序列 -> 截断/填充至 300
4. **模型初始化**: `BiLSTMClassifier(vocab_size, embed_dim=256, hidden_dim=512, num_layers)`
5. **训练**: `train_model()` -> 20 epochs, ReduceLROnPlateau 调度
6. **评估**: 验证集准确率 (threshold=0.5)
7. **对比**: 8 组超参数组合, 输出汇总表

## 可扩展性

- **更换数据集**: 只需 CSV 包含 `review` 和 `sentiment` 两列
- **增加模型深度**: 修改 `num_layers`
- **使用预训练词向量**: 替换 `nn.Embedding` 为加载的 GloVe / Word2Vec 权重
- **添加测试集**: 在 `load_csv_data` 中支持三路划分
