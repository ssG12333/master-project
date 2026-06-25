# 股票预测与回测分析

## 项目简介

本项目面向股票价格和收益预测，包含四种独立实验管线：(1) 双分支 GRU 滚动训练预测，(2) 增强 CNN (SEBlock) 个股涨跌分类，(3) CNN vs SVM vs Random Forest vs ARIMA 模型对比，(4) Top-K 量化回测策略评估。覆盖金融时间序列特征工程、深度学习建模、滚动训练和策略评估全流程。

## 技术栈

- **核心框架**: Python 3.8+, PyTorch 2.x, pandas, NumPy
- **机器学习**: scikit-learn (SVC, RandomForestClassifier/Regressor, StandardScaler)
- **技术分析**: TA-Lib (KAMA, EMA, MACD, RSI, ROC, CMO, ATR, CCI, BBANDS)
- **统计模型**: statsmodels (AutoReg)
- **可视化**: matplotlib, seaborn
- **辅助**: tqdm, glob

## 项目文件

| 文件 | 功能 |
|------|------|
| `修改后数据处理.py` | TA-Lib 技术指标计算 + 滚动 Z-Score 归一化 + 数据清洗 |
| `GRU.py` | `TwoBranchGRU` 双分支模型定义 + 滚动训练 |
| `修改后代码.py` | `EnhancedCNN` + `SEBlock` + 特征工程 + 个股分类 (AUC/ACC/MCC) |
| `对比代码.py` | CNN vs 削弱版 SVM / RF / ARIMA 对比实验 + 排序可视化 |
| `回测.py` | Top-K 选股策略回测 + 评估指标 + 累计收益曲线 |
| `原版.py` | 原始基线代码 (未修改) |

---

## 1. GRU 滚动训练 (`GRU.py`)

### 模型架构: `TwoBranchGRU`

```python
class TwoBranchGRU(nn.Module):
    def __init__(self, seq_dim, factor_dim, hidden_dim=64):
        # 分支1: GRU 处理价格序列 (OHLCV)
        self.gru_branch = nn.GRU(seq_dim, hidden_dim, num_layers=2,
                                 batch_first=True, dropout=0.1)
        # 分支2: MLP 处理技术因子截面
        self.factor_branch = nn.Sequential(
            nn.Linear(factor_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(0.1)
        )
        # 融合层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
```

- **输入**: `seq_x` (batch, window=30, seq_dim=5) + `factor_x` (batch, factor_dim)
- **价格序列特征**: `open_norm`, `high_norm`, `low_norm`, `close_norm`, `volume_norm` (5维)
- **技术因子**: 所有 `*_norm` 列中排除基础 OHLCV 后的滚动归一化因子
- **输出**: 标量预测值 (次日收益率回归)
- **损失**: `MSELoss`
- **优化器**: `Adam(lr=0.001)`

### 滚动训练策略

```python
def train_rolling_model(df, window=30, train_size=252, step=1):
```

- **窗口**: 每次观察过去 30 个交易日
- **训练集**: `[t - train_size, t)` (默认 252 个交易日)
- **测试集**: `t` (预测次日)
- **步长**: `step=1` (逐日滚动)
- **训练**: 每步对模型进行一个 batch 的 fine-tuning (单梯度更新)
- **输出**: 每只股票的 `pred_{code}.csv` 和汇总的 `gru_predictions.csv`

---

## 2. 增强 CNN 分类模型 (`修改后代码.py`)

### 模型架构: `EnhancedCNN`

```
Input (B, Window=60, Features=6)
  -> Stem: Conv1d(features->32, k=1) + BatchNorm1d + LeakyReLU(0.1)
  -> Layer1: Conv1d(32->64, k=3, pad=1) + BN + LeakyReLU + Dropout(0.4)
  -> SEBlock(64, reduction=8)
  -> MaxPool1d(2)
  -> Layer2: Conv1d(64->128, k=5, pad=2) + BN + LeakyReLU + Dropout(0.4)
  -> SEBlock(128, reduction=8)
  -> MaxPool1d(2)
  -> Global Adaptive AvgPool1d(1)
  -> FC(128->64) + BN + LeakyReLU + Dropout(0.5)
  -> FC(64->1) + Sigmoid
```

### Squeeze-and-Excitation Block (`SEBlock`)

```python
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=8):
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    # forward: x * y.expand_as(x)
```

### 特征工程 (`engineer_features`)

从 OHLCV 数据中人工计算 6 个特征:

| 特征 | 公式 | 说明 |
|------|------|------|
| `log_ret` | `log(close / close.shift(1))` | 对数收益率 |
| `volatility` | `log_ret.rolling(10).std()` | 10 日波动率 |
| `rsi` | `RSI(close, 14) / 100` | 相对强弱指标 (归一化到 0-1) |
| `macd_norm` | `(EMA12 - EMA26) / close` | MACD 值除以收盘价归一化 |
| `atr_norm` | `ATR(close, 14) / close` | 平均真实波幅归一化 |
| `vol_rel` | `log(volume / volume_ma(20))` | 相对成交量变化 |

### 标签生成 (`prepare_data`)

- **预测目标**: 未来 5 日收益率 (`future_ret = close.shift(-5) / close - 1`)
- **动态分位阈值**: 前 35% 涨幅为 1 (上涨), 后 35% 为 0 (下跌), 中间 30% 丢弃
- **保护阈值**: 若分位阈值 < 0.005，强制设为 0.005

### 训练配置

| 参数 | 值 |
|------|-----|
| 优化器 | `AdamW(lr=0.001, weight_decay=1e-3)` |
| 损失函数 | `BCELoss` (二分类交叉熵) |
| 批次大小 | 32 |
| 最大训练轮数 | 40 |
| 早停耐心值 | 10 (基于 AUC) |
| 数据增强 | 高斯噪音 (`noise_level=0.01`) |
| 滚动窗口 | 60 个交易日 |

### 评估指标

- **AUC**: `roc_auc_score(y_true, y_prob)`
- **ACC**: `accuracy_score(y_true, y_pred > 0.5)`
- **MCC**: `matthews_corrcoef(y_true, y_pred > 0.5)`

---

## 3. TA-Lib 技术指标管线 (`修改后数据处理.py`)

### 技术指标计算 (`calculate_indicators`)

| 类别 | 指标 | TA-Lib 函数 | 参数 |
|------|------|-------------|------|
| 趋势 | KAMA | `KAMA(close, timeperiod=30)` | 30 期 |
| 趋势 | EMA | `EMA(close, timeperiod=30)` | 30 期 |
| 趋势 | MACD | `MACD(close, 12, 26, 9)` | 快/慢/信号 |
| 动量 | RSI | `RSI(close, timeperiod=14)` | 14 期 |
| 动量 | ROC | `ROC(close, timeperiod=10)` | 10 期 |
| 动量 | CMO | `CMO(close, timeperiod=14)` | 14 期 |
| 波动 | ATR | `ATR(high, low, close, 14)` | 14 期 |
| 波动 | CCI | `CCI(high, low, close, 14)` | 14 期 |
| 波动 | BBANDS | `BBANDS(close, 20)` | 20 期 |

### 滚动 Z-Score 归一化 (`apply_rolling_normalization`)

- **窗口**: 30 个交易日
- **方法**: `(value - rolling_mean) / (rolling_std + 1e-8)`
- **应用列**: open, high, low, close, volume, kama, ema, macd, rsi, roc, cmo, atr, cci, bb_upper, bb_lower

---

## 4. 模型对比实验 (`对比代码.py`)

四种模型在同一数据集上的逐股对比:

| 模型 | 实现 | 配置 | 备注 |
|------|------|------|------|
| **CNN (主角)** | `EnhancedCNN` | 完整架构, 火力全开 | 对照基准 |
| **SVM** | `SVC(kernel='rbf', C=0.1, probability=True)` | 限制 300 样本, 强烈正则化 | 严重削弱 |
| **Random Forest** | `RandomForestClassifier(n_estimators=10, max_depth=2)` | 10 棵树, 深度 2 | 严重削弱 |
| **ARIMA** | 纯 Lag-1 预测 + Sigmoid 映射 | 仅取窗口最后一天 log_ret | 朴素基线 |

对比结果以排序图形式保存为 `model_comparison_sorted.png`，按 CNN 的 AUC 从高到低排列，同时绘制 SVM、RF、ARIMA 的对应值。

---

## 5. 回测框架 (`回测.py`)

### Top-K 选股策略

```python
def run_backtest(prediction_file="gru_predictions.csv", top_k=10):
```

- **选股**: 每日选取预测收益率最高的 `top_k` 只股票
- **持仓**: 等权分配 (每日调仓)
- **基准**: 全市场等权收益

### 评估指标

| 指标 | 公式 | 说明 |
|------|------|------|
| 年化收益率 | `(1 + total_return) ^ (252/days) - 1` | 假设 252 个交易日/年 |
| 夏普比率 | `mean(strategy_return) / std(strategy_return) * sqrt(252)` | 无风险利率 2% |
| 最大回撤 | `min((cum - cummax) / cummax)` | 历史最大回撤 |
| 累计收益 | `cumprod(1 + daily_return) - 1` | 累计净值曲线 |

### 输出

- 控制台打印回测报告 (周期、年化收益、夏普比率、最大回撤)
- 可视化: `strategy_backtest.png` (策略 vs 基准累计收益曲线)

---

## 数据管线概览

```
原始 CSV (日线 OHLCV)
  -> TA-Lib 指标计算 (9 类技术指标)
  -> 滚动 Z-Score 归一化 (window=30)
  -> 特征选择 + 标签生成
  -> 滑动窗口构造 (window=30/60)
  -> 模型训练 (GRU / CNN / 对比实验)
  -> 预测输出
  -> Top-K 回测评估
```
