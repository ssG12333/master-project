# 数据分析与可视化合集

数据预测、统计分析和学术图表的补充项目合集，涵盖 BP 神经网络预测、音乐特征分析、学术箱线图/柱状图绘制和深度学习推理。

## 收录项目

### 天气预测 — BP 神经网络

```python
# bp_weather.py
# BP (Back Propagation) 神经网络天气预测
# 输入特征: 温度、湿度、气压、风速等气象因子
# 网络结构: input → hidden(ReLU) → output (回归)
# 训练: 梯度下降 + MSE Loss
```

### 音乐预测 — 特征分析与预测

```python
# music_prediction.py (新稿.py)
# 音乐特征提取与流派/受欢迎度预测
# 特征: 音频特征 (节奏、音调、能量、舞蹈性等)
# 模型: 回归/分类模型
#
# 画图.py: 结果可视化 (特征分布、预测 vs 实际散点)
```

### 学术图表绘制

```python
# box_plot.py (箱线图.py)
# 多组数据对比箱线图
# matplotlib + seaborn 出版级渲染
# 支持多子图布局、统计标注 (中位数/四分位数/异常值)

# bar_chart.py (柱状图.py)
# 分组/堆叠柱状图
# 误差棒 + 显著性标注 (p-value)
# 前三个.py / 后六个.py: 分批数据处理与图表生成
```

### 深度学习推理

```python
# dl_inference.py (123.py)
# 模型推理脚本: 加载预训练模型 → 前向推断

# mlp.py
# MLP 推理实现
```

## 技术栈

| 项目 | 技术 |
|------|------|
| 天气预测 | BP 神经网络, numpy, pandas |
| 音乐预测 | scikit-learn, pandas, matplotlib |
| 学术图表 | matplotlib, seaborn, 中文字体配置 |
| 深度学习推理 | PyTorch |

## 与主 curated 项目的关系

- [time-series-forecasting](../time-series-forecasting/) — 通用时序预测框架
- [stock-forecasting](../stock-forecasting/) — 股票预测 (GRU + 回测)
- [accident-random-forest-forecasting](../accident-random-forest-forecasting/) — 随机森林事故预测
- 本合集 — BP/统计分析/图表/推理的补充实现

## 运行方式

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch
cd data-analysis-collection

python bp_weather.py          # BP 天气预测
python music_prediction.py    # 音乐特征预测
python box_plot.py            # 箱线图
python bar_chart.py           # 柱状图
```

## 原始目录

- `天气预测/`
- `音乐预测/`
- `画表/`
- `深度学习推理/`
