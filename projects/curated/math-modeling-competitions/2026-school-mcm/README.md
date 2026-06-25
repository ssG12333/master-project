# 2026 校赛 — Problem C 边坡位移监测与预警

## 赛题概述

基于边坡多源传感器监测数据（光纤位移计 A、振弦式位移计 B、降雨量、孔隙水压力、微震事件、爆破振动），完成从**传感器校正 → 阶段识别 → 插补关联 → 分阶段预测 → 特征寻优与预警**的全链路建模。

每个问题均给出 Baseline（传统模型）和 Advanced（深度/强化学习）双轨方案。

## 代码架构

### 问题 1 — 传感器校正 → [q1_sensor_calibration/](q1_sensor_calibration/)

```
Class DataProcessor        — 数据加载、EDA(4张图)、CEEMDAN分解、特征工程(11维)
Class ModelBase            — 模型基类: evaluate(rmse,mae,r2) + cross_validate(TimeSeriesSplit)
Class RidgeRegressor       — PolynomialFeatures(2) + Ridge(α=1.0)  ← 最优模型 CV R²=0.9542
Class PolynomialRegressor  — PolynomialFeatures(3) + Ridge(α=0.01) → 过拟合
Class SVRRegressor         — StandardScaler + SVR(rbf, C=10, ε=0.5~1.0) → 严重过拟合
Class BaselineXGB          — XGBoost(50树, depth=2, lr=0.01, α=10, λ=50)
```

**核心数据流**: `主函数.py` 调度全部管线：
1. `DataProcessor.load_data()` → 读取 Excel，自动识别列名
2. `DataProcessor.plot_eda()` → 4 张探索性分析图
3. `DataProcessor.apply_ceemdan()` → PyEMD.CEEMDAN(max_imf=5)
4. `DataProcessor.extract_features()` → 构建 11 维特征
5. `RidgeRegressor/Poly/SVR/XGB.cross_validate()` → 5 折 TimeSeriesSplit
6. `visualization.py` → 21 张出版级图表(300 DPI)

### 问题 2 — 阶段识别 → [q2_stage_segmentation/](q2_stage_segmentation/)

```
Class Q2DataProcessor      — Hampel滤波(窗口7×3σ) + Savitzky-Golay(窗口51×3阶) + 速度/加速度/切线角
Class BaselinePELT         — 双准则变点检测: 速度阈值0.6 + 加速度极值(argrelextrema)
Class Q2PhysicsModeler     — 三阶段物理模型: Linear → Exponential/Power → Voight/Saito
Class NoiseDiscriminator   — 噪声跳变检测(5σ阈值) + 真假转换判别
Class Q2Visualizer         — 12张学术图表渲染
```

**三阶段物理模型**:
- 阶段1（缓慢匀速）: `Linear: x(t) = k*t + c` — scipy curve_fit
- 阶段2（加速形变）: `Exponential: A*exp(B*t)+C` 或 `Power: A*t^B+C` — 择优
- 阶段3（快速失稳）: `Voight/Saito: -ln(λ*(tf-t))/λ + C` — 预测失稳时刻 tf

**变点检测核心逻辑**:
```
cp1 (第一变点): 速度 > 0.6 持续窗口起点 → 约束 ∈ [7000, 8500]
cp2 (第二变点): 速度二阶导极大值点 → 约束 ∈ [cp1+500, n-50]
                ↓
         NoiseDiscriminator.is_real_transition()
         判别: 后段速度均值 > 前段×1.5 且 后段趋势 > 0
```

### 问题 3 — 插补与关联 → [q3_association_analysis/](q3_association_analysis/)

```
Class Q3DataProcessor      — 异构时间戳对齐(resample 1H) + 时滞特征构建(Lag 12~72h)
Class Q3Modeler            — MissForest(RF迭代填补) + IsolationForest + XGBoost+SHAP
Class PPOAnomalyAgent      — Gymnasium自定义环境 + Stable-Baselines3 PPO
```

**关键组件**:
- 多源数据按不同策略重采样：Rainfall/Seismic → SUM，其他 → MEAN
- 24h/12h/48h/72h 滞后降雨 + 6h 滞后孔压 → 捕捉渗透延迟效应
- MissForest(IterativeImputer, max_iter=50) 迭代填补
- IsolationForest 取多变量交集 + PPO智能体对比
- SHAP beeswarm/dependence/importance 三图联动

### 问题 4 — 分阶段预测 → [q4_staged_prediction/](q4_staged_prediction/)

```python
FEATURES = ['降雨量_mm', '孔隙水压力_kPa', '微震事件数', '爆破振动烈度_V',
            '降雨量_2h_sum', '孔压_2h_mean', '孔压_2h_diff', '水力耦合因子']
TARGET = '表面位移增量_mm'

# 爆破特征: 萨道夫斯基公式 V ∝ (Q^(1/3) / R)
df['爆破振动烈度_V'] = (Q^(1/3)) / R

# 水力耦合: 降雨渗流 × 孔压响应
df['水力耦合因子'] = 降雨量_24h × 孔压差分

# 阶段划分: K-Means 聚类 → 3阶段
# 集成模型: VotingRegressor(GBDT + RandomForest + ExtraTrees)
# 校准: 校准系数 = 训练集阶段位移变化 / 预测阶段总增量
```

### 问题 5 — 特征寻优与预警 → [q5_feature_warning/](q5_feature_warning/)

```python
# 5.1 穷举搜索 C(6,5)=6 组合
for drop_col in all_features:
    selected = [f for f in all_features if f != drop_col]
    rf_model = RandomForestRegressor.fit(X[selected], y)
    gb_model = GradientBoostingRegressor.fit(X[selected], y)
    scores[drop_col] = {rmse, r2}

# 5.2 LSTM-Attention 特征权重
class AttentionModel(nn.Module):
    lstm + MultiHeadAttention + FC → 从 attention 权重提取重要性

# 5.3 四级预警: 速度-加速度双参数判据
Level 0 (安全): v < v1, a ≈ 0
Level 1 (注意): v1 < v < v2
Level 2 (警戒): v2 < v < v3, a > 0
Level 3 (危险): v > v3, a >> 0
```

## 图表产出

| 问题 | 图表数 | 关键图表 |
|------|-------|---------|
| Q1 | 21 张 | CEEMDAN模态分解、非线性漂移校正主图、残差热力图、误差QQ图、Ridge校正散点 |
| Q2 | 12 张 | 原始数据/Hampel+SG对比、噪声vs真实转换判别、三阶段着色位移、KDE速度分布、综合诊断 |
| Q3 | 20 张 | 缺失值热力图、填补前后对比、SHAP beeswarm/dependence/importance、残差直方图 |
| Q4 | 18 张 | 4维时序EDA、相关性热力图、K-Means阶段聚类、速度KDE、多源箱线图、95%置信包络 |
| Q5 | 16 张 | 6组合RMSE柱状图、互信息排序、LSTM-Attention损失曲线、TTF预警图、四级预警阶梯图 |
