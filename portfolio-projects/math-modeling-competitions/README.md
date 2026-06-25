# 数学建模竞赛合集

三次数学建模竞赛的完整作品合集，涵盖研究生国赛（华为杯）和校级选拔赛。

## 竞赛列表

### 2026 校赛 — Problem C 边坡位移监测与预警

最完整的项目。边坡位移时序数据的 5 问全链路建模（传感器校正 → 阶段识别 → 关联分析 → 分阶段预测 → 预警机制），每问均实现 Baseline + Advanced 双轨方案。

**核心代码已抽取**: [projects/curated/math-modeling-competitions/2026-school-mcm/](../../projects/curated/math-modeling-competitions/2026-school-mcm/)

**技术栈**: CEEMDAN · Savitzky-Golay · Hampel 滤波 · Ridge/SVR/XGBoost · MissForest · IsolationForest · K-Means · VotingRegressor(GBDT+RF+ET) · LSTM-Attention · 萨道夫斯基爆破公式 · Voight/Saito 失稳模型

**图表产出**: 87 张出版级学术图表 (300 DPI)

### 2025 国赛（华为杯）— Problem E 跨域光纤数据校准

16 路光纤传感器高频信号去噪、重采样、特征提取、聚类分组和域自适应校准。含 Stacking 集成分类器（LR+RF+SVC+XGBoost+LGB）+ 自定义 FeatureAttention 网络。

**核心代码已抽取**: [projects/curated/math-modeling-competitions/2025-national-cumcm/](../../projects/curated/math-modeling-competitions/2025-national-cumcm/)

**技术栈**: FFT · PyWavelets 小波去噪 · 重采样 · KMeans 聚类 · MLP · FeatureAttention · StackingClassifier

### 2025 校赛 — Problem C 社交媒体博主预测

MLP+LSTM+Transformer 三级联架构融合预测博主粉丝增长，含 4 问完整代码与论文。

**核心代码已抽取**: [projects/curated/math-modeling-competitions/2025-school-mcm/](../../projects/curated/math-modeling-competitions/2025-school-mcm/)

**技术栈**: PyTorch (MLP 256→128, LSTM hidden=128, Transformer d_model=128 nhead=8) · 回归分析 · 相关性矩阵 2D/3D

## 技术亮点

- **双轨方法论** (2026 校赛): Baseline 保证可解释性与可靠性，Advanced 展示前沿方法创新
- **物理约束建模**: 萨道夫斯基爆破经验公式、水力耦合因子、Voight/Saito 失稳模型
- **强化学习创新应用**: DDPG/DQN/SAC/PPO/MARL 分别用于传感器校正、阶段切分、异常检测、预测自适应、特征选择
- **信号处理管线**: CEEMDAN 模态分解 → 剔除高频 IMF1 → 重构低频趋势
- **异构集成**: VotingRegressor (GBDT+RF+ExtraTrees) 分阶段动态路由
