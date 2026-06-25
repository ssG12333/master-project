# 数学建模竞赛合集

三场数学建模竞赛的完整代码与文档，包含研究生国赛（华为杯）和校级选拔赛。

## 技术栈全景

### 信号处理
| 方法 | 用途 | 应用竞赛 |
|------|------|----------|
| **CEEMDAN** (PyEMD) | 自适应模态分解去噪，剔除 IMF1 高频毛刺，保留 IMF2+ 低频趋势 | 2026 Q1, 2025 国赛 |
| **Savitzky-Golay 滤波** | 窗口 51 × 3 阶多项式平滑位移数据，保持物理单调递增 | 2026 Q2, Q4, Q5 |
| **Hampel 滤波** | 窗口 7 × 3σ 剔除孤立毛刺（如附件 2 第 34 点异常跳变） | 2026 Q2 |
| **FFT + 小波去噪** | PyWavelets 对 32kHz 振动信号去噪，保留时频特征 | 2025 国赛 |
| **卡尔曼滤波** | 状态空间模型估计传感器真实值（Baseline 方案） | 2026 Q1 设计 |
| **高斯平滑** | σ=100 平滑速度场，σ=200 平滑加速度场，消除微分噪声 | 2026 Q2 |

### 机器学习
| 方法 | 关键参数 | 应用竞赛 |
|------|---------|----------|
| **Ridge 回归** + PolynomialFeatures(2) | α=1.0, TimeSeriesSplit(n_splits=5), CV R²=0.9542 | 2026 Q1 |
| **SVR** + StandardScaler | kernel='rbf', C=10, ε=0.5~1.0, gamma='auto' | 2026 Q1 |
| **XGBoost** | n_estimators=50, max_depth=2, lr=0.01, subsample=0.5, α=10/λ=50 | 2026 Q1, Q3 |
| **RandomForestRegressor** | 集成模型基学习器之一 | 2026 Q4, Q5 |
| **GradientBoostingRegressor** | 集成模型基学习器之一 | 2026 Q4, Q5 |
| **ExtraTreesRegressor** | 集成模型基学习器之一 | 2026 Q4 |
| **VotingRegressor** | GBDT + RF + Extra Trees 异构集成 | 2026 Q4 |
| **K-Means** | 自动划分三个形变演化阶段 | 2026 Q4 |
| **IterativeImputer + RandomForest** | MissForest 迭代式缺失值填补，max_iter=50 | 2026 Q3 |
| **IsolationForest** | 无监督多变量联合异常检测 | 2026 Q3 |
| **Linear / Exponential / Power / Voight** | 三阶段物理模型拟合（scipy curve_fit） | 2026 Q2 |

### 深度学习
| 方法 | 架构细节 | 应用竞赛 |
|------|---------|----------|
| **MLP + LSTM + Transformer** (MLPLSTMTransformerModel) | MLP(256→128), LSTM(hidden=128), Transformer(d_model=128, nhead=8, 2层) | 2025 校赛 Q1 |
| **LSTM-Attention** | 基于 Multi-Head Self-Attention 的特征重要性权重学习 | 2026 Q5 |
| **TransformerEncoder** (自定义) | nn.TransformerEncoderLayer → nn.TransformerEncoder(num_layers=2) | 2025 校赛 Q1 |
| **LSTM-AE 时序自编码器** | 编码→隐空间→解码，重构误差梯度用于阶段切分（Advanced 设计） | 2026 Q2 设计 |
| **TFT (Temporal Fusion Transformer)** | 原生支持已知未来输入 + 静态协变量（Advanced 方案） | 2026 Q4 设计 |
| **ST-GAT (时空图注意力网络)** | 5 节点图（5 传感器变量），学习时序演进中的动态注意力权重 | 2026 Q3 设计 |
| **Diffusion Model (TS-Diff)** | 生成式扩散模型去噪 → 缺失值填补 + 异常修复 | 2026 Q3 设计 |

### 强化学习
| 方法 | 问题形式化 | 应用竞赛 |
|------|-----------|----------|
| **DDPG (Deep Deterministic Policy Gradient)** | 连续控制：观测数据窗口→输出校正系数/截距，最小化 DTW 距离 | 2026 Q1 Advanced |
| **DQN (Deep Q-Network)** | 离散决策：在时间轴上"游走"，基于重构误差梯度决定阶段切分点 | 2026 Q2 Advanced |
| **SAC (Soft Actor-Critic)** | 连续控制：检测爆破→输出 Attention Mask 或特征重加权 | 2026 Q4 Advanced |
| **PPO (Proximal Policy Optimization)** | Stable-Baselines3 PPO 训练异常检测智能体 | 2026 Q3 Advanced |
| **MARL (Multi-Agent RL)** | 多智能体特征选择：每个特征一个 agent，决定是否加入预测联盟 | 2026 Q5 Advanced |

### 特征工程
| 特征 | 构建方法 | 应用竞赛 |
|------|---------|----------|
| 爆破振动烈度 V | 萨道夫斯基公式 V ∝ (Q^(1/3) / R) | 2026 Q4 |
| 水力耦合因子 | 降雨渗流 × 孔压响应交叉特征 | 2026 Q4 |
| 时滞特征 | Rainfall_lag_{12,24,48,72}h, PorePressure_lag_6h, rolling window统计 | 2026 Q3, Q4 |
| CEEMDAN 衍生特征 | A_smooth, A_sq, A_cube, A_vel, A_bin, A_diff2, A_ratio, A_deviation (11维) | 2026 Q1 |
| 宏观切线角 | arctan(v / v0)，v0 = 前 144 点均值 | 2026 Q2 |
| 频域特征 | 频谱质心、带宽、自相关（scipy FFT, PyWavelets） | 2025 国赛 |

## 项目结构

```
math-modeling-competitions/
├── README.md                                    # 本文件 — 技术栈全景
├── 2026-school-mcm/                             # 2026 校赛（最完整，5 问代码 + 图表）
│   ├── README.md                                #
│   ├── overall_approach.md                      #
│   ├── q1_sensor_calibration/    # Q1: CEEMDAN + Ridge/SVR/XGBoost → CV R²=0.9542
│   │   ├── README.md / figures/
│   │   └── main.py, data_processing.py, model_algorithm.py, visualization.py
│   ├── q2_stage_segmentation/    # Q2: Hampel+SG滤波 → PELT变点 → 三阶段物理模型
│   │   ├── README.md / figures/
│   │   └── main.py, data_processor.py, models.py, visualization.py
│   ├── q3_association_analysis/  # Q3: MissForest插补 + IsolationForest + SHAP
│   │   ├── README.md / figures/
│   │   └── main.py, core_models.py, visualizations.py
│   ├── q4_staged_prediction/     # Q4: K-Means阶段划分 + VotingRegressor集成
│   │   ├── README.md / figures/
│   │   └── main.py
│   └── q5_feature_warning/       # Q5: 穷举C(6,5) + LSTM-Attention + 四级预警
│       ├── README.md / figures/
│       └── main.py
├── 2025-national-cumcm/                         # 2025 国赛（华为杯）Problem E
│   ├── README.md / figures/
│   └── q1_stage1~3, q2, q3_q4 (5个.py)
└── 2025-school-mcm/                             # 2025 校赛 Problem C
    ├── README.md / figures/
    └── q1~q4 (7个.py)
```
