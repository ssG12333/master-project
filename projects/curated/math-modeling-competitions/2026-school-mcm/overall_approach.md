
基线模型（Baseline）用于对齐验证，以及前沿模型（Advanced SOTA）

📌 问题 1：多源传感器数据校正 (Data Calibration)

🎯 核心难点： 传感器之间的非线性零漂与时变累积误差。

🟢 Baseline: 岭回归 (Ridge Regression) + 卡尔曼滤波 (Kalman Filter) 状态估计。

🔴 Advanced (大炮打蚊子): 基于 DDPG 的自适应时序校准智能体 (DDPG-based Calibration Agent)

思路： 将数据校正定义为连续控制问题。强化学习智能体观测当前及历史的光纤数据窗口（State），输出动态的校正系数和截距（Action），以最小化与振弦式基准数据的动态时间规整（DTW）距离作为奖励（Reward）。

📌 问题 2：三段式形变阶段识别 (Stage Segmentation)

🎯 核心难点： 区分工程扰动引起的瞬时跳变与真实的长期形变状态转换。

🟢 Baseline: Savitzky-Golay 滤波平滑 + PELT 突变点检测。

🔴 Advanced (大炮打蚊子): LSTM-时序自编码器 (LSTM-AE) + Deep Q-Network (DQN) 动态切分

思路： 首先用 LSTM-AE 将时序数据映射到隐空间，滤除高频噪声；然后利用 DQN 智能体在时间轴上“游走”，基于重构误差梯度的突变来决定是否在该时间点“下刀”（划定阶段转换节点）。

📌 问题 3：时序预处理与多维关联分析 (Imputation, Denoising & Anomaly Association)

🎯 核心难点： 多变量（降雨、微震、位移等）之间的空间与时滞耦合关系及联合异常。

🟢 Baseline: 随机森林插补 (MissForest) + 孤立森林 (Isolation Forest) 取交集。

🔴 Advanced (大炮打蚊子): 时序扩散模型 (TS-Diff) + 时空图注意力网络 (ST-GAT)

思路： 利用生成式 Diffusion Model 的去噪能力来做缺失值补齐和异常值修复。随后构建一张图（节点为5个传感器变量），使用 GAT 学习不同变量在时序演进过程中的动态图注意力权重，以此定量评估“贡献度”。

📌 问题 4：含突发扰动的分阶段演化预测 (Multi-Stage Time-Series Forecasting)

🎯 核心难点： “爆破”这种稀疏、极值型偶发事件对位移的滞后和非线性冲击。

🟢 Baseline: XGBoost 特征工程 + 分阶段 GRU 网络。

🔴 Advanced (大炮打蚊子): 时序融合Transformer (Temporal Fusion Transformer, TFT) + Soft Actor-Critic (SAC) 扰动自适应

思路： TFT 原生支持已知未来输入（如预测降雨）和静态协变量。为了处理爆破脉冲，引入 SAC 强化学习智能体。当检测到“爆破”发生时，SAC 输出动态的 Attention 掩码（Attention Mask）或特征重加权，强制 Transformer 网络关注爆破后的演化特征。

📌 问题 5：特征组合寻优与预警机制 (Feature Selection & Warning Mechanism)

🎯 核心难点： 最优特征子集搜索与高维连续空间的预警阈值设定。

🟢 Baseline: 穷举搜索 (Exhaustive Search) + K-Means 聚类分级预警。

🔴 Advanced (大炮打蚊子): 多智能体强化学习 (MARL) 特征选择 + 动态风险流形预警

思路： 设定多个智能体，每个代表一个特征（降雨、微震等），智能体决定是否“加入”预测联盟（Action: 0 or 1）。团队总奖励为预测模型的验证集负 MSE。对于预警，建立基于相空间重构（Phase Space Reconstruction）的李雅普诺夫指数预警，当系统轨迹从稳定吸引子走向混沌时触发预警。

🛠️ 技术栈与依赖 (Tech Stack)

# 核心环境
Python >= 3.8
torch >= 2.0.0      # 深度学习框架
ray[rllib]          # 深度强化学习框架 (用于DDPG/DQN/SAC)
transformers        # HuggingFace (用于时序Transformer)
torch_geometric     # 图神经网络 (GAT)

# 基线与数据处理
numpy, pandas, scipy
scikit-learn, xgboost
ruptures            # 基线突变点检测


📂 项目结构 (Directory Structure)

├── data/                  # 赛题原始数据 (附件1-5)
├── preprocess/            # 数据清洗、平滑、归一化模块
│   ├── ts_diffusion.py    # 基于扩散模型的高级插补
│   └── baseline_clean.py  # 传统填补与去噪
├── agents/                # 强化学习智能体目录
│   ├── ddpg_calibrator.py # 问题1：数据校正智能体
│   ├── dqn_segmenter.py   # 问题2：阶段识别切分智能体
│   └── sac_forecaster.py  # 问题4：爆破扰动自适应预测智能体
├── models/                # 深度学习模型目录
│   ├── st_gat.py          # 问题3：时空图注意力网络
│   └── tft_model.py       # 问题4：时序融合Transformer
├── baselines/             # 传统数学模型基准目录 (对比打分用)
├── notebooks/             # Jupyter探索性数据分析 (EDA) 与可视化
├── main.py                # 整体执行入口
└── README.md              # 项目文档


🚦 运行指南 (How to Run)

环境配置：

conda create -n slope_warning python=3.9
conda activate slope_warning
pip install -r requirements.txt


基线验证 (快速跑通，拿到保底数据)：

python main.py --mode baseline --task all


大炮模式启动 (训练深度/强化学习模型，需GPU)：

# 针对问题1训练校正智能体
python main.py --mode advanced --task 1 --use_gpu True

# 针对问题4训练包含爆破因子的 TFT+SAC 模型
python main.py --mode advanced --task 4 --epochs 100


📝 论文写作建议

由于使用的方法极其先进，在论文写作时必须着重解释“为什么这么做”（Why it works）。
例如：不要只说“我们用了强化学习”，而是要解释：“传统固定阈值无法应对多变的工程扰动（如爆破、电磁噪声），因此我们将阶段划分定义为马尔可夫决策过程（MDP），利用强化学习在时序环境中的探索机制，寻找最优的非线性切分策略，这比单纯的物理切线角法更加鲁棒。”