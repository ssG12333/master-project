# 👋 我的项目作品集

一个专注于深度学习和强化学习方向的研究生。这个仓库整理了我近两年完成的大部分项目，覆盖计算机视觉、强化学习、路径规划、时间序列预测、数学建模、NLP 和 Web 系统等多个方向。

> 📌 仓库里只放核心源码和说明文档，不传数据集、模型权重、训练输出和压缩包。完整工程文件保留在本地对应目录中。

---

## 📂 项目总览

```
计算机视觉 (YOLO系列)
├── 密集人群检测与密度监测          ← YOLOv8 + PyQt5 桌面应用
├── 肺结核/胸片疾病检测             ← YOLOv8 + CBAM 注意力 + PyQt5
├── FIRE-YOLOv5 火灾检测            ← YOLOv5 + CBAM/SE + TensorRT 推理加速
├── LPRNet 车牌识别                 ← 轻量 CNN + CTC 端到端序列识别
├── 医学图像二分类                   ← Swin Transformer + 肺部分割注意力
└── 农业害虫/疲劳驾驶/车道线/安全帽/山体滑坡/羽毛球识别  ← 已完成初步整理

强化学习 & 路径规划
├── DQN/PPO 奖励函数优化实验         ← AdamW + Advantage + 消融实验 (4组对比)
├── DQN 路径规划可视化系统           ← PyTorch ConvNet + Tkinter 实时渲染
├── PPO 二阶旋转倒立摆控制           ← Gymnasium 自定义环境 + 拉格朗日动力学
├── 智能仓储与多智能体任务分配        ← DQN + CBBA 分布式任务分配
├── Hybrid A* 泊车路径规划           ← Reeds-Shepp 曲线 + MATLAB 对比
├── 多智能体路径规划仿真 (MAPF)      ← CBS 冲突搜索 + A* + p5.js 在线 Demo
├── 多智能体 RL 算法合集             ← MASAC/MADDPG/MAPPO/IPPO/DQN 横向对比
├── 路径规划算法合集                 ← DQN 变体 + Q-Learning 韧性 + A*/Dijkstra
└── 强化学习对抗 (1v1)               ← 多智能体对抗框架

Web 管理系统
├── 医院管理系统                     ← Spring Boot + Vue2 + JWT + ECharts
└── 失能老年人照护服务系统            ← Spring Boot 4 + Java 21 + MyBatis

时间序列 & 数据建模
├── 通用时间序列预测框架             ← 9 模型 (Autoformer/Transformer/GRU/DLinear/AGCRN...)
├── 股票预测与回测分析               ← GRU + TwoBranch CNN + 技术指标 + 回测
├── 随机森林交通事故预测             ← RF + 特征编码 + 7步管线 (RMSE=16.02, R²=0.68)
├── 数学建模竞赛合集                 ← 2025/2026 校赛 + 2025 华为杯国赛 (3场)
└── 数据分析与可视化合集             ← BP 天气预测 + 音乐分析 + 学术图表

NLP & 其他
├── NLP 词义替换工具                 ← spaCy + WordNet + CEFR 六等级词汇映射
├── BiLSTM 情感分析模型             ← Embedding→BiLSTM(512)→Sigmoid + 8组消融
├── 区块链拜占庭共识仿真             ← Hash-DAC + Merkle Proof + Reed-Solomon 纠删码
└── Azul 棋盘游戏 DQN 智能体         ← 自对弈训练 + Minimax + α-β 剪枝 (3000局)
```

---

## 🧠 深度学习与计算机视觉

### YOLO 系列

这个方向是我做得最多的应用类项目。用 YOLO 在不同场景下做目标检测，然后封装 PyQt5 界面交付。

- **[密集人群检测与密度监测](projects/curated/yolo-crowd-density/README.md)** — YOLOv8 作为检测 backbone，前端用 PyQt5 写了完整的桌面应用。支持图片/视频/摄像头三种输入，实时统计画面人数并计算密度（人数/面积），人群密集时触发提醒。
- **[肺结核与胸片疾病检测](projects/curated/yolo-tuberculosis-detection/README.md)** — 用 YOLOv8 做胸部 X 光片的病灶检测，5 类疾病标签（肺结核、浸润、结节、空洞、积液），加了 CBAM 注意力模块提升对小病灶的敏感度。同样封装了 PyQt5 界面。
- **[FIRE-YOLOv5 火灾检测与边缘推理](projects/curated/yolo-fire-detection/README.md)** — 在 YOLOv5 基础上加了 CBAM 和 SE 两种注意力模块做对比，改进了 BiFPN 特征融合结构。最大的亮点是接入了 TensorRT 推理加速和串口通信（`/dev/ttyTHS1`, 115200 8N1），可以在 Jetson 边缘设备上部署，检测到火焰/烟雾后通过串口发信号。

除了这三个重点整理的，还有安全帽佩戴检测、山体滑坡识别、车道线检测、羽毛球动作识别、农业害虫检测等，都在本地原始目录里，后续会逐步整理上传。

### 其他视觉项目

- **[LPRNet 车牌识别](projects/curated/license-plate-recognition-lprnet/README.md)** — 端到端的车牌字符识别，不需要字符分割。核心是 LPRNet 的 small_basic_block（用 3×1 + 1×3 非对称卷积代替 3×3 标准卷积，参数量减 ~33%），多尺度特征通过 4 层跳跃连接收集后用 1×1 卷积融合，最后 CTC 解码输出中文字符序列（31 省级缩写 + 字母数字）。
- **[医学图像二分类](projects/curated/medical-image-classification/README.md)** — NIH Chest X-ray 数据集（112,120 张，14 种疾病标签）。backbone 用的 Swin Transformer，我自己加了两个注意力模块：`LungAttentionModule`（空间注意力，配合 DeepLabV3 生成的肺部 mask 抑制背景干扰）和 `LesionChannelAttention`（通道注意力，建模 5 类病灶的响应模式）。数据增强用了 Albumentations 的 CLAHE + Gamma + Brightness 组合，更适合医学图像。

---

## 🤖 强化学习与路径规划

强化学习是我研究生阶段的主攻方向，这块的项目最多也最深。

### 算法改进类

- **[DQN/PPO 奖励函数优化实验](projects/curated/rl-reward-optimization/README.md)** — 这是我做得比较深入的一个实验。在标准 DQN 和 PPO 的基础上做了两处改进：① 引入 Advantage 函数替代原始回报来降低方差；② 用 AdamW 替代 Adam（解耦权重衰减和自适应学习率）。做了 2×2 的消融实验（Value+Adam / Adv+Adam / Value+AdamW / Adv+AdamW），跑自动化脚本对比收敛速度和最终性能。DQN 用的是 `QNetwork(obs_dim→120→84→action_dim)`，PPO 的 Actor-Critic 各用两层 Tanh。

### 路径规划类

- **[DQN 路径规划可视化系统](projects/curated/dqn-path-planning/README.md)** — 11×11 栅格环境的路径规划。DQN 用了 ConvNet 结构（`Conv2d(2→16→32) → Linear(800→64→8)`），8 方向离散动作。用 Tkinter 写了 GUI，可以在界面上调参数（ε 衰减速度、学习率、batch size 等），实时看训练过程和路径变化。还加了 A* 作为对比基线。
- **[Hybrid A* 泊车路径规划](projects/curated/hybrid-a-star-path-planning/README.md)** — 3 种泊车场景（超车/U 型调头/直角转弯），在连续位姿空间 (x,y,θ) 搜索。启发式用的是 `max(A* 2D 距离, Reeds-Shepp 无碰撞长度)`，Reeds-Shepp 搜了全部 48 种路径类型（CSC/CCC/CCCC/CCSC/CCSCC）找最短。Python 和 MATLAB 都写了，方便对比验证。
- **[PPO 二阶旋转倒立摆控制](projects/curated/ppo-rotary-pendulum-control/README.md)** — 用 Gymnasium 从头写了一个二阶旋转倒立摆的物理环境，动力学用拉格朗日方程推导的 3×3 质量-惯性矩阵 M(q)，数值积分用的 RK4。PPO 配置：`lr=3e-4, n_steps=2048, batch=512, 10 epochs`，policy 和 value 各用两层 [128,128] 全连接。reward 函数专门设计了 5 个惩罚项来约束摆杆角度和小车位移。

### 多智能体类

- **[智能仓储与多智能体任务分配](projects/curated/warehouse-rl-scheduling/README.md)** — 10×10 仓储栅格环境，支持最多 6 个 AGV 同时运行。DQN 做 CTDE（集中训练分散执行），另外用 CBBA（Consensus-Based Bundle Algorithm）做分布式任务分配。UI 用 ttkbootstrap + matplotlib 做的，训练线程和 UI 线程分离，实时渲染 AGV 运动轨迹。
- **[多智能体路径规划仿真 (MAPF/CBS)](projects/curated/multi-agent-path-finding/README.md)** — 我的本科毕设。核心算法是 CBS（Conflict-Based Search），在高层 CT Tree 上检测冲突后分裂约束节点，低层用 A* 带约束重规划。V2 优化了 Cardinal Conflict 优先选择和 BP（Bypass）假分裂机制。前端用 p5.js 写的，部署了在线 Demo。
- **[多智能体 RL 算法合集](projects/curated/multi-agent-rl-collection/README.md)** — 把 MASAC、MADDPG、MAPPO、IPPO、DQN 五种算法在同一个多智能体路径规划环境里做了横向对比。MASAC 的 Actor 输出 μ 和 log_σ（LOG_SIG_MAX=2, LOG_SIG_MIN=-20），Critic 是集中式的（输入全局状态 + 所有动作）。都用的是 ReplayBuffer + 目标网络软更新。

---

## 🌐 Web 管理系统

- **[医院管理系统](projects/curated/hospital-management-system/README.md)** — 前后端分离的 HIS 系统，后端 Spring Boot + MyBatis Plus + MySQL，前端 Vue2 + Element UI。覆盖管理员/医生/患者三类角色，模块包括挂号预约、医生排班、床位管理、药品库存、检查项目和订单统计。认证用的 JWT，统计面板用 ECharts 做了就诊量折线图、科室占比饼图。Controller 层写了 10+ 个模块的完整 REST API。
- **[失能老年人照护服务系统](projects/curated/elderly-care-system/README.md)** — Spring Boot 4 + Java 21 + MyBatis，算是对新技术栈的一次尝试。业务流程是：老人/家属注册 → 填写健康档案 → 浏览照护服务项目 → 下单 → 护工接单 → 上门服务 → 确认完成。Controller 有 User、HealthRecord、ServiceItem、ServiceOrder、Recommendation 五个模块。

---

## 📊 时间序列、预测与数学建模

- **[通用时间序列预测框架](projects/curated/time-series-forecasting/README.md)** — 集成了 9 个主流时序预测模型（Autoformer、Transformer、DLinear、GraphPatchTST、GRU、CNN1D、Mamba 等），统一的数据加载接口支持 ETTh1/2、ETTm1/2、M4、UEA、PSM、SWAT 等 12 个数据集。实验框架用 `Exp_Basic → Exp_Long_Term_Forecast` 继承设计，optimizer 固定 Adam，loss 固定 MSE。
- **[股票预测与回测分析](projects/curated/stock-forecasting/README.md)** — 核心模型是 `TwoBranchGRU`（GRU 分支 hidden=64×2 层 + MLP 分支融合）和 `EnhancedCNN with SEBlock`（Conv1d 32→64→128 + SE 通道注意力）。特征工程做了 9 个 TA-Lib 技术指标（KAMA、EMA、MACD、RSI、ROC、CMO、ATR、CCI、BBANDS），滚动训练窗口 30 天。回测用 Top-K=10 选股，计算 Sharpe 比率（rf=2%）和最大回撤。
- **[随机森林交通事故预测](projects/curated/accident-random-forest-forecasting/README.md)** — 7 步管线：读取 → 预处理（fillna、时间特征提取）→ 按天聚合 → 补全日期范围 → 标签编码 + 比例加权 → 训练/测试划分（2023-08-01 为界）→ 预测。7 个特征里 4 个是类别特征的比例加权编码。最终 RMSE=16.02、MAE=12.64、R²=0.68。
- **[数学建模竞赛合集](projects/curated/math-modeling-competitions/README.md)** — 我参加的三次数学建模竞赛的完整代码和论文：
  - **2026 校赛 C 题（边坡监测）**: 5 问全链路，每问都做了 Baseline + Advanced 双轨方案。Q1 用 CEEMDAN 分解 + Ridge 回归做传感器校正（CV R²=0.9542），Q2 用 Hampel+SG 滤波 + PELT 变点检测做三阶段形变识别，Q3 用 MissForest 插补 + IsolationForest + SHAP 做多源关联分析，Q4 用 K-Means + VotingRegressor(GBDT+RF+ET) 做分阶段预测，Q5 用穷举搜索 C(6,5) + LSTM-Attention 做特征寻优 + 四级预警。
  - **2025 华为杯国赛 E 题（光纤校准）**: 16 路传感器的 32kHz 高频信号处理，小波去噪 + FFT 频域分析 + 聚类分组 + Stacking 集成分类器（LR+RF+SVC+XGBoost+LGB）。
  - **2025 校赛 C 题（博主预测）**: MLP(256→128) + LSTM(hidden=128) + Transformer(d_model=128, nhead=8) 三模型级联融合。

---

## 📝 NLP 与其他

- **[NLP 词义替换工具](projects/curated/nlp-word-substitution/README.md)** — 输入英文句子 + 源 CEFR 等级 + 目标 CEFR 等级，自动把词汇替换成对应难度的近义词。管道：spaCy 分词+词性标注 → WordNet 查近义词 → CEFR 等级过滤 → pyinflect 词形还原。
- **[BiLSTM 情感分析](projects/curated/sentiment-analysis/README.md)** — Embedding(256) → BiLSTM(512, bidirectional) → Linear(1024→1) → Sigmoid。做了 8 组消融实验，分别对比了 lr=[0.005, 0.0005]、LSTM 层数=[1, 2]、batch=[64, 128] 的组合效果。
- **[区块链拜占庭共识仿真](projects/curated/blockchain-byzantine-consensus/README.md)** — Hash-DAC 多值拜占庭共识协议，CandidateID = (proposer, value_hash, merkle_root, data_len, N, f, threshold)。Reed-Solomon 纠删码在 GF(257) 上做编码和拉格朗日插值恢复，Merkle Tree 做数据可用性证明。同时写了 PBFT 三阶段共识做对比。
- **[Azul 棋盘游戏 DQN 智能体](projects/curated/azul-rl-agent/README.md)** — COMP90054 AI Planning 课程项目。Azul 是策略棋盘游戏，我做了 DQN 自对弈训练（100 轮 × 3 对手 × 10 局 = 3000 局），对手包括随机策略和 Minimax + α-β 剪枝。游戏引擎是课程提供的，我的工作主要在智能体策略设计和训练流程。

---

## 🔧 技术栈总览

| 领域 | 常用技术 |
|------|---------|
| **深度学习框架** | PyTorch, torchvision, Stable-Baselines3, Gymnasium, Ray/RLlib |
| **计算机视觉** | YOLOv5/v8, OpenCV, PIL, Albumentations, TensorRT |
| **强化学习** | DQN, PPO, SAC, DDPG, MADDPG, MASAC, IPPO, MAPPO |
| **路径规划** | A*, Hybrid A*, CBS, Reeds-Shepp, PELT |
| **传统 ML** | scikit-learn (Ridge/SVR/RF/GBDT/XGBoost), scipy, statsmodels |
| **信号处理** | PyEMD (CEEMDAN), PyWavelets, scipy.signal (Savitzky-Golay, FFT) |
| **NLP** | spaCy, NLTK, WordNet, pyinflect, BiLSTM |
| **后端** | Spring Boot, MyBatis/MyBatis Plus, MySQL, JWT |
| **前端** | Vue2/3, Element UI, PyQt5, Tkinter, p5.js, ECharts |
| **工具链** | Git, Maven, Docker, matplotlib, seaborn, pandas, numpy |

---

## 📁 仓库结构

```
/
├── README.md                     ← 你正在看的这个文件
├── projects/
│   ├── curated/                  ← 24 个精选项目 (核心源码 + 详细 README)
│   └── from-zips/               ← 从 zip 解压的补充项目
├── portfolio-projects/           ← 各项目的作品集口径 README 草稿
├── docs/                         ← 项目清单、整理记录、模板
├── [原始项目目录]/               ← 60+ 个本地工程目录 (含完整数据/输出)
└── .gitignore                    ← 忽略大文件 (zip/weights/data/runs/...)
```

---

## 📎 相关链接

- **GitHub**: [项目仓库](https://github.com/)
- [项目总清单](docs/PROJECTS.md) — 60+ 个本地目录的完整索引
- [Zip 项目检查与解压记录](docs/ZIP_PROJECTS.md) — 从 zip 补充的项目
- [精选项目源码目录](projects/curated/README.md) — 24 个 curated 项目列表

---

*持续整理中，后续会逐步把本地剩下的项目（YOLO 应用、路径规划变体等）补全。*
