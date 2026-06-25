# 项目总清单

这份清单基于当前本地目录的只读扫描结果整理，用于后续拆分 GitHub 仓库和补充项目 README。

## 推荐优先展示

| 项目 | 原始位置 | 类型 | 技术栈 | 当前定位 |
| --- | --- | --- | --- | --- |
| 医院管理系统 | `医院管理系统/` | Web 全栈 | Spring Boot, MyBatis Plus, MySQL, Vue2, Element UI, ECharts, JWT | 完整管理系统，适合作为 Web 项目重点展示 |
| 失能老年人照护服务系统 | `老年人项目全包/` | Web 后台 + 静态前端 | Spring Boot 4, Java 21, MyBatis, MySQL, Vue/Element 静态页 | 适合作为 Java 后端 CRUD 和业务建模项目 |
| DQN/PPO 奖励函数优化 | `DQNPPO优化损失函数/` | 强化学习实验 | PyTorch, Gymnasium, DQN, PPO, SAC, TD3, AdamW, Advantage | 算法改进、实验自动化、消融实验 |
| YOLO 密集人群检测 | `密集人群yolo/` | 目标检测应用 | YOLOv8, PyQt5, OpenCV, PIL | 人数统计、密度计算、图片/视频检测界面 |
| YOLO 肺结核检测 | `肺结核yolo/` | 医学影像检测 | YOLOv8, PyQt5, OpenCV, PIL | 胸片疾病检测和可视化界面 |
| FIRE-YOLOv5 | `FIRE-YOLOV5-master/` | 目标检测 + 推理部署 | YOLOv5, PyTorch, CBAM, SE, TensorRT, CUDA, OpenCV, serial | 火灾检测、注意力模块、串口通信、推理加速 |
| 智能仓储与任务分配 | `智能仓储/`, `多智能体任务动态分配/` | 强化学习/组合优化 | PyTorch, CBBA, 多智能体, Plotly, Matplotlib | 仓储路径、任务分配、对比实验和可视化 |
| 通用时间序列预测 | `forecast/` | 深度学习预测框架 | PyTorch, Autoformer, Transformer, GRU, AGCRN, GraphPatchTST | 多模型预测框架，适合展示模型工程能力 |
| 股票预测与回测 | `股票预测/` | 金融时间序列 | PyTorch, GRU, pandas, scikit-learn, statsmodels, seaborn | 特征工程、滚动训练、回测分析 |
| NLP 词义替换 | `自然语言处理词义替换/` | NLP 工具 | spaCy, NLTK, WordNet, pyinflect, pandas | 词性识别、词义替换、自动测试 |
| LPRNet 车牌识别 | `LPRNet/` | 计算机视觉/序列识别 | PyTorch, LPRNet, CTC, OpenCV, PIL | 车牌字符识别、CTC 解码、中文车牌字符集 |
| 医学图像二分类 | `医学图像二分类/` | 医学影像分类 | PyTorch, Swin Transformer, DeepLabV3, Albumentations, OpenCV | 肺部区域注意力、病灶通道注意力、分类评估 |
| BiLSTM 情感分析 | `情感分析/` | NLP 文本分类 | PyTorch, NLTK, BiLSTM, pandas, matplotlib | 文本分词、词表构建、情感分类训练流程 |
| DQN 路径规划 | `DQN路径规划/` | 强化学习路径规划 | PyTorch, DQN, Replay Buffer, Tkinter, A* | 栅格环境、路径规划、GUI 可视化、A* 对比 |
| Hybrid A* 泊车路径规划 | `改进a星路径规划/` | 传统路径规划/车辆运动学 | Python, MATLAB, Hybrid A*, Reeds-Shepp, matplotlib | 车辆约束路径搜索、多场景仿真和指标评估 |
| PPO 二阶旋转倒立摆 | `二阶旋转倒立摆/` | 强化学习控制 | Gymnasium, Stable-Baselines3, PPO, pygame, PyTorch | 自定义控制环境、PPO 训练、物理渲染 |
| 区块链拜占庭共识 | `区块链拜占庭/` | 分布式系统/区块链 | Python, PBFT/MVBC, Hash-DAC, Merkle Tree, matplotlib | 拜占庭节点仿真、数据可用性证明、性能统计 |
| 随机森林交通事故预测 | `随机森林事故预测/` | 机器学习预测 | pandas, scikit-learn, RandomForestRegressor, seaborn | 事故时间序列、类别编码、预测评估和可视化 |
| 数学建模竞赛合集 | `2025国赛数学建模/`, `2025校赛数学建模/`, `2026校赛数学建模/` | 数学建模竞赛 | Python, PyTorch, CEEMDAN, Ridge/SVR, RLlib, DQN/SAC/DDPG, TFT, GAT, XGBoost | 传感器数据校正、边坡监测、社交媒体预测、双轨方法论(Baseline+Advanced) |

## 可合并展示

| 合并主题 | 涉及目录 | 整合建议 |
| --- | --- | --- |
| YOLO 应用合集 | `260313yolo（复件）`, `ultralytics`, `yolo改`, zip 中的安全帽/车道线/山体滑坡/羽毛球识别 | 不上传完整框架，按应用场景写成多个 README 或一个 YOLO 专题页 |
| 路径规划算法合集 | `my_DQN`, `ppo路径规划`, `改dqn`, `基于韧性评估的路径规划`, `图书馆多维路径规划` | 已抽取 DQN 与 Hybrid A* 作为代表项目，其余按 DQN/PPO/Q-learning/A* 分章节整理 |
| 数据预测合集 | `天气预测`, `音乐预测`, `数学建模`, `画表` | 已抽取股票预测和事故预测作为代表项目，其余可整合成数据分析和建模案例集 |
| NLP 基础项目 | `情感分析`, `自然语言处理词义替换` | 已作为重点项目保留，后续可拆成 NLP 专题仓库 |

## 暂不建议直接上传

- 所有根目录 zip：多数是备份或未清理交付包，且体积过大。
- `data/`, `train/`, `valid/`, `test/`, `runs/`, `wandb/`, `weights/`：包含数据集、训练缓存、日志和模型权重。
- 上游框架完整副本：例如原始 `ultralytics`、YOLOv5/YOLOv8 框架源码，应只保留你修改和使用的部分。
- `.idea/`, `__pycache__/`, `target/`, `node_modules/`, `.venv/`：开发环境或构建产物。

## 已从 zip 补充的项目

已将非重复 zip 过滤解压到 `projects/from-zips/`，并补充了根 README。详情见 [Zip 项目检查与解压记录](ZIP_PROJECTS.md)。

重点新增方向包括：

- 车道线检测、安全帽检测、山体滑坡检测、羽毛球动作识别。
- eVTOL 强化学习控制、强化学习路径规划、强化学习对抗游戏。
- AS 路由网络分析、爬虫加深度学习预测分析、江西旅游分析。
- BOM 符合性验证工具、飞机路径建模、图像分割工具。
