# 个人项目作品集

这个仓库用于整理近期完成的项目，目标是作为 GitHub 主页和作品集导航。当前只保留关键说明文档，不直接上传原始数据集、模型权重、运行结果、压缩包和重复框架源码。

## 项目分类

### 深度学习与计算机视觉

- [YOLO 密集人群检测与密度监测](projects/curated/yolo-crowd-density/README.md)
- [YOLO 肺结核与胸片疾病检测系统](projects/curated/yolo-tuberculosis-detection/README.md)
- [FIRE-YOLOv5 火灾检测与边缘推理](projects/curated/yolo-fire-detection/README.md)
- [LPRNet 车牌识别系统](projects/curated/license-plate-recognition-lprnet/README.md)
- [医学图像二分类与肺部注意力模型](projects/curated/medical-image-classification/README.md)
- 农业害虫检测、疲劳驾驶检测、车道线检测、安全帽预警、山体滑坡检测、羽毛球动作识别等项目已完成初步归类。

### 强化学习与路径规划

- [DQN/PPO 奖励函数优化实验](projects/curated/rl-reward-optimization/README.md)
- [智能仓储与多智能体任务分配](projects/curated/warehouse-rl-scheduling/README.md)
- [DQN 路径规划与可视化系统](projects/curated/dqn-path-planning/README.md)
- [Hybrid A* 泊车路径规划仿真](projects/curated/hybrid-a-star-path-planning/README.md)
- [PPO 二阶旋转倒立摆控制](projects/curated/ppo-rotary-pendulum-control/README.md)
- PPO 路径规划、韧性评估路径规划、图书馆多维路径规划等项目已归入算法实验类。

### Web 管理系统

- [医院管理系统](projects/curated/hospital-management-system/README.md)
- [失能老年人照护服务系统](projects/curated/elderly-care-system/README.md)

### 时间序列、预测与数据分析

- [通用时间序列预测框架](projects/curated/time-series-forecasting/README.md)
- [股票预测与回测分析](projects/curated/stock-forecasting/README.md)
- [随机森林交通事故预测](projects/curated/accident-random-forest-forecasting/README.md)
- 天气预测、音乐预测、数学建模、可视化绘图等项目已归入数据建模类。

### 自然语言处理与其他

- [自然语言处理词义替换工具](projects/curated/nlp-word-substitution/README.md)
- [BiLSTM 情感分析模型](projects/curated/sentiment-analysis/README.md)
- [区块链拜占庭共识仿真](projects/curated/blockchain-byzantine-consensus/README.md)
- 符合性验证工具等项目作为补充项目整理。

## 整理原则

- 只提交 README、项目索引和后续可复用的核心源码。
- 不提交 `.zip`、`data/`、`runs/`、`wandb/`、`weights/`、`*.pt`、`*.pth`、`*.onnx` 等大文件。
- 重复项目合并描述，例如 `1v1源代码 张志豪改` 与 `强化学习对抗` 归并为同一类强化学习对抗项目。
- 上游框架副本不作为独立作品展示，只展示在其基础上完成的任务、改进点和系统界面。

## 文档入口

- [项目总清单](docs/PROJECTS.md)
- [Zip 项目检查与解压记录](docs/ZIP_PROJECTS.md)
- [重构计划](docs/REFACTOR_PLAN.md)
- [项目 README 模板](docs/README_TEMPLATE.md)
- [精选项目源码目录](projects/curated/README.md)
- [从 zip 解压出的非重复项目](projects/from-zips/README.md)

## 已执行的整理动作

- 已链接远程仓库：`https://github.com/ssG12333/-.git`
- 已建立 Git 白名单，默认忽略原始大文件和重复资料。
- 已将非重复 zip 过滤解压到 `projects/from-zips/`，并二次裁剪第三方框架副本、训练数据和批量输出。
- 已将 18 个重点项目的核心源码和配置抽取到 `projects/curated/`。
- 已为重点项目和 zip 项目补充 README，重点突出技术栈、工作链路和技术方向。
