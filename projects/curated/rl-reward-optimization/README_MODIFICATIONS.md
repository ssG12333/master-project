# 强化学习改进项目 - 修改说明

## 1. 项目概述

本项目对原始强化学习代码进行了以下改进：
- **优化器改进**：将Adam优化器替换为AdamW优化器
- **算法改进**：使用Advantage函数替代Q值作为策略梯度的权重
- **用户体验**：添加tqdm进度条和详细的每轮print输出
- **实验自动化**：创建完整的实验脚本和消融实验

## 2. 主要修改文件

### 2.1 核心算法文件

#### `casestudy1/dqn_ours.py`
- 添加了AdamW优化器支持
- 引入ValueNetwork计算Advantage
- 添加了详细的注释和文档
- 集成tqdm进度条
- 增加每轮训练的详细print输出

#### `casestudy1/ppo_ours.py`
- 添加了AdamW优化器支持
- 实现了Advantage函数计算
- 添加了详细的注释和文档
- 集成tqdm进度条
- 增加每轮训练的详细print输出

#### `utils/reward_machine.py`
- 增加了`use_advantage`和`use_adamw`参数
- 支持AdamW优化器配置
- 支持Advantage函数计算

### 2.2 实验脚本

#### `run_experiments.py`
- 自动化运行所有实验组合
- 支持2个环境（CartPole-v1, LunarLander-v2）
- 支持2个算法（DQN, PPO）
- 支持3种奖励函数（sparse, rmbo, ours）
- 集成tqdm进度条
- 详细的实验结果输出

#### `run_ablation.py`
- 运行消融实验（仅LunarLander-v2 + PPO）
- 测试4种配置：
  1. Value+Adam（原始基线）
  2. Adv+Adam（仅优势函数）
  3. Value+AdamW（仅AdamW）
  4. Adv+AdamW（完整方法）
- 集成tqdm进度条
- 详细的实验结果输出

#### `plot_results.py`
- 生成学习曲线（带平均值和标准差）
- 生成最终性能柱状图
- 支持消融实验结果可视化

## 3. 如何运行实验

### 3.1 安装依赖

```bash
pip install -r requirements.txt
```

### 3.2 运行完整实验

```bash
python run_experiments.py
```

这将运行：
- 2个环境 × 2个算法 × 3种奖励函数 × 3个种子 = 36个实验
- 结果保存在`experiment_results`目录

### 3.3 运行消融实验

```bash
python run_ablation.py
```

这将运行：
- LunarLander-v2环境 × PPO算法 × 4种配置 × 3个种子 = 12个实验
- 结果保存在`ablation_results`目录

### 3.4 生成图表

```bash
python plot_results.py
```

这将生成：
- DQN算法在2个任务上的学习曲线
- PPO算法在2个任务上的学习曲线
- 所有配置的最终性能柱状图
- 消融实验的对比图表

## 4. 改进说明

### 4.1 AdamW优化器

AdamW是Adam的改进版本，通过将权重衰减（weight decay）从梯度更新中分离出来，提高了模型的泛化能力和训练稳定性。

### 4.2 Advantage函数

Advantage函数定义为：A(s, a) = Q(s, a) - V(s)，其中：
- Q(s, a)是状态-动作价值函数
- V(s)是状态价值函数

使用Advantage函数作为策略梯度的权重，可以：
- 减少梯度估计的方差
- 提高训练稳定性
- 加速收敛

### 4.3 消融实验设计

通过2×2单变量消融，分离两项改进的独立作用：
- **优势函数消融**：Adv+AdamW vs Value+AdamW（固定AdamW，验证优势函数的方差降低效果）
- **优化器消融**：Adv+Adam vs Adv+AdamW（固定优势函数，验证AdamW的训练稳定效果）

## 5. 预期结果

- **加速收敛**：改进方法应该比基线方法更快达到高回报
- **最终性能**：改进方法应该达到更高的最终回报
- **稳定性**：改进方法的学习曲线应该更加平滑

## 6. 代码结构

```
Discovery-of-Optimal-Reward-function-main/
├── casestudy1/
│   ├── dqn.py              # 原始DQN实现
│   ├── dqn_ours.py         # 改进的DQN实现
│   ├── ppo.py              # 原始PPO实现
│   └── ppo_ours.py         # 改进的PPO实现
├── utils/
│   └── reward_machine.py   # 奖励函数实现
├── run_experiments.py      # 实验运行脚本
├── run_ablation.py         # 消融实验脚本
├── plot_results.py         # 结果可视化脚本
└── README_MODIFICATIONS.md # 本文件
```

## 7. 注意事项

- 所有实验使用默认参数运行
- 每个实验使用3个不同的随机种子以确保结果的统计显著性
- 实验结果会自动保存到对应的目录中
- 生成的图表可以直接用于论文

## 8. 联系方式

如有任何问题，请联系项目维护者。