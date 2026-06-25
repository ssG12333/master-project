# DQN/PPO 奖励函数优化实验

## 项目简介

本项目基于强化学习中的奖励函数发现与优化问题，围绕 DQN 和 PPO 算法进行了系统性改进与对比实验。核心贡献包括：(1) 使用 **Advantage 函数** (A(s,a) = Q(s,a) - V(s)) 替代原始 Q 值作为策略梯度权重，以降低梯度估计方差；(2) 将 **Adam 优化器替换为 AdamW**，通过解耦权重衰减提升训练稳定性；(3) 设计了完整的自动化实验框架和 **2x2 消融实验**，分离评估两项改进的独立贡献。

原始目录：`DQNPPO优化损失函数/Discovery-of-Optimal-Reward-function-main/`

## 技术栈

- Python 3.8+, PyTorch, Gymnasium
- DQN, PPO (CleanRL 风格实现)
- Stable-Baselines3 (ReplayBuffer)
- TensorBoard, tyro (CLI 参数解析)
- tqdm, matplotlib, seaborn, numpy

## 网络架构

### DQN QNetwork

```python
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 120),   # obs_dim = env.observation_space.shape
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_dim), # action_dim = env.action_space.n
        )

    def forward(self, x):
        return self.network(x)
```

- **输入**: 展平后的状态向量 (CartPole: 4, LunarLander: 8)
- **输出**: 每个离散动作的 Q 值
- **激活函数**: ReLU (隐藏层), 无 (输出层)

### PPO Actor-Critic

```python
class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
```

- **Actor** (策略网络): 输出离散动作 logits, 经 `Categorical` 分布采样
- **Critic** (值函数网络): 输出标量状态值 V(s)
- **初始化**: 正交初始化 (`torch.nn.init.orthogonal_`), Actor 最后一层 std=0.01

### ValueNetwork (改进版 DQN Ours)

与 QNetwork 结构相同, 但输出维度为 1 (标量 V(s)), 用于 Advantage 计算:

```python
class ValueNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(), nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, 1),  # 标量输出
        )
```

### 奖励函数网络 (Reward Model)

奖励函数由编码器-前向模型构成:

```python
# StateEncoder: 3层 MLP (input_dim -> hidden_dim -> hidden_dim -> encode_dim)
# ActionEncoder: 3层 MLP (action_dim -> hidden_dim -> hidden_dim -> encode_dim)
# ForwardModel: Linear(encode_dim * 2, 1)
class Reward(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, encode_dim=64):
        self.state_encoder = StateEncoder(state_dim, hidden_dim, encode_dim)
        self.action_encoder = ActionEncoder(action_dim, hidden_dim, encode_dim)
        self.forward_model = ForwardModel(encode_dim, 1)
```

## 超参数

### DQN 超参数

| 参数 | 基线值 (dqn.py) | 改进值 (dqn_ours.py) |
|------|----------------|---------------------|
| `learning_rate` | 2.5e-4 | 2.5e-4 |
| `buffer_size` | 10000 | 10000 |
| `gamma` | 0.99 | 0.99 |
| `tau` | 1.0 | 1.0 |
| `target_network_frequency` | 500 | 500 |
| `batch_size` | 256 | 128 |
| `start_e` | 1.0 | 1.0 |
| `end_e` | 0.05 | 0.05 |
| `exploration_fraction` | 0.5 | 0.5 |
| `learning_starts` | 5000 | 10000 |
| `train_frequency` | 4 | 10 |
| `reward_frequency` | 2000 | 1000 |
| `hidden_dim` | 256 | 256 |
| `encode_dim` | 64 | 64 |
| 优化器 | Adam | Adam 或 AdamW (weight_decay=0.01) |

### PPO 超参数

| 参数 | 值 |
|------|-----|
| `learning_rate` | 2.5e-4 (线性退火) |
| `num_envs` | 4 (并行环境) |
| `num_steps` | 128 (每环境每轮采样步数) |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `num_minibatches` | 4 |
| `update_epochs` | 4 (K epochs) |
| `norm_adv` | True (优势归一化) |
| `clip_coef` | 0.2 |
| `clip_vloss` | True |
| `ent_coef` | 0.01 |
| `vf_coef` | 0.5 |
| `max_grad_norm` | 0.5 |
| `reward_frequency` | 1024 |

### 改进方案的额外参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `use_advantage` | True/False | 是否使用 Advantage 替代 Q 值 |
| `use_adamw` | True/False | 是否使用 AdamW 优化器 |
| `weight_decay` | 0.01 | AdamW 权重衰减系数 |
| `reward_lr` | 1e-4 | 奖励模型学习率 |
| `n_samples` | 10 | Advantage 估计的采样动作数 |

## 奖励函数设计

### 三种奖励类型

1. **sparse** (环境原生奖励): 使用 Gymnasium 环境的默认奖励信号 (CartPole: 每步 +1, LunarLander: 基于着陆质量)
2. **rmbo** (原始方法): 使用 `RewardFunction.observe_reward()` 通过学习的奖励模型 R(s,a) 生成奖励
3. **ours** (改进方法): 结合 rmbo 的奖励模型 + Advantage 函数 + AdamW 优化器

### Advantage 函数

改进核心是将策略梯度的权重从 Q(s,a) 替换为 Advantage 函数:

```
A(s,a) = Q(s,a) - V(s)
```

在 DQN 中, Advantage 通过 ValueNetwork 计算:
```python
advantage = q_values_current.gather(1, actions).squeeze() - v_values_current.squeeze()
```

在 PPO 中, 通过 `policy_weight = mb_advantages` (当 `use_advantage=True`) 替代原始的 `b_returns - b_values`.

### 奖励函数优化

奖励函数的训练基于双层优化框架:
- **内层**: Agent 在奖励函数 R(s,a) 下优化策略
- **外层**: 奖励函数通过 `optimize_reward()` 更新, 使 R(s,a) 与标准化回报 `overline_V` 对齐

```python
# 外层优化损失
accumulator_2 = prob_a * (overline_V - V(s))       # 价值差异
accumulator_1 = reward_hat - E[R(s, a)]              # 奖励预测误差
loss = mean(accumulator_2) * mean(accumulator_1)
```

## 实验设计

### 完整实验 (run_experiments.py)

- **2 个环境**: CartPole-v1 (500K 步), LunarLander-v2 (1M 步)
- **2 种算法**: DQN, PPO
- **3 种奖励类型**: sparse, rmbo, ours
- **3 个随机种子**: seed=1,2,3
- **总计**: 2 x 2 x 3 x 3 = **36 个独立实验**
- 结果保存至 `experiment_results/` 目录

### 消融实验 (run_ablation.py)

在 LunarLander-v2 + PPO 上运行 2x2 消融, 分离两项改进的独立贡献:

| 方法 | Advantage | AdamW | 说明 |
|------|-----------|-------|------|
| Value+Adam | No | No | 原始基线 |
| Adv+Adam | Yes | No | 仅优势函数 |
| Value+AdamW | No | Yes | 仅 AdamW |
| Adv+AdamW | Yes | Yes | 完整方法 (ours) |

- **每个配置 3 个种子**, 共 12 个实验
- 结果保存至 `ablation_results/` 目录

### 可视化 (plot_results.py)

生成四类图表:
1. **学习曲线**: 每个环境-算法组合的三种奖励类型对比 (含均值 + 标准差)
2. **最终性能柱状图**: DQN vs PPO 在两种环境下的最终回报对比
3. **消融学习曲线**: 四种配置在 LunarLander-v2 上的收敛过程
4. **消融结果柱状图**: 四种配置的最终回报对比

可视化使用指数移动平均平滑 (weight=0.7) 和数据插值对齐.

## 关键文件

```
rl-reward-optimization/
├── casestudy1/
│   ├── dqn.py               # 原始 DQN 基线 (CleanRL 风格, Adam 优化器, Q 值权重)
│   ├── dqn_ours.py          # 改进 DQN (AdamW + Advantage + ValueNetwork)
│   ├── ppo.py               # 原始 PPO 基线 (CleanRL 风格, GAE, Adam)
│   └── ppo_ours.py          # 改进 PPO (AdamW + Advantage 策略梯度权重)
├── utils/
│   ├── reward_machine.py    # 奖励函数类: RewardFunction (双层优化主体)
│   ├── reward_model.py      # 神经网络模块: Network, Critic, Reward, StateEncoder, ActionEncoder
│   └── agent.py             # 辅助智能体工具
├── run_experiments.py       # 批量实验脚本 (36 个实验)
├── run_ablation.py          # 消融实验脚本 (12 个实验)
├── plot_results.py          # 结果可视化脚本 (学习曲线 + 柱状图)
└── README.md                # 本文件
```

## 运行方式

```bash
# 安装依赖
pip install -r requirements/requirements.txt

# 运行完整实验 (36 个组合)
python run_experiments.py

# 运行消融实验 (12 个组合)
python run_ablation.py

# 生成可视化图表
python plot_results.py experiment_results ablation_results
```

## 知识点

- 强化学习价值函数 (Q(s,a)), 策略梯度, 优势函数 (A(s,a))
- DQN 与 PPO 训练流程 (经验回放, GAE, PPO-clip)
- 优化器选择对训练稳定性的影响 (Adam vs AdamW)
- 双层优化框架下的奖励函数学习
- 消融实验设计与统计显著性 (多种子评估)
- TensorBoard 日志解析与学习曲线可视化
