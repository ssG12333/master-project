# 多智能体强化学习算法合集

SAC / PPO / MADDPG / MAPPO / MASAC / IPPO 多算法多智能体路径规划对比实验合集。

## 收录算法

### MASAC (Multi-Agent Soft Actor-Critic)

```python
# train_masac.py + masac_network.py
# 集中训练分散执行 (CTDE) 架构
# ReplayBuffer: deque(maxlen=capacity), 存储 (local_state, global_state, action, reward, ...)
#
# MASACActor: 策略网络 → 输出 μ, σ → Tanh(μ + ε·σ) (LOG_SIG_MAX=2, LOG_SIG_MIN=-20)
# MASACCritic: 集中式 Q 网络 → Q(global_state, all_actions)
# 温度参数 α: 自动调节探索/利用平衡

class MASACActor(nn.Module):
    # local_obs → FC(256) → ReLU → FC(256) → ReLU → FC(action_dim) ×2 (μ, log_σ)

class MASACCritic(nn.Module):
    # global_state + all_actions → FC(256) → ReLU → FC(256) → ReLU → FC(1)
```

### MADDPG (Multi-Agent Deep Deterministic Policy Gradient)

```python
# train_maddpg.py + maddpg_network.py
# DDPG 的多智能体扩展
# 每个 agent 有独立的 Actor (确定策略) 和 Critic (集中式)
# 目标网络软更新: θ_target ← τ·θ + (1-τ)·θ_target
```

### PPO / IPPO (Independent PPO)

```python
# train_ppo.py + ppo_network.py  → 标准 PPO (单智能体)
# train_ippo.py + ippo_network.py → IPPO (独立 PPO, 多智能体)
# PPO Clip: L = min(ratio·A, clip(ratio, 1-ε, 1+ε)·A)
# GAE 优势估计
```

### DQN (Baseline)

```python
# train_dqn.py + dqn_network.py
# 标准 DQN + 经验回放 + 目标网络
# ε-greedy 探索策略
```

## 环境 (`environment.py`)

```python
class MultiAgentPathPlanningEnv:
    """多智能体路径规划环境
    MAP_BOUNDS: 地图边界
    OBSTACLES: 障碍物列表
    PURE_WALKABLE_COORDS: 可行走坐标
    GOAL_COORDS: 各 agent 目标位置
    AGENT_RADIUS: 智能体碰撞半径

    状态空间:
      local_state:  (agent_x, agent_y, goal_x, goal_y, 周围障碍物距离)
      global_state: (所有 agent 位置 + 所有 goal 位置 + 障碍物地图)

    动作空间: 连续 (dx, dy) 或 离散 (8 方向)

    奖励函数:
      - 到达目标: +10
      - 碰撞 (障碍物/其他 agent): -5
      - 每步: -0.1 (鼓励最短路径)
      - 接近目标: +dist_reward
    """
```

## 技术栈

| 算法 | 类型 | 关键特征 |
|------|------|---------|
| MASAC | Off-policy, CTDE | 最大熵, 自动温度调节, 连续动作 |
| MADDPG | Off-policy, CTDE | 确定策略, 集中 Critic |
| MAPPO | On-policy, CTDE | PPO Clip, GAE |
| IPPO | On-policy, Independent | 独立 PPO, 无参数共享 |
| DQN | Off-policy | 离散动作, 经验回放 |

## 相关对比

此合集与工作区中其他强化学习项目的关系:

| 项目 | 特点 |
|------|------|
| [rl-reward-optimization](../rl-reward-optimization/) | DQN/PPO 奖励函数优化 + AdamW/Advantage 消融 |
| [dqn-path-planning](../dqn-path-planning/) | DQN 单智能体路径规划 + Tkinter GUI |
| [warehouse-rl-scheduling](../warehouse-rl-scheduling/) | DQN 仓储多 AGV + CBBA 任务分配 |
| **本合集** | SAC/MADDPG/MAPPO/MASAC 多算法横向对比 |

## 运行方式

```bash
pip install torch numpy pandas matplotlib tqdm
cd multi-agent-rl-collection

# MASAC 训练
python train_masac.py

# MADDPG 训练
python train_maddpg.py

# PPO 训练
python train_ppo.py

# IPPO 训练
python train_ippo.py
```

## 原始目录

`D:\010\master\code\dqn\`
