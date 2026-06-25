# DQN 路径规划与可视化系统

## 项目简介

本项目基于 DQN (Deep Q-Network) 实现二维栅格环境中的路径规划，结合卷积神经网络从栅格状态中提取空间特征，并通过 Tkinter GUI 提供实时训练监控、参数调整和 A* 算法对比。项目完整展示了强化学习环境建模、卷积 DQN、经验回放、目标网络机制，以及学习型路径规划与经典搜索算法的对比分析。

原始目录：`DQN路径规划/`

## 技术栈

- Python, PyTorch
- DQN, ConvNet, Replay Buffer, Target Network
- NumPy, matplotlib
- Tkinter GUI (ttk, ScrolledText, FigureCanvasTkAgg)
- A* 搜索 (heapq 优先队列)

## 环境设计 (env.py)

### Environment 类

```python
class Environment(object):
    def __init__(self, initial_position, target_position, X_max, Y_max, num_actions):
```

**物理空间**: 100 x 100 的二维平面, 离散化为 11 x 11 的栅格 (每格 10 单位)。

**状态表示**: 2 通道 11 x 11 的栅格张量:
- 通道 0: 智能体当前位置 (one-hot 编码)
- 通道 1: 障碍物和终点位置 (障碍物为 1, 终点为 1)

**动作空间**: 8 个离散动作, 每步移动 10 个单位:

```python
self.actionspace = {
    0: [v, 0],   # 右
    1: [0, v],   # 上
    2: [-v, 0],  # 左
    3: [0, -v],  # 下
    4: [-v, v],  # 左上
    5: [-v, -v], # 左下
    6: [v, v],   # 右上
    7: [v, -v],  # 右下
}
```

**障碍物配置**: 默认地图包含 14 个障碍物, 分布在指定栅格坐标:
```python
Obstacle_x = [3,3,3,3,3,3,3,7,7,7,7,7,7,7]
Obstacle_y = [4,5,6,7,8,9,10,0,1,2,3,4,5,6]
```
起点: (10, 0) 对应栅格 (1, 10), 终点: (90, 100) 对应栅格 (9, 0)。

### 奖励函数

```python
def get_reward(self, state, action):
    if self.is_collision(state):
        reward = -20      # 碰撞惩罚
    elif action in [0,1,2,3]:
        reward = -1       # 直行步长惩罚
    else:
        reward = -1.5     # 对角步长惩罚 (更大)
    if self.doneType == 1:
        reward = 20       # 到达终点奖励
```

- 碰撞障碍物: -20 (并回退到上一步位置)
- 直行动作 (0-3): -1 每步
- 对角动作 (4-7): -1.5 每步 (鼓励直行)
- 到达终点: +20
- 超出最大步数 (5000): 额外 -20

### 终止条件

```python
def isTerminal(self):
    Distance2Terminal = np.linalg.norm(
        np.subtract(self.vector_agentState, self.Terminal))
    if Distance2Terminal ** 0.5 == 0:  # 精确位置匹配
        return True
    return False
```

碰撞检测基于轴对齐包围盒 (AABB):
```python
def is_collision(self, state):
    delta = 0.5 * obstacle_width  # delta = 5
    for (x, y, w, h) in self.obstacle:
        if 0 <= state[0] - (x - delta) <= w and 0 <= state[1] - (y - delta) <= h:
            return True
    return False
```

## DQN 网络架构 (Agent.py)

### 卷积 Q 网络

```python
class DQN(nn.Module):
    def __init__(self, state_space_dim, action_space_dim, hidden):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(state_space_dim, 16, 4, 1)   # 输入 2ch, 输出 16ch, 4x4 核
        self.conv2 = nn.Conv2d(16, 32, 4, 1)                # 16ch -> 32ch, 4x4 核
        self.fc1 = nn.Linear(5 * 5 * 32, 64)                # 展平后 800 -> 64
        self.fc4 = nn.Linear(64, action_space_dim)           # 64 -> 8 (动作数)
```

网络层细节:
| 层 | 输入 | 输出 | 核/大小 | 激活 |
|----|------|------|---------|------|
| Conv2d | 2 x 11 x 11 | 16 x 8 x 8 | 4x4, stride=1 | ReLU |
| Conv2d | 16 x 8 x 8 | 32 x 5 x 5 | 4x4, stride=1 | ReLU |
| Flatten | 32 x 5 x 5 | 800 | - | - |
| Linear | 800 | 64 | - | ReLU |
| Linear | 64 | 8 | - | 无 |

### DQNAgent 类

```python
class DQNAgent(object):
    def __init__(self, state_space, n_actions, replay_buffer_size, batch_size, hidden_size, gamma):
        self.policy_net = DQN(state_space, n_actions, hidden_size)   # 策略网络
        self.target_net = DQN(state_space, n_actions, hidden_size)   # 目标网络
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.memory = ReplayMemory(replay_buffer_size)
```

### 经验回放 (ReplayMemory)

```python
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity  # 100000
        self.memory = []
        self.position = 0

    def push(self, *args):       # 存储 Transition
    def sample(self, batch_size): # 随机采样
```

Transition 结构:
```python
Transition = namedtuple('Transition',
    ('state', 'action', 'next_state', 'reward', 'done'))
```

### 网络更新 (Double DQN 风格)

```python
def _do_network_update(self):
    # 使用 policy_net 选择最优动作
    policy_actions = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
    # 使用 target_net 评估动作价值
    next_state_values[non_final_mask] = self.target_net(
        non_final_next_states).gather(1, policy_actions).squeeze().detach()

    expected_state_action_values = reward_batch + self.gamma * next_state_values
    loss = F.smooth_l1_loss(state_action_values.squeeze(), expected_state_action_values)
```

损失函数: **Huber loss** (smooth L1), 梯度裁剪到 [-0.1, 0.1].

### Epsilon-Greedy 动作选择

```python
def get_action(self, state, epsilon):
    if random.random() > epsilon:
        q_values = self.policy_net(state)
        return torch.argmax(q_values).item()  # 贪婪选择
    else:
        return random.randrange(self.n_actions)  # 随机探索
```

## 超参数配置 (DQN.py)

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_episodes` | 300 | 总训练轮数 |
| `hidden` | 128 | 网络隐藏层维度 (未使用, 实际为 64) |
| `gamma` | 0.99 | 折扣因子 |
| `replay_buffer_size` | 100000 | 经验回放容量 |
| `batch_size` | 256 | 训练批次大小 |
| `initial_epsilon` | 0.6 | 初始探索率 |
| `epsilon_stop` | 0.1 | 最小探索率 |
| `epsilon_decay_start` | 0 | 开始衰减轮次 |
| `epsilon_decay_end` | 150 | 结束衰减轮次 (= num_episodes // 2) |
| `TARGET_UPDATE` | 3 | 目标网络更新间隔 (轮次) |
| `learning_rate` | 2e-3 (Adam) | 优化器学习率 (GUI 可调) |
| `max_episode_steps` | 5000 | 每轮最大步数 |

Epsilon 衰减策略:
```python
epsilon_decaying = (epsilon - eps_stop) / (End_epsilon_decaying - Start_epsilon_decaying)
current_epsilon = max(eps_stop, epsilon - ep * epsilon_decaying)
# 从 0.6 线性衰减到 0.1, 持续 150 轮
```

## A* 基线对比

GUI 内置 A* 搜索用于路径可行性验证和对比:

```python
def astar_path(self, start, target, obstacles):
    # 启发式: 曼哈顿距离
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    # 8 方向搜索, 移动代价: 直行 1.0, 对角 1.4
    directions = [
        (0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0),
        (1, 1, 1.4), (1, -1, 1.4), (-1, 1, 1.4), (-1, -1, 1.4)
    ]
    queue = [(0, start_grid, [start_grid])]  # (f_cost, position, path)
    # 使用 heapq 优先队列, 按 f(n) = g(n) + h(n) 排序
```

A* 使用曼哈顿距离启发式函数, 支持 8 方向移动, 对角线代价为 sqrt(2) ~~ 1.4。测试时, DQN 路径 (epsilon=0.0 贪婪) 与 A* 路径同时显示在地图上供直观对比。

## GUI 功能 (DQN_GUI)

Tkinter 图形界面提供以下模块:

- **参数面板**: 可拖动滑块实时调整 Epsilon (0.01~1.0), Gamma (0.0~1.0), 学习率 (1e-5~1e-2)
- **控制按钮**: 更换地图 (随机生成可达障碍物布局), 开始/停止训练, 加载模型, 测试 (含 A* 对比), 重置, 保存
- **地图显示**: matplotlib 实时渲染栅格地图、障碍物 (蓝色方块)、起点 (红色)、终点 (绿色)、DQN 路径 (蓝色实线)、A* 路径 (橙色虚线)
- **训练曲线**: 累计奖励和每轮步数实时更新
- **终端输出**: 训练日志, 含轮次、奖励、步数、最短/最长路径信息
- **路径可达性检测**: 地图随机生成时通过 BFS 验证起点到终点连通性

```python
def change_map(self):
    # 随机生成 12 个障碍物 + 随机终点
    # 通过 BFS 验证路径可达性, 最多尝试 100 次
    def is_path_clear(self, start, target, obstacles):
        # 使用 deque BFS 搜索, 八方向
```

## 关键文件

```
dqn-path-planning/
├── env.py       # 环境类 Environment (状态、动作、奖励、碰撞检测、终止判断)
├── Agent.py     # DQN 网络、DQNAgent、经验回放 ReplayMemory
├── DQN.py       # 训练入口、Tkinter GUI、A* 对比、可视化
└── README.md    # 本文件
```

## 运行方式

```bash
python DQN.py
```

启动后:
1. 点击 "开始训练" 启动 DQN 训练 (300 轮)
2. 训练过程中可在参数面板实时调整 epsilon/gamma/学习率
3. 训练完成后点击 "开始测试" 对比 DQN 与 A* 路径
4. 点击 "更换地图" 随机生成新的障碍物布局
5. 点击 "保存" 保存模型权重、训练曲线和最终路径

## 知识点

- 卷积 DQN 处理栅格状态表示 (空间特征提取)
- 经验回放机制 (ReplayMemory) 打破序列相关性
- 目标网络 (Target Network) 稳定训练 + Double DQN 风格更新
- Huber Loss 在 Q 学习中的应用
- Epsilon 衰减探索策略
- 路径规划奖励函数设计 (步长惩罚 + 碰撞惩罚 + 终点奖励)
- A\* 搜索与强化学习路径规划的性能对比
- Tkinter + matplotlib 实时可视化训练过程
- BFS 路径可达性验证
