# 智能仓储与多智能体任务分配

## 项目简介

基于 DQN 强化学习的 AGV 仓储路径规划与多智能体任务分配系统，包含 Tkinter 可视化界面、多 AGV 协同调度和 CBBA 任务分配算法。

## 代码架构

### 仓储环境 (`agv_CTDEUI.py`)

```python
class Env:
    """AGV 仓储栅格环境
    栅格编码:
      0: 空地, 1: 障碍物/货架
      2,5,10-13: 目标点/货位 (黄色系)
      3,4,6-9: AGV 起点 (红/蓝/紫/橙/青/粉)
    最多支持 6 个 AGV 同时运行
    """
    def __init__(self):
        self.state = None           # 全局状态编码
        self.step_counter = 0
        self.agent_positions = []   # [(x,y), ...]
        self.target_positions = []  # [(x,y), ...]
        self.maze = None            # 10×10 np.array

    def reset(self): ...
    def step(self, actions): ...    # 多智能体并行动作
    def get_state(self): ...        # 返回全局状态
```

### 默认仓库布局 (10×10)

```
5 0 0 0 1 0 1 0 2 1     # 5=目标, 2=目标, 1=货架
0 0 0 0 1 0 1 0 0 1     # 0=过道
0 0 0 0 1 0 0 0 1 1
1 0 0 0 0 1 0 0 0 0
1 0 1 1 0 0 0 0 1 0
1 0 1 1 0 0 4 1 0 0     # 4=AGV2 起点(蓝)
0 0 0 0 1 0 1 0 0 0
0 0 0 0 0 0 1 1 0 1
1 1 3 1 1 1 1 0 0 1     # 3=AGV1 起点(红)
1 1 0 0 0 0 0 0 0 1
```

### DQN 网络

```python
class DQN(nn.Module):
    """DQN 网络: Linear(obs_dim, 128) → ReLU → Linear(128, 128) → ReLU → Linear(128, 5)
    动作空间: 上/下/左/右/停留 (5 actions)
    训练: replay buffer + target network + ε-greedy
    """
```

### Tkinter 可视化界面

- **ttkbootstrap** 主题 UI，地图编辑 + AGV 数量设置 + 训练控制
- **matplotlib FigureCanvasTkAgg** 嵌入 → 实时渲染 AGV 运动
- 颜色映射: 6 种 AGV 色 + 6 种目标色 + 路径半透明覆盖
- `threading.Thread` 后台运行 DQN 训练循环，前端实时刷新 reward 曲线

### CBBA 任务分配 (`task_allocation/`)

```python
# cbba_solver.py: Consensus-Based Bundle Algorithm
#   Phase 1 - Bundle Construction: 每个 agent 贪心构建任务包
#   Phase 2 - Consensus: agent 间交换出价信息解决冲突
#
# worker.py:  智能体类 (位置、bundle、winning_bids)
# task_env.py: 任务环境 (位置、时间窗)
# trainer.py: 训练循环 + 评估指标
# net.py:     策略网络 (Actor-Critic)
# parameters.py: 全局配置 (agent数量、任务数、通信半径)
```

## 技术栈

| 类别 | 技术 |
|------|------|
| 强化学习 | DQN (replay buffer, target network, ε-greedy decay) |
| 多智能体 | CTDE (集中训练分散执行), CBBA 分布式任务分配 |
| 网络架构 | PyTorch MLP (128→128→Action) |
| 可视化 | Tkinter + ttkbootstrap + matplotlib (FigureCanvasTkAgg) |
| 并发 | threading.Thread (训练/UI 分离) |

## 运行方式

```bash
pip install torch numpy matplotlib ttkbootstrap pillow
python agv_CTDEUI.py                  # AGV 仓储 DQN 可视化系统
python task_allocation/trainer.py     # CBBA 多智能体任务分配训练
```

## 关键文件

| 文件 | 说明 |
|------|------|
| `agv_CTDEUI.py` | DQN 仓储环境 + Tkinter 界面 + 实时渲染 |
| `task_allocation/cbba_solver.py` | CBBA 分布式任务分配求解器 |
| `task_allocation/worker.py` | 智能体类 (bundle, bids, 通信) |
| `task_allocation/trainer.py` | 训练循环与评估 |
| `plot_metrics.py` | 训练指标可视化曲线 |
