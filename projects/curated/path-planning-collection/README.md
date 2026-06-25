# 路径规划算法合集

多个路径规划变体项目的补充合集，展示 DQN、Q-Learning、多策略对比等不同路径规划方法的实现。

## 收录项目

### my_DQN — DQN 路径规划变体

```python
# dqn_variant.py:  DQN + 经验回放 + 目标网络
# env_variant.py:  自定义栅格环境 (含障碍物、起点、目标)
# network_variant.py: MLP 网络 (obs_dim → 128 → 128 → action_dim)
# replay_buffer.py: 经验回放缓冲区 (capacity=100000)
# map.py:          带障碍物地图
# main.py:         训练主循环 (ε-greedy 衰减、损失曲线)
```

### Q-Learning 韧性评估路径规划

```python
# q_learning_resilience.py
# 基于 Q-Learning 的韧性评估路径规划
# 将路径韧性指标纳入奖励函数设计
# Q 表更新: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
```

### 图书馆多维路径规划

```python
# library_multi_path.py (v5): 多策略路径搜索
#   - A* 最短路径
#   - Dijkstra 最短时间
#   - 综合权重路径 (距离 × 时间 × 拥挤度)
# library_path_v2.py (第二版): 双目标优化版本
#   - Pareto 前沿搜索: 同时优化路径长度和转弯次数

# 图层检查脚本: GIS 图层数据验证与可视化
```

## 技术栈

| 项目 | 算法 | 技术 |
|------|------|------|
| my_DQN | DQN | PyTorch MLP, Replay Buffer, ε-greedy |
| Q-Learning 韧性 | Q-Learning | Q-Table, 韧性奖励函数 |
| 图书馆多维 | A*/Dijkstra/多权重 | 多目标优化, GIS 图层 |

## 与主 curated 项目的关系

- [dqn-path-planning](../dqn-path-planning/) — 主 DQN 路径规划项目 (含 Tkinter GUI)
- [hybrid-a-star-path-planning](../hybrid-a-star-path-planning/) — Hybrid A* 泊车规划 (含 MATLAB)
- 本合集 — DQN/Q-Learning/A* 的变体实现与多策略对比

## 运行方式

```bash
pip install torch numpy matplotlib
cd path-planning-collection

# DQN 变体
python dqn_variant.py

# Q-Learning 韧性
python q_learning_resilience.py

# 图书馆多维路径
python library_multi_path.py
```

## 原始目录

- `my_DQN/`
- `基于韧性评估的路径规划/`
- `图书馆多维路径规划/`
- `ppo路径规划/`
- `改dqn/`
