# 强化学习路径规划

## 技术方向

强化学习、路径规划、迷宫环境、多算法对比。

## 技术栈

- Python
- PyTorch
- Pygame
- NumPy
- Matplotlib
- Q-learning, DQN, DDPG, A* 搜索

## 工作链路

1. 使用 `maze.py` 构建网格迷宫环境。
2. 在 `state.py` 和 `player.py` 中定义状态、动作和智能体移动。
3. 分别训练 Q-learning、DQN、DDPG 等算法。
4. 使用 A* 作为传统路径规划对照。
5. 输出奖励、损失、步数和 Q 值变化曲线。
6. 对比不同算法的收敛速度和路径质量。

## 关键内容

- `QLearning.py`, `DQN.py`, `DDPG.py`：强化学习算法。
- `Apathfinding.py`：A* 路径搜索。
- `main.py`：运行入口。
- `reward_*.png`, `loss_*.png`, `steps_*.png`：实验结果。

## 后续整理

- 合并 `原版/` 与当前版本差异说明。
- 增加统一配置文件和复现实验命令。

