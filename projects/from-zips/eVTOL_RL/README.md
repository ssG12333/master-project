# eVTOL 强化学习控制

## 技术方向

强化学习、飞行器控制、轨迹规划、连续动作空间控制。

## 技术栈

- Python
- PyTorch
- TD3 / Actor-Critic
- 自定义 eVTOL 环境
- 经验回放、课程学习
- Matplotlib 可视化

## 工作链路

1. 在 `envs/` 中构建 eVTOL 动力学、风场和障碍物环境。
2. 定义状态空间、动作空间、奖励函数和终止条件。
3. 使用 `agents/td3.py` 和 `networks/` 中的 Actor/Critic 网络训练智能体。
4. 通过 `train.py` 执行训练，使用回放池提升样本利用率。
5. 使用 `evaluate.py` 和可视化脚本评估轨迹、安全性和收敛效果。

## 关键内容

- `envs/evtol_env.py`：环境封装。
- `envs/dynamics.py`：动力学相关逻辑。
- `agents/td3.py`：TD3 智能体。
- `networks/actor.py`, `networks/critic.py`：策略网络和价值网络。
- `utils/replay_buffer.py`：经验回放。

## 后续整理

- 修正入口文件命名，例如 `mian.py`。
- 补充训练曲线和轨迹图。
- 移除环境缓存，固定最小可运行配置。

