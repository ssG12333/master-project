# PPO 二阶旋转倒立摆控制

## 项目简介

本项目自定义 Gymnasium 环境模拟二阶旋转倒立摆控制任务，并使用 Stable-Baselines3 的 PPO 算法进行训练和测试。项目适合展示强化学习控制、连续动作空间、物理仿真和可视化渲染能力。

原始目录：`二阶旋转倒立摆/`

## 技术栈

- Python
- Gymnasium
- Stable-Baselines3 PPO
- PyTorch
- NumPy, pygame
- tqdm, EvalCallback

## 主要功能

- 自定义 `RotaryDoublePendulumEnv` 强化学习环境。
- 定义连续状态空间和动作空间。
- 根据角度、角速度和控制动作设计奖励函数。
- 使用 PPO 训练倒立摆稳定控制策略。
- 支持 pygame 渲染和训练/测试流程。
- 使用 EvalCallback 保存最佳模型。

## 工作链路

1. 建立二阶旋转倒立摆动力学状态和动作定义。
2. 在 `step` 中计算状态导数、积分更新和奖励。
3. 使用 Gymnasium 接口封装环境。
4. 用 Stable-Baselines3 PPO 训练控制策略。
5. 通过评估回调保存最佳策略并记录训练进度。
6. 在测试阶段加载模型并渲染控制效果。

## 知识点

- 自定义 Gymnasium 环境。
- PPO 在连续控制问题中的应用。
- 倒立摆动力学建模。
- 强化学习奖励函数设计。
- pygame 物理过程可视化。

## 关键文件

- `ppo_rotary_pendulum.py`：环境定义、PPO 训练、评估回调和测试入口。

## 整理说明

- 已移除训练权重、TensorBoard 日志和评估缓存。
- 当前目录保留可复现控制任务逻辑的核心脚本。
