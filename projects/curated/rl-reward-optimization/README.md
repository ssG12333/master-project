# DQN/PPO 奖励函数优化实验

## 项目简介

本项目基于强化学习奖励函数发现与优化实验，围绕 DQN、PPO 等算法进行了改进和对比。核心工作包括引入 AdamW 优化器、使用 Advantage 函数改进策略梯度权重、增加自动化实验脚本，并通过消融实验验证不同改动的影响。

原始目录：`DQNPPO优化损失函数/Discovery-of-Optimal-Reward-function-main/`

## 技术栈

- Python, PyTorch
- Gymnasium
- DQN, PPO, SAC, TD3
- TensorBoard 日志解析
- tqdm, matplotlib, numpy

## 主要功能

- 实现 DQN/PPO 基线算法与改进版本。
- 支持 CartPole-v1、LunarLander-v2 等环境实验。
- 增加 AdamW 优化器配置。
- 使用 Advantage 替代部分 Q 值权重设计。
- 自动运行多环境、多算法、多奖励函数组合实验。
- 提供消融实验脚本和结果可视化脚本。

## 工作链路

1. 定义 Gymnasium 环境、状态空间、动作空间和奖励函数。
2. 运行 DQN/PPO/SAC/TD3 等基线算法。
3. 在改进版本中加入 AdamW 和 Advantage 相关配置。
4. 使用自动化脚本批量运行环境、算法、奖励函数和随机种子组合。
5. 解析训练日志，绘制学习曲线和消融实验图。

## 知识点

- 强化学习价值函数、策略梯度、优势函数。
- DQN 与 PPO 的训练流程。
- 优化器对训练稳定性的影响。
- 消融实验设计。
- 实验日志解析与可视化。

## 后续清理

- 统一 README 编码，修复乱码。
- 删除日志缓存和临时输出。
- 精简上游模板代码，只保留改动文件和实验脚本。
