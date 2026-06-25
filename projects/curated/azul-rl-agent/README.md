# Azul 棋盘游戏 DQN 智能体

COMP90054 AI Planning for Autonomous Systems 课程项目。为策略棋盘游戏 Azul 设计的基于 DQN 强化学习的竞技智能体。

## 游戏简介

Azul 是一款策略性桌面游戏，玩家轮流从工厂展示区选取彩色瓷砖，放置到自己的棋盘上完成图案行。计分规则基于相邻瓷砖的连锁得分，目标最大化总分。

- **计算时间**: 每步 1 秒限制，15 秒初始加载
- **锦标赛**: ELO 排名制 (胜 3 分，平 1 分)
- **对手**: 随机智能体、Minimax 智能体、同课程其他团队

## 代码架构

### 游戏引擎 (`azul_model.py`, `azul_utils.py`)

```python
# azul_model.py: 游戏状态机
#   - getLegalActions(): 当前合法动作列表
#   - generateSuccessor(action): 状态转移
#   - isGameOver(): 终局判定
#
# azul_utils.py: 常量定义
#   - 瓷砖颜色编码 (5 种)
#   - 工厂展示区结构
#   - 计分规则参数

# azul_displayer.py: Pygame GUI 渲染
```

### DQN 训练 (`train_dqn.py`)

```python
def main():
    my_agent = "agents.t_070.train"       # 自对弈训练
    opponents = [
        my_agent,                          # 自我对弈
        "agents.generic.random",           # 随机基线
        "agents.mmAgent"                   # Minimax 对手
    ]
    num_iterations = 100                   # 训练迭代
    games_per_iteration = 10               # 每轮游戏数

    for iteration in range(num_iterations):
        for opponent in opponents:
            run_game(my_agent, opponent, games_per_iteration)
    # 总计: 100 × 3 × 10 = 3000 场自对弈训练
```

### 智能体策略

```python
# agents/myTeam.py — 主智能体 (我的提交)
#   - State: 游戏盘面编码 + 工厂展示区 + 对手棋盘
#   - Action: {选择哪个工厂, 选什么颜色, 放到哪行}
#   - 基于启发式评估 + DQN 策略

# agents/t_070/myTeam.py — DQN 增强版本
#   - 神经网络特征提取 (盘面编码 → 动作 Q 值)
#   - ε-greedy 探索策略

# agents/mmAgent.py — Minimax 博弈树搜索
#   - 深度限制搜索 + 评估函数
#   - α-β 剪枝

# agents/generic/random.py — 随机基线
# agents/generic/first_move.py — 简单启发式 (选第一个合法动作)
# agents/generic/timeout.py — 超时测试用
```

## 技术栈

| 类别 | 技术 |
|------|------|
| 强化学习 | DQN (离线训练 + 在线推理) |
| 博弈论 | Minimax + α-β 剪枝 |
| 游戏引擎 | Python (状态机, Pygame GUI) |
| 锦标赛 | ELO 排名, general_game_runner.py |
| 环境 | Docker 支持, func-timeout 时间控制 |

## 运行方式

```bash
pip install func_timeout pytz pygame
cd azul-code

# 对战随机智能体
python general_game_runner.py -g Azul -a [agents.myTeam,agents.generic.random]

# DQN 训练
python agents/train_dqn.py

# 文本模式 (无 GUI)
python general_game_runner.py -g Azul -t -a [agents.myTeam,agents.mmAgent]
```

## 关键文件

| 文件 | 说明 |
|------|------|
| `myTeam.py` | 主智能体 (SelectAction 入口) |
| `myTeam_v2.py` | DQN 增强智能体 |
| `train_dqn.py` | DQN 自对弈训练脚本 |
| `azul_model.py` | 游戏状态机 (getLegalActions, generateSuccessor) |
| `azul_utils.py` | 游戏常量与计分规则 |
| `niubi.py` | 辅助工具/实验脚本 |

## 原始目录

`D:\010\master\code\azul-code\`
