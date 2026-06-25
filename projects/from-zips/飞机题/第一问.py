import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import copy

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 目标点（T1-T5）
targets = torch.tensor([
    [1200, 800],  # T1
    [300, 450],   # T2
    [950, 200],   # T3
    [600, 1200],  # T4
    [1500, 500]   # T5
], dtype=torch.float32, requires_grad=False)

origin = torch.tensor([0, 0], dtype=torch.float32, requires_grad=False)
max_speed = 50

# 环境类
class DroneEnv:
    def __init__(self):
        self.targets = targets
        self.max_speed = max_speed
        self.min_dist = 50
        self.max_dist = 1000
        self.max_time = 600
        self.reset()

    def reset(self):
        self.positions = [origin.clone() for _ in range(3)]
        self.visited = [False] * 5
        self.time = 0
        self.done = False
        self.path_lengths = [0.0] * 3
        return self.get_state()

    def get_state(self):
        return torch.cat([
            torch.stack(self.positions).flatten(),
            torch.tensor(self.visited, dtype=torch.float32),
            torch.tensor([self.time], dtype=torch.float32)
        ])

    def step(self, actions):
        reward = 0
        max_action = len(self.targets) + 1
        new_positions = [pos.clone() for pos in self.positions]
        follow_actions = [False] * 3

        # 第一步：检查距离约束并决定是否需要跟随
        for i in range(3):
            if not isinstance(actions[i], int) or actions[i] < 0 or actions[i] >= max_action:
                print(f"无人机 {i+1} 无效动作: {actions[i]}")
                reward -= 300
                continue

            if actions[i] < max_action - 1:
                target = self.targets[actions[i]]
                if not self.visited[actions[i]]:
                    new_positions[i] = target.clone()
                else:
                    follow_actions[i] = True
            else:
                new_positions[i] = torch.rand(2) * 1500

            for j in range(3):
                if i != j:
                    dist = torch.norm(new_positions[i] - new_positions[j])
                    if dist > self.max_dist:
                        follow_actions[i] = True
                        break

        # 第二步：执行动作或跟随
        for i in range(3):
            if follow_actions[i]:
                closest_dist = float('inf')
                closest_drone = None
                for j in range(3):
                    if i != j:
                        dist = torch.norm(self.positions[i] - self.positions[j])
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_drone = j
                if closest_drone is not None and closest_dist > 0:  # 修复：检查 closest_drone 和 dist
                    direction = self.positions[closest_drone] - self.positions[i]
                    move_dist = min(closest_dist, 900)
                    new_positions[i] = self.positions[i] + direction / closest_dist * move_dist
                    reward -= 10
                    self.path_lengths[i] += move_dist
                    self.time += move_dist / self.max_speed
                else:
                    reward -= 10  # 不移动时的惩罚
            else:
                dist = torch.norm(self.positions[i] - new_positions[i])
                self.positions[i] = new_positions[i].clone()
                self.path_lengths[i] += dist.item()
                self.time += dist / self.max_speed
                if actions[i] < max_action - 1 and not self.visited[actions[i]]:
                    self.visited[actions[i]] = True
                    reward += 1000
                else:
                    reward -= 500
                reward -= dist * 0.5

        # 第三步：检查时间约束
        if self.time > self.max_time:
            self.done = True
            reward -= 2000
            return self.get_state(), reward, self.done

        # 第四步：检查间距约束
        for i in range(3):
            for j in range(i + 1, 3):
                dist = torch.norm(self.positions[i] - self.positions[j])
                if dist < self.min_dist or dist > self.max_dist:
                    reward -= 300

        # 第五步：奖励优化
        total_path_length = sum(self.path_lengths)
        reward -= total_path_length * 0.01
        if all(self.visited):
            self.done = True
            reward += 10000
            reward += max(0, (self.max_time - self.time) * 20)

        return self.get_state(), reward, self.done

# DQN 模型
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.action_size = action_size
        self.net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_size * 3)
        )

    def forward(self, x):
        return self.net(x).view(-1, 3, self.action_size)

# 绘制路径图
def plot_paths(episode, positions, visited, targets, filename, title="路径图"):
    plt.figure(figsize=(10, 8))
    plt.scatter(targets[:, 0].detach().numpy(), targets[:, 1].detach().numpy(), c='red', label='目标点')
    for i, pos in enumerate(positions):
        pos_np = pos.detach().numpy()
        plt.plot(pos_np[:, 0], pos_np[:, 1], label=f'无人机{i + 1}')
    plt.scatter([0], [0], c='green', label='起点')
    plt.title(title)
    plt.xlabel('X坐标 (m)')
    plt.ylabel('Y坐标 (m)')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

# 训练
env = DroneEnv()
state_size = 12
action_size = 6
model = DQN(state_size, action_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
memory = deque(maxlen=10000)
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.998
gamma = 0.95
batch_size = 128
episodes = 6000
path_history = []
optimal_path_length = float('inf')
optimal_path = None
optimal_visited = None
optimal_time = None
optimal_episode = -1

for e in range(episodes):
    state = env.reset()
    positions = [[] for _ in range(3)]
    for i in range(3):
        positions[i].append(origin.clone())

    for t in range(300):
        if random.random() < epsilon:
            max_action = len(env.targets) + 1
            actions = [random.randrange(max_action) for _ in range(3)]
        else:
            with torch.no_grad():
                q_values = model(state.unsqueeze(0))
                if q_values.shape != torch.Size([1, 3, action_size]):
                    print(f"q_values 形状错误: {q_values.shape}")
                    max_action = len(env.targets) + 1
                    actions = [random.randrange(max_action) for _ in range(3)]
                else:
                    actions = q_values.squeeze(0).argmax(dim=1).tolist()

        next_state, reward, done = env.step(actions)
        memory.append((state, actions, reward, next_state, done))
        state = next_state
        for i, pos in enumerate(env.positions):
            positions[i].append(pos.clone())

        if done:
            break

    if len(memory) > batch_size:
        batch = random.sample(memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = model(states)
        next_q_values = model(next_states)
        targets = q_values.clone()
        for i in range(batch_size):
            for j, a in enumerate(actions[i]):
                targets[i, j, a] = rewards[i] + (1 - dones[i]) * gamma * next_q_values[i, j].max()

        optimizer.zero_grad()
        loss = nn.MSELoss()(q_values, targets)
        loss.backward()
        optimizer.step()

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    positions = [torch.stack(pos) for pos in positions]
    total_path_length = sum(env.path_lengths)
    if sum(env.visited) == 5 and total_path_length < optimal_path_length and env.time <= env.max_time:
        optimal_path_length = total_path_length
        optimal_path = copy.deepcopy(positions)
        optimal_visited = env.visited.copy()
        optimal_time = env.time
        optimal_episode = e

    if e % 100 == 0:
        print(f"轮次 {e}: 时间={env.time:.2f}s, 路径长度={total_path_length:.2f}m, 访问目标={sum(env.visited)}, 奖励={reward}")

# 输出最优路径
if optimal_path is not None:
    print(f"\n最优路径 (轮次 {optimal_episode}, 时间 {optimal_time:.2f}s, 总路径长度 {optimal_path_length:.2f}m):")
    for i, pos in enumerate(optimal_path):
        print(f"无人机 {i + 1}: {[tuple(p.detach().numpy().tolist()) for p in pos]}")
    print(f"访问状态: {optimal_visited}")
    print(f"总时间: {optimal_time:.2f}s")
    print(f"总路径长度: {optimal_path_length:.2f}m")
    plot_paths(optimal_episode, optimal_path, optimal_visited, env.targets, "optimal_path.png", f"最优路径图 (轮次 {optimal_episode}, 时间 {optimal_time:.2f}s, 路径长度 {optimal_path_length:.2f}m)")
else:
    print("\n未找到覆盖所有目标点的路径")
    print("最后一轮路径:")
    positions = [torch.stack(pos) for pos in positions]
    for i, pos in enumerate(positions):
        print(f"无人机 {i + 1}: {[tuple(p.detach().numpy().tolist()) for p in pos]}")
    print(f"访问状态: {env.visited}")
    print(f"总时间: {env.time:.2f}s")
    print(f"总路径长度: {sum(env.path_lengths):.2f}m")
    plot_paths(episodes, positions, env.visited, env.targets, "final_path.png", f"最后一轮路径图 (时间 {env.time:.2f}s, 路径长度 {sum(env.path_lengths):.2f}m)")