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

# 初始目标点（T1-T5）
initial_targets = torch.tensor([
    [1200, 800],  # T1
    [300, 450],   # T2
    [950, 200],   # T3
    [600, 1200],  # T4
    [1500, 500]   # T5
], dtype=torch.float32, requires_grad=False)

# T6 和障碍物（100s 时添加）
t6 = torch.tensor([800, 600], dtype=torch.float32, requires_grad=False)
obstacle_center = torch.tensor([900, 250], dtype=torch.float32, requires_grad=False)
obstacle_radius = 100
origin = torch.tensor([0, 0], dtype=torch.float32, requires_grad=False)
max_speed = 50
min_dist = 50
max_dist = 1000

# 环境类
class DroneEnv:
    def __init__(self):
        self.initial_targets = initial_targets
        self.t6 = t6
        self.obstacle_center = obstacle_center
        self.obstacle_radius = obstacle_radius
        self.reset()

    def reset(self):
        self.positions = [origin.clone() for _ in range(3)]
        self.visited = [False] * 6  # T1-T5 + T6
        self.time = 0
        self.done = False
        self.targets = self.initial_targets
        self.obstacle_active = False
        self.t6_active = False
        self.avoidance_history = [[] for _ in range(3)]
        self.path_lengths = [0.0] * 3  # 路径长度
        self.positions_at_100s = None  # 100秒位置
        self.recorded_100s = False  # 记录标志
        return self.get_state()

    def get_state(self):
        visited_tensor = torch.tensor(self.visited, dtype=torch.float32)
        return torch.cat([
            torch.stack(self.positions).flatten(),  # 6 维
            visited_tensor,  # 6 维
            torch.tensor([self.time], dtype=torch.float32),  # 1 维
            torch.tensor([int(self.obstacle_active)], dtype=torch.float32),  # 1 维
            torch.tensor([int(self.t6_active)], dtype=torch.float32)  # 1 维
        ])  # 总计 15 维

    def step(self, actions):
        reward = 0
        old_time = self.time

        # 检查 100s 更新
        if self.time >= 100 and not self.t6_active:
            self.targets = torch.cat([self.initial_targets, self.t6.unsqueeze(0)], dim=0)
            self.t6_active = True
            self.obstacle_active = True

        max_action = len(self.targets) + 1
        for i, action in enumerate(actions):
            if not isinstance(action, int):
                print(f"无效动作类型: {action} (类型: {type(action)})")
                reward -= 100
                continue
            if action < 0 or action >= max_action:
                print(f"动作超出范围: {action}, 最大允许: {max_action - 1}")
                reward -= 100
                continue

            if action < max_action - 1:
                target = self.targets[action]
                if target.shape != torch.Size([2]):
                    print(f"目标点形状错误: {target.shape}")
                    reward -= 100
                    continue
                if not self.visited[action]:
                    dist = torch.norm(self.positions[i] - target)
                    self.positions[i] = target.clone()
                    self.visited[action] = True
                    self.path_lengths[i] += dist.item()
                    reward += 200
                    self.time += dist / max_speed
            else:
                grid_point = torch.rand(2) * 1500
                dist = torch.norm(self.positions[i] - grid_point)
                self.positions[i] = grid_point.clone()
                self.path_lengths[i] += dist.item()
                self.time += dist / max_speed
                reward -= 100

            # 记录 100s 位置（改进逻辑）
            if not self.recorded_100s and self.time >= 100:
                self.positions_at_100s = [pos.clone() for pos in self.positions]
                self.recorded_100s = True
                print(f"记录100秒位置: 时间={self.time:.2f}s, 位置={[pos.tolist() for pos in self.positions_at_100s]}")

        # 调试时间更新
        if old_time != self.time:
            print(f"时间更新: {old_time:.2f}s -> {self.time:.2f}s")

        # 障碍物惩罚
        if self.obstacle_active:
            for pos in self.positions:
                if torch.norm(pos - self.obstacle_center) < self.obstacle_radius:
                    reward -= 500

        # 间距约束
        for i in range(3):
            for j in range(i + 1, 3):
                dist = torch.norm(self.positions[i] - self.positions[j])
                if dist < min_dist or dist > max_dist:
                    reward -= 200

        reward -= 5

        if all(self.visited[:len(self.targets)]):
            self.done = True
            reward += 2000

        return self.get_state(), reward, self.done

# DQN 模型
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.action_size = action_size
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size * 3)
        )

    def forward(self, x):
        return self.net(x).view(-1, 3, self.action_size)

# 绘制路径图
def plot_paths(episode, positions, visited, targets, obstacle_active, positions_at_100s, filename, title="路径图"):
    plt.figure(figsize=(12, 10))  # 增大图像尺寸
    # 绘制目标点
    plt.scatter(targets[:, 0].detach().numpy(), targets[:, 1].detach().numpy(), c='red', s=100, marker='o', label='目标点')
    # 绘制无人机路径
    colors = ['blue', 'orange', 'green']  # 为每架无人机分配不同颜色
    for i, pos in enumerate(positions):
        pos_np = pos.detach().numpy()
        plt.plot(pos_np[:, 0], pos_np[:, 1], color=colors[i], linewidth=2, label=f'无人机{i + 1}')
        # 标记路径起点
        plt.scatter(pos_np[0, 0], pos_np[0, 1], c=colors[i], s=150, marker='^', edgecolors='black')
    # 绘制障碍物
    if obstacle_active:
        circle = plt.Circle(obstacle_center.detach().numpy(), obstacle_radius, color='gray', alpha=0.3, label='障碍区域')
        plt.gca().add_patch(circle)
    # 绘制原点
    plt.scatter([0], [0], c='limegreen', s=150, marker='s', label='起点')
    # 绘制100秒位置
    if positions_at_100s is not None:
        pos_100s_np = [pos.detach().numpy() for pos in positions_at_100s]
        plt.scatter(
            [p[0] for p in pos_100s_np],
            [p[1] for p in pos_100s_np],
            c='purple',
            marker='*',
            s=300,  # 增大标记尺寸
            edgecolors='black',  # 添加黑色边框
            linewidths=1.5,
            label='100秒位置'
        )
    # 设置图表属性
    plt.title(title, fontsize=14)
    plt.xlabel('X坐标 (m)', fontsize=12)
    plt.ylabel('Y坐标 (m)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().set_aspect('equal')  # 确保比例一致
    plt.tight_layout()
    plt.savefig(filename, dpi=300)  # 高分辨率保存
    plt.close()

# 输出避障策略
def print_avoidance_strategy(avoidance_history):
    print("\n避障策略：")
    for i, history in enumerate(avoidance_history):
        print(f"无人机 {i + 1}:")
        for pos, status, dist in history:
            print(f"  位置: {pos}, 状态: {status}, 距障碍中心距离: {dist:.2f}m")
        if not history:
            print("  无避障记录（障碍物未激活或路径未经过障碍区域）")

# 训练
env = DroneEnv()
state_size = 15
action_size = 7
model = DQN(state_size, action_size)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
memory = deque(maxlen=10000)
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.9995
gamma = 0.99
batch_size = 32
episodes = 2000  # 增加轮次以确保 T6 访问
optimal_time = float('inf')
optimal_path = None
optimal_visited = None
optimal_targets = None
optimal_obstacle_active = False
optimal_avoidance_history = None
optimal_path_length = None
optimal_positions_at_100s = None
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
    if sum(env.visited) >= 6 and env.time < optimal_time:
        optimal_time = env.time
        optimal_path = copy.deepcopy(positions)
        optimal_visited = env.visited.copy()
        optimal_targets = env.targets.clone()
        optimal_obstacle_active = env.obstacle_active
        optimal_avoidance_history = copy.deepcopy(env.avoidance_history)
        optimal_path_length = sum(env.path_lengths)
        optimal_positions_at_100s = copy.deepcopy(env.positions_at_100s)
        optimal_episode = e

    if e % 10 == 0:
        print(f"轮次 {e}: 时间={env.time:.2f}s, 路径长度={sum(env.path_lengths):.2f}m, 访问目标={sum(env.visited)}, 奖励={reward}")

# 输出最优路径
if optimal_path is not None:
    print(f"\n最优路径 (轮次 {optimal_episode}, 时间 {optimal_time:.2f}s, 总路径长度 {optimal_path_length:.2f}m):")
    for i, pos in enumerate(optimal_path):
        print(f"无人机 {i + 1}: {[tuple(p.detach().numpy().tolist()) for p in pos]}")
    print(f"访问状态: {optimal_visited}")
    print(f"总时间: {optimal_time:.2f}s")
    print(f"总路径长度: {optimal_path_length:.2f}m")
    if optimal_positions_at_100s is not None:
        print("\n第100秒无人机位置：")
        for i, pos in enumerate(optimal_positions_at_100s):
            print(f"无人机 {i + 1}: {pos.detach().numpy().tolist()}")
    else:
        print("\n未达到100秒")
    print_avoidance_strategy(optimal_avoidance_history)
    plot_paths(optimal_episode, optimal_path, optimal_visited, optimal_targets, optimal_obstacle_active, optimal_positions_at_100s, "第二问optimal_path.png", f"最优路径图 (轮次 {optimal_episode}, 时间 {optimal_time:.2f}s, 路径长度 {optimal_path_length:.2f}m)")
else:
    print("\n未找到覆盖所有目标点的路径")
    print("最后一轮路径:")
    positions = [torch.stack(pos) for pos in positions]
    for i, pos in enumerate(positions):
        print(f"无人机 {i + 1}: {[tuple(p.detach().numpy().tolist()) for p in pos]}")
    print(f"访问状态: {env.visited}")
    print(f"总时间: {env.time:.2f}s")
    print(f"总路径长度: {sum(env.path_lengths):.2f}m")
    if env.positions_at_100s is not None:
        print("\n第100秒无人机位置：")
        for i, pos in enumerate(env.positions_at_100s):
            print(f"无人机 {i + 1}: {pos.detach().numpy().tolist()}")
    else:
        print("\n未达到100秒")
    print_avoidance_strategy(env.avoidance_history)
    plot_paths(episodes, positions, env.visited, env.targets, env.obstacle_active, env.positions_at_100s, "final_path.png", f"最后一轮路径图 (时间 {env.time:.2f}s, 路径长度 {sum(env.path_lengths):.2f}m)")