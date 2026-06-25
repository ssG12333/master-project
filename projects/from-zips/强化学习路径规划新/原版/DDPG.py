# ai/DQN.py 修改后的完整代码
import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import time
import pygame
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from state import GameState


# DDPG相关类
class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class CriticNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(CriticNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


class DDPG:
    def __init__(self, maze, player, game_manager):
        self.maze = maze
        self.player = player
        self.game_manager = game_manager
        self.input_dim = 13  # 4基础坐标 + 9周围墙壁
        self.action_dim = 4
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.001
        self.memory = deque(maxlen=10000)
        self.policy_net = ActorNetwork(self.input_dim, self.action_dim)
        self.target_policy_net = ActorNetwork(self.input_dim, self.action_dim)
        self.target_policy_net.load_state_dict(self.policy_net.state_dict())
        self.critic_net = CriticNetwork(self.input_dim, self.action_dim)
        self.target_critic_net = CriticNetwork(self.input_dim, self.action_dim)
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=0.002)
        self.loss_fn = nn.MSELoss()
        self.step_count = 0
        self.path_time = 0.0
        self.is_success = False
        self.start_time = time.time()
        self.times = []
        self.rewards = []

    def _get_state(self):
        px = self.player.x / self.maze.maze_width
        py = self.player.y / self.maze.maze_height
        ex = self.maze.end_pos[0] / self.maze.grid_cols
        ey = self.maze.end_pos[1] / self.maze.grid_rows

        grid_x = int(self.player.x // self.maze.cell_size)
        grid_y = int(self.player.y // self.maze.cell_size)
        walls = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = grid_x + dx, grid_y + dy
                if 0 <= nx < self.maze.grid_cols and 0 <= ny < self.maze.grid_rows:
                    walls.append(float(self.maze.maze_layout[ny][nx]))
                else:
                    walls.append(1.0)

        return torch.FloatTensor([px, py, ex, ey] + walls)

    def _choose_action(self, state):
        with torch.no_grad():
            return self.policy_net(state).numpy()

    def _move(self, action):
        dx, dy = 0, 0
        action_index = np.argmax(action)
        if action_index == 0:
            dy = -1  # 上
        elif action_index == 1:
            dx = 1  # 右
        elif action_index == 2:
            dy = 1  # 下
        elif action_index == 3:
            dx = -1  # 左

        new_x = (self.player.x // self.maze.cell_size) + dx
        new_y = (self.player.y // self.maze.cell_size) + dy

        reward = -0.1
        done = False
        if 0 <= new_x < self.maze.grid_cols and 0 <= new_y < self.maze.grid_rows:
            if self.maze.maze_layout[new_y][new_x] == 1:
                reward = -50
                done = True
            else:
                self.player.x = new_x * self.maze.cell_size + self.maze.cell_size // 2
                self.player.y = new_y * self.maze.cell_size + self.maze.cell_size // 2
        else:
            reward = -50
            done = True

        if (self.player.x, self.player.y) == self.maze.end_pixel_pos:
            reward = 100
            done = True

        self.rewards.append(reward)
        return reward, done

    def update(self, events=None):
        if self.is_success:
            return

        state = self._get_state()
        action = self._choose_action(state)
        reward, done = self._move(action)
        next_state = self._get_state()

        self.memory.append((state, action, reward, next_state, done))

        if len(self.memory) >= self.batch_size:
            self._replay_experience()

        if done and (self.player.x, self.player.y) == self.maze.end_pixel_pos:
            self.is_success = True
            self.path_time = time.time() - self.start_time
            self.times.append(self.path_time)
            self.game_manager.current_state = GameState.SUCCESS
            self.game_manager.success_time = self.path_time
            self._save_model()

    def _replay_experience(self):
        batch = random.sample(self.memory, self.batch_size)
        states = torch.stack([s[0] for s in batch])
        actions = torch.stack([s[1] for s in batch])
        rewards = torch.FloatTensor([s[2] for s in batch])
        next_states = torch.stack([s[3] for s in batch])
        dones = torch.FloatTensor([float(s[4]) for s in batch])

        # 更新 critic 网络
        with torch.no_grad():
            next_actions = self.target_policy_net(next_states)
            target_values = self.target_critic_net(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_values

        current_q = self.critic_net(states, actions)
        critic_loss = self.loss_fn(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新 actor 网络
        policy_loss = -self.critic_net(states, self.policy_net(states)).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # 更新目标网络
        for target_param, param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def _save_model(self):
        map_size = f"{self.maze.grid_cols}x{self.maze.grid_rows}"
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_policy_net': self.target_policy_net.state_dict(),
            'critic_net': self.critic_net.state_dict(),
            'target_critic_net': self.target_critic_net.state_dict()
        }, f"ddpg_model_{map_size}.pth")
        print(f"DDPG 模型已保存为 ddpg_model_{map_size}.pth")

    def _load_model(self):
        map_size = f"{self.maze.grid_cols}x{self.maze.grid_rows}"
        file_path = f"ddpg_model_{map_size}.pth"
        if os.path.exists(file_path):
            checkpoint = torch.load(file_path)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_policy_net.load_state_dict(checkpoint['target_policy_net'])
            self.critic_net.load_state_dict(checkpoint['critic_net'])
            self.target_critic_net.load_state_dict(checkpoint['target_critic_net'])
            print(f"已加载 DDPG 模型: {file_path}")
        else:
            print(f"未找到 DDPG 模型文件: {file_path}, 将使用新的模型")

