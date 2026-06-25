# ai/DQN.py
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
from matplotlib.backends.backend_template import FigureCanvas

from state import GameState


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class DeepQNetwork:
    def __init__(self, maze, player, game_manager):
        super().__init__(maze, player, game_manager)
        self.maze = maze
        self.player = player
        self.game_manager = game_manager
        self.input_dim = 13  # 4基础坐标 + 9周围墙壁
        self.output_dim = 4
        self.batch_size = 128
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.memory = deque(maxlen=10000)
        self.priorities = deque(maxlen=10000)
        self.alpha = 0.6  # 优先级强度
        self.path = []
        self.path_time = 0.0
        self.is_success = False
        self.start_time = time.time()
        self.reward_history = []
        self.loss_history = []
        self.step_count = 0  # 新增步数统计
        # 双网络架构
        self.policy_net = DQN(self.input_dim, self.output_dim)
        self.target_net = DQN(self.input_dim, self.output_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0005)
        self.loss_fn = nn.SmoothL1Loss()
        self.update_counter = 0

    def _get_state(self):
        """增强状态表征：包含周围3x3网格信息"""
        # 基础坐标
        px = self.player.x / self.maze.maze_width
        py = self.player.y / self.maze.maze_height
        ex = self.maze.end_pos[0] / self.maze.grid_cols
        ey = self.maze.end_pos[1] / self.maze.grid_rows

        # 周围3x3墙壁信息
        grid_x = int(self.player.x // self.maze.cell_size)
        grid_y = int(self.player.y // self.maze.cell_size)
        walls = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = grid_x + dx, grid_y + dy
                if 0 <= nx < self.maze.grid_cols and 0 <= ny < self.maze.grid_rows:
                    walls.append(float(self.maze.maze_layout[ny][nx]))
                else:
                    walls.append(1.0)  # 边界视为墙壁

        return torch.FloatTensor([px, py, ex, ey] + walls)

    def _choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.output_dim - 1)
        else:
            with torch.no_grad():
                return torch.argmax(self.policy_net(state)).item()

    def _move(self, action):
        dx, dy = 0, 0
        if action == 0:
            dy = -1  # 上
        elif action == 1:
            dx = 1  # 右
        elif action == 2:
            dy = 1  # 下
        elif action == 3:
            dx = -1  # 左

        new_grid_x = (self.player.x // self.maze.cell_size) + dx
        new_grid_y = (self.player.y // self.maze.cell_size) + dy

        # 碰撞检测
        valid_move = False
        if 0 <= new_grid_x < self.maze.grid_cols and 0 <= new_grid_y < self.maze.grid_rows:
            if self.maze.maze_layout[new_grid_y][new_grid_x] == 0:
                self.player.x = new_grid_x * self.maze.cell_size + self.maze.cell_size // 2
                self.player.y = new_grid_y * self.maze.cell_size + self.maze.cell_size // 2
                valid_move = True
                self.path.append((new_grid_x, new_grid_y))

        return (new_grid_x, new_grid_y), valid_move

    def _calculate_reward(self, new_pos, valid_move):
        """改进的奖励函数"""
        # 基础距离奖励
        current_dist = abs(self.player.x - self.maze.end_pixel_pos[0]) + \
                       abs(self.player.y - self.maze.end_pixel_pos[1])
        new_dist = abs(new_pos[0] * self.maze.cell_size - self.maze.end_pixel_pos[0]) + \
                   abs(new_pos[1] * self.maze.cell_size - self.maze.end_pixel_pos[1])
        distance_reward = (current_dist - new_dist) * 2  # 每接近一步+2分

        # 事件奖励
        if new_pos == self.maze.end_pos:
            return 100, True
        elif not valid_move:
            return -20, False
        else:
            return distance_reward + 0.5, False

    def _store_memory(self, state, action, reward, next_state, done):
        max_prio = max(self.priorities) if self.priorities else 1.0
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(max_prio)

    def _replay_experience(self):
        # 计算采样概率
        prios = np.array(self.priorities) ** self.alpha
        probs = prios / sum(prios)

        indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        batch = [self.memory[i] for i in indices]

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.stack(next_states)
        dones = torch.BoolTensor(dones)

        # Double DQN
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_actions = self.policy_net(next_states).max(1)[1]
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1))
            target_q = rewards.unsqueeze(1) + (1 - dones.float().unsqueeze(1)) * self.gamma * next_q

        # 更新优先级
        td_errors = torch.abs(target_q - current_q).detach().numpy()
        for i, idx in enumerate(indices):
            self.priorities[idx] = td_errors[i][0] ** self.alpha + 1e-5

        # 计算损失
        loss = self.loss_fn(current_q, target_q)
        self.loss_history.append(loss.item())

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # 更新目标网络
        self.update_counter += 1
        if self.update_counter % 10 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def update(self, events=None):
        if self.is_success:
            return

        state = self._get_state()
        action = self._choose_action(state)
        new_pos, valid_move = self._move(action)
        reward, done = self._calculate_reward(new_pos, valid_move)
        next_state = self._get_state()

        self._store_memory(state, action, reward, next_state, done)
        self.reward_history.append(reward)

        if valid_move:
            self.step_count += 1  # 步数递增

        if len(self.memory) >= self.batch_size:
            self._replay_experience()

        # 动态epsilon衰减
        if done and reward > 0:
            self.epsilon *= 0.98  # 成功时加速衰减
        else:
            self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)

        # 成功处理
        if new_pos == self.maze.end_pos:
            self.is_success = True
            self.path_time = time.time() - self.start_time
            self.game_manager.current_state = GameState.SUCCESS
            self.game_manager.success_time = self.path_time
            self._save_training_curves()








