import heapq
import random
import time
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import time
import pygame
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

from ai.DDPG import ActorNetwork, CriticNetwork
from ai.DQN import DQN
from game.state import GameState  # 正确导入路径
from ai.Apathfinding import heuristic, AStar
import torch.nn as nn  # 添加这行到文件顶部

class PlayerController:
    def __init__(self, player):
        self.player = player
        self.move_speed = 1
        self.player.radius = self.player.cell_size // 4  # 重要！添加这行
        self.move_history = []  # 初始化move_history
        self.step_count = 0  # 新增步数统计
        # 新增 reset_path 方法
        def reset_path(self):
            self.move_history = []
            self.player.reset()
    def update(self, events):
        dx, dy = 0, 0
        keys = pygame.key.get_pressed()
        # 添加对退格键的检查（测试用）
        if keys[pygame.K_LEFT]: dx = -1
        if keys[pygame.K_RIGHT]: dx = 1
        if keys[pygame.K_UP]: dy = -1
        if keys[pygame.K_DOWN]: dy = 1
        # 移动逻辑（每次按键触发增加步数）
        if dx != 0 or dy != 0:
            self.step_count += 1  # 步数递增
        # 修改移动逻辑（添加调试输出）
        new_x = self.player.x + dx * self.move_speed
        new_y = self.player.y + dy * self.move_speed


        if not self.player.maze.is_wall(new_x, new_y, self.player.radius):
            self.player.x = new_x
            self.player.y = new_y

# 在BaseAIController中添加reset_path方法
class BaseAIController:
    def __init__(self, maze, player, game_manager):
        self.maze = maze
        self.player = player
        self.path = []
        self.path_time = 0.0
        self.is_success = False
        self.start_time = time.time()
        self.game_manager = game_manager
        self.step_count = 0  # 新增步数统计
        self.start_time = 0  # 控制器自身的时间记录（可选）
    def reset_path(self):
        """重置路径和步数"""
        self.path = []
        self.is_success = False
        self.step_count = 0
        self.start_time = time.time()
        # 确保玩家位置对齐迷宫起点
        self.player.x = self.maze.start_pos[0] * self.maze.cell_size + self.maze.cell_size // 2
        self.player.y = self.maze.start_pos[1] * self.maze.cell_size + self.maze.cell_size // 2
    def _record_success(self):
     if not self.is_success:
        self.is_success = True
        self.path_time = time.time() - self.start_time
        self.game_manager.current_state = GameState.SUCCESS
        self.game_manager.success_time = self.path_time


class AStarController(BaseAIController):
    def __init__(self, maze, player, game_manager):
        super().__init__(maze, player, game_manager)
        self.astar = AStar(maze, player, game_manager)
        self.path = self.astar.path
        self.step_interval = 0.15
        self.last_move_time = time.time()
        self.is_success = self.astar.is_success

    def reset_path(self):
        """重置路径并重新寻路"""
        super().reset_path()
        self.path = []
        self.is_success = False
        self._find_path()
        self.player.x = self.maze.start_pos[0] * self.maze.cell_size + self.maze.cell_size // 2
        self.player.y = self.maze.start_pos[1] * self.maze.cell_size + self.maze.cell_size // 2

    def _find_path(self):
        """A*寻路核心逻辑，生成合法路径"""
        start = self.maze.start_pos
        end = self.maze.end_pos

        open_heap = []
        closed_set = set()
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, end)}
        heapq.heappush(open_heap, (f_score[start], start))

        cols = self.maze.grid_cols
        rows = self.maze.grid_rows

        while open_heap:
            current = heapq.heappop(open_heap)[1]
            if current == end:
                self.path = self._reconstruct_path(came_from, end)
                return

            closed_set.add(current)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)

                if not (0 <= neighbor[0] < cols and 0 <= neighbor[1] < rows):
                    continue
                if self.maze.maze_layout[neighbor[1]][neighbor[0]] == 1:
                    continue

                tentative_g = g_score[current] + 1
                if neighbor in closed_set and tentative_g >= g_score.get(neighbor, float('inf')):
                    continue

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, end)
                    heapq.heappush(open_heap, (f_score[neighbor], neighbor))

        self.path = []

    def _reconstruct_path(self, came_from, end):
        """重建路径并确保顺序正确"""
        path = [end]
        current = end
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    def update(self, events=None):
        """移动逻辑（包含路径空时自动重新寻路）"""
        if self.is_success or not self.path:
            if not self.path:  # 路径为空时重新寻路
                self._find_path()
            return

        current_time = time.time()
        if current_time - self.last_move_time < self.step_interval:
            return

        # 移动逻辑（严格碰撞检测）
        if self.path:
            target_cell = self.path.pop(0)
            target_x = target_cell[0] * self.maze.cell_size + self.maze.cell_size // 2
            target_y = target_cell[1] * self.maze.cell_size + self.maze.cell_size // 2
            self.step_count += 1  # 每次移动时递增

            # 严格检测目标单元格是否为墙壁
            target_grid_x = target_cell[0]
            target_grid_y = target_cell[1]
            if self.maze.maze_layout[target_grid_y][target_grid_x] == 1:
                self.path = []  # 立即终止非法路径
                return

            # 更新玩家位置
            self.player.x = target_x
            self.player.y = target_y
            self.last_move_time = current_time

            # 成功检测
            if (self.player.x, self.player.y) == self.maze.end_pixel_pos:
                self.is_success = True
                self._record_success()

    def _record_success(self):
        """记录成功状态"""
        self.path_time = time.time() - self.start_time
        self.is_success = True
        self.game_manager.current_state = GameState.SUCCESS
        self.game_manager.success_time = self.path_time



class QLearningController(BaseAIController):
    def __init__(self, maze, player, game_manager):
        super().__init__(maze, player, game_manager)
        self.maze = maze
        self.player = player
        self.game_manager = game_manager
        self.actions = [0, 1, 2, 3]
        self.alpha = 0.2
        self.gamma = 0.95
        self.epsilon = 0.9
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        self.path = []
        self.path_time = 0.0
        self.is_success = False
        self.start_time = time.time()
        self._init_q_table()
        self.step_count = 0

    def _init_q_table(self):
        self.grid_cols = self.maze.grid_cols
        self.grid_rows = self.maze.grid_rows
        self.q_table = np.zeros((self.grid_cols, self.grid_rows, 4), dtype=np.float32) + 1e-4

    def _get_state(self):
        x = self.player.x // self.maze.cell_size
        y = self.player.y // self.maze.cell_size
        return (x, y)

    def _choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return np.argmax(self.q_table[state[0], state[1]])

    def _move(self, action):
        dx, dy = 0, 0
        if action == 0:
            dy = -1
        elif action == 1:
            dx = 1
        elif action == 2:
            dy = 1
        elif action == 3:
            dx = -1

        new_x = (self.player.x // self.maze.cell_size) + dx
        new_y = (self.player.y // self.maze.cell_size) + dy

        # 检查新位置是否在迷宫范围内
        if 0 <= new_x < self.grid_cols and 0 <= new_y < self.grid_rows:
            if self.maze.maze_layout[new_y][new_x] != 1:
                self.player.x = new_x * self.maze.cell_size + self.maze.cell_size // 2
                self.player.y = new_y * self.maze.cell_size + self.maze.cell_size // 2
                self.path.append((new_x, new_y))
        else:
            # 如果超出边界，返回当前位置（避免生成非法状态）
            new_x = self.player.x // self.maze.cell_size
            new_y = self.player.y // self.maze.cell_size

        return (new_x, new_y)

    def _get_reward(self, state):
        # 检查状态是否超出迷宫边界
        if (state[0] < 0 or state[0] >= self.maze.grid_cols or
                state[1] < 0 or state[1] >= self.maze.grid_rows):
            return -50, True  # 超出边界视为撞墙

        if state == self.maze.end_pos:
            return 100, True
        elif self.maze.maze_layout[state[1]][state[0]] == 1:
            return -50, True
        else:
            return -0.1, False

    def update(self, events=None):
        if self.is_success:
            return

        current_state = self._get_state()
        action = self._choose_action(current_state)
        new_state = self._move(action)
        reward, done = self._get_reward(new_state)

        old_value = self.q_table[current_state[0], current_state[1], action]
        next_max = np.max(self.q_table[new_state[0], new_state[1]])
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[current_state[0], current_state[1], action] = new_value

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

        if new_state == self.maze.end_pos:
            self.is_success = True
            self.path_time = time.time() - self.start_time
            self.game_manager.current_state = GameState.SUCCESS
            self._record_success()

    def _record_success(self):
        if not self.is_success:
            self.is_success = True
            self.path_time = time.time() - self.start_time
            self.game_manager.current_state = GameState.SUCCESS
            self.game_manager.success_time = self.path_time

    def reset_path(self):
        self.path = []
        self.is_success = False
        self.step_count = 0
        self.start_time = time.time()
        self.player.x = self.maze.start_pos[0] * self.maze.cell_size + self.maze.cell_size // 2
        self.player.y = self.maze.start_pos[1] * self.maze.cell_size + self.maze.cell_size // 2

    def _save_q_table(self):
        map_size = f"{self.maze.grid_cols}x{self.maze.grid_rows}"
        np.save(f"q_table_{map_size}.npy", self.q_table)

    def _load_q_table(self):
        map_size = f"{self.maze.grid_cols}x{self.maze.grid_rows}"
        file_path = f"q_table_{map_size}.npy"
        if os.path.exists(file_path):
            self.q_table = np.load(file_path)
        else:
            print(f"未找到 Q 表文件: {file_path}, 将使用新的 Q 表")

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

class DeepQNetworkController(BaseAIController):
    def __init__(self, maze, player, game_manager):
        super().__init__(maze, player, game_manager)
        self.maze = maze
        self.player = player
        self.game_manager = game_manager
        self.input_dim = 13  # 基础坐标 + 周围墙壁
        self.output_dim = 4
        self.batch_size = 128
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.memory = deque(maxlen=10000)
        self.priorities = deque(maxlen=10000)
        self.alpha = 0.6
        self.path = []
        self.path_time = 0.0
        self.is_success = False
        self.start_time = time.time()
        self.reward_history = []
        self.loss_history = []
        self.step_count = 0

        # 启用 GPU 加速
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.input_dim, self.output_dim).to(self.device)
        self.target_net = DQN(self.input_dim, self.output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0005)
        self.loss_fn = nn.SmoothL1Loss()
        self.update_counter = 0

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

        state = torch.FloatTensor([px, py, ex, ey] + walls).to(self.device)
        return state

    def _choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.output_dim - 1)
        else:
            with torch.no_grad():
                return torch.argmax(self.policy_net(state)).item()

    def _move(self, action):
        dx, dy = 0, 0
        if action == 0:
            dy = -1
        elif action == 1:
            dx = 1
        elif action == 2:
            dy = 1
        elif action == 3:
            dx = -1

        new_grid_x = (self.player.x // self.maze.cell_size) + dx
        new_grid_y = (self.player.y // self.maze.cell_size) + dy

        valid_move = False
        if 0 <= new_grid_x < self.maze.grid_cols and 0 <= new_grid_y < self.maze.grid_rows:
            if self.maze.maze_layout[new_grid_y][new_grid_x] == 0:
                self.player.x = new_grid_x * self.maze.cell_size + self.maze.cell_size // 2
                self.player.y = new_grid_y * self.maze.cell_size + self.maze.cell_size // 2
                valid_move = True
                self.path.append((new_grid_x, new_grid_y))

        return (new_grid_x, new_grid_y), valid_move

    def _calculate_reward(self, new_pos, valid_move):
        current_dist = abs(self.player.x - self.maze.end_pixel_pos[0]) + \
                       abs(self.player.y - self.maze.end_pixel_pos[1])
        new_dist = abs(new_pos[0] * self.maze.cell_size - self.maze.end_pixel_pos[0]) + \
                   abs(new_pos[1] * self.maze.cell_size - self.maze.end_pixel_pos[1])
        distance_reward = (current_dist - new_dist) * 2

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
        prios = np.array(self.priorities) ** self.alpha
        probs = prios / sum(prios)

        indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        batch = [self.memory[i] for i in indices]

        states = torch.stack([s[0] for s in batch]).to(self.device)
        actions = torch.LongTensor([s[1] for s in batch]).to(self.device)
        rewards = torch.FloatTensor([s[2] for s in batch]).to(self.device)
        next_states = torch.stack([s[3] for s in batch]).to(self.device)
        dones = torch.BoolTensor([s[4] for s in batch]).to(self.device)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_actions = self.policy_net(next_states).max(1)[1]
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1))
            target_q = rewards.unsqueeze(1) + (1 - dones.float().unsqueeze(1)) * self.gamma * next_q

        td_errors = torch.abs(target_q - current_q).detach().cpu().numpy()
        for i, idx in enumerate(indices):
            self.priorities[idx] = td_errors[i][0] ** self.alpha + 1e-5

        loss = self.loss_fn(current_q, target_q)
        self.loss_history.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

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
            self.step_count += 1

        if len(self.memory) >= self.batch_size:
            self._replay_experience()

        if done and reward > 0:
            self.epsilon *= 0.98
        else:
            self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)

        if new_pos == self.maze.end_pos:
            self.is_success = True
            self.path_time = time.time() - self.start_time
            self.game_manager.current_state = GameState.SUCCESS
            self.game_manager.success_time = self.path_time

    def reset_path(self):
        self.path = []
        self.is_success = False
        self.step_count = 0
        self.start_time = time.time()
        self.player.x = self.maze.start_pos[0] * self.maze.cell_size + self.maze.cell_size // 2
        self.player.y = self.maze.start_pos[1] * self.maze.cell_size + self.maze.cell_size // 2

    def _save_model(self):
        map_size = f"{self.maze.grid_cols}x{self.maze.grid_rows}"
        torch.save(self.policy_net.state_dict(), f"dqn_model_{map_size}.pth")

    def _load_model(self):
        map_size = f"{self.maze.grid_cols}x{self.maze.grid_rows}"
        file_path = f"dqn_model_{map_size}.pth"
        if os.path.exists(file_path):
            self.policy_net.load_state_dict(torch.load(file_path, map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            print(f"未找到 DQN 模型文件: {file_path}, 将使用新的模型")


class DDPGController(BaseAIController):
    def __init__(self, maze, player, game_manager):
        super().__init__(maze, player, game_manager)
        self.maze = maze
        self.player = player
        self.game_manager = game_manager
        self.input_dim = 13  # 基础坐标 + 周围墙壁
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

        # 启用 GPU 加速
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = self.policy_net.to(self.device)
        self.target_actor = self.target_policy_net.to(self.device)
        self.critic = self.critic_net.to(self.device)
        self.target_critic = self.target_critic_net.to(self.device)

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

        state = torch.FloatTensor([px, py, ex, ey] + walls).to(self.device)
        return state

    def _choose_action(self, state):
        with torch.no_grad():
            return self.actor(state)

    def _move(self, action):
        dx, dy = 0, 0
        action_index = torch.argmax(action).item()
        if action_index == 0:
            dy = -1
        elif action_index == 1:
            dx = 1
        elif action_index == 2:
            dy = 1
        elif action_index == 3:
            dx = -1

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
        states = torch.stack([s[0] for s in batch]).to(self.device)
        actions = torch.stack([s[1] for s in batch]).to(self.device)
        rewards = torch.FloatTensor([s[2] for s in batch]).to(self.device)
        next_states = torch.stack([s[3] for s in batch]).to(self.device)
        dones = torch.FloatTensor([float(s[4]) for s in batch]).to(self.device)

        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_values = self.target_critic(next_states, next_actions)
            # 确保 target_q 和 current_q 的形状一致
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * target_values

        current_q = self.critic(states, actions)
        critic_loss = self.loss_fn(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        policy_loss = -self.critic(states, self.actor(states)).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def _save_model(self):
        map_size = f"{self.maze.grid_cols}x{self.maze.grid_rows}"
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict()
        }, f"ddpg_model_{map_size}.pth")
        print(f"DDPG 模型已保存为 ddpg_model_{map_size}.pth")

    def _load_model(self):
        map_size = f"{self.maze.grid_cols}x{self.maze.grid_rows}"
        file_path = f"ddpg_model_{map_size}.pth"
        if os.path.exists(file_path):
            checkpoint = torch.load(file_path, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.target_actor.load_state_dict(checkpoint['target_actor'])
            self.target_critic.load_state_dict(checkpoint['target_critic'])
            print(f"已加载 DDPG 模型: {file_path}")
        else:
            print(f"未找到 DDPG 模型文件: {file_path}, 将使用新的模型")




