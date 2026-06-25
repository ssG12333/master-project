import heapq
import random
import time
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pygame
import matplotlib.pyplot as plt
import os
import pandas as pd

from DDPG import ActorNetwork, CriticNetwork
from DQN import DQN
from state import GameState
from Apathfinding import heuristic, AStar

class PlayerController:
    def __init__(self, player):
        self.player = player
        self.move_speed = 1
        self.player.radius = self.player.cell_size // 4
        self.move_history = []
        self.step_count = 0

    def reset_path(self):
        self.move_history = []
        self.player.reset()
        self.step_count = 0

    def update(self, events):
        dx, dy = 0, 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: dx = -1
        if keys[pygame.K_RIGHT]: dx = 1
        if keys[pygame.K_UP]: dy = -1
        if keys[pygame.K_DOWN]: dy = 1
        if dx != 0 or dy != 0:
            self.step_count += 1
        new_x = self.player.x + dx * self.move_speed
        new_y = self.player.y + dy * self.move_speed
        if not self.player.maze.is_wall(new_x, new_y, self.player.radius):
            self.player.x = new_x
            self.player.y = new_y
            self.move_history.append((new_x, new_y))

class BaseAIController:
    def __init__(self, maze, player, game_manager):
        self.maze = maze
        self.player = player
        self.path = []
        self.path_time = 0.0
        self.is_success = False
        self.start_time = time.time()
        self.game_manager = game_manager
        self.step_count = 0
        self.is_training = False

    def reset_path(self):
        self.path = []
        self.is_success = False
        self.step_count = 0
        self.start_time = time.time()
        self.player.x = self.maze.start_pos[0] * self.maze.cell_size + self.maze.cell_size // 2
        self.player.y = self.maze.start_pos[1] * self.maze.cell_size + self.maze.cell_size // 2

    def _record_success(self):
        if not self.is_success:
            self.is_success = True
            self.path_time = time.time() - self.start_time
            self.game_manager.current_state = GameState.SUCCESS
            self.game_manager.success_time = self.path_time

    def _save_training_data(self, episode, episode_reward):
        pass

    def _generate_training_charts(self):
        pass

class AStarController(BaseAIController):
    def __init__(self, maze, player, game_manager):
        super().__init__(maze, player, game_manager)
        self.astar = AStar(maze, player, game_manager)
        self.path = self.astar.path
        self.step_interval = 0.15
        self.last_move_time = time.time()
        self.is_success = self.astar.is_success

    def reset_path(self):
        super().reset_path()
        self.path = []
        self.is_success = False
        self._find_path()
        self.player.x = self.maze.start_pos[0] * self.maze.cell_size + self.maze.cell_size // 2
        self.player.y = self.maze.start_pos[1] * self.maze.cell_size + self.maze.cell_size // 2

    def _find_path(self):
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
        path = [end]
        current = end
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    def update(self, events=None):
        if self.is_success or not self.path:
            if not self.path:
                self._find_path()
            return
        current_time = time.time()
        if current_time - self.last_move_time < self.step_interval:
            return
        if self.path:
            target_cell = self.path.pop(0)
            target_x = target_cell[0] * self.maze.cell_size + self.maze.cell_size // 2
            target_y = target_cell[1] * self.maze.cell_size + self.maze.cell_size // 2
            self.step_count += 1
            target_grid_x = target_cell[0]
            target_grid_y = target_cell[1]
            if self.maze.maze_layout[target_grid_y][target_grid_x] == 1:
                self.path = []
                return
            self.player.x = target_x
            self.player.y = target_y
            self.last_move_time = current_time
            if (self.player.x, self.player.y) == self.maze.end_pixel_pos:
                self.is_success = True
                self._record_success()

class QLearningController(BaseAIController):
    def __init__(self, maze, player, game_manager):
        super().__init__(maze, player, game_manager)
        self.actions = [0, 1, 2, 3]
        self.alpha = 0.2
        self.gamma = 0.95
        self.epsilon = 0.9
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        self.q_history = []
        self.training_data = []
        self._init_q_table()

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
        if action == 0: dy = -1
        elif action == 1: dx = 1
        elif action == 2: dy = 1
        elif action == 3: dx = -1
        new_x = (self.player.x // self.maze.cell_size) + dx
        new_y = (self.player.y // self.maze.cell_size) + dy
        if 0 <= new_x < self.grid_cols and 0 <= new_y < self.grid_rows:
            if self.maze.maze_layout[new_y][new_x] != 1:
                self.player.x = new_x * self.maze.cell_size + self.maze.cell_size // 2
                self.player.y = new_y * self.maze.cell_size + self.maze.cell_size // 2
                self.path.append((new_x, new_y))
                self.step_count += 1
        else:
            new_x = self.player.x // self.maze.cell_size
            new_y = self.player.y // self.maze.cell_size
        return (new_x, new_y)

    def _get_reward(self, state):
        if (state[0] < 0 or state[0] >= self.maze.grid_cols or
                state[1] < 0 or state[1] >= self.maze.grid_rows):
            return -50, True
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
        avg_q = np.mean(self.q_table)
        self.q_history.append(avg_q)
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
        if new_state == self.maze.end_pos:
            self.is_success = True
            self.path_time = time.time() - self.start_time
            self._record_success()

    def reset_path(self):
        self.path = []
        self.is_success = False
        self.step_count = 0
        self.start_time = time.time()
        self.player.x = self.maze.start_pos[0] * self.maze.cell_size + self.maze.cell_size // 2
        self.player.y = self.maze.start_pos[1] * self.maze.cell_size + self.maze.cell_size // 2
        self.training_data = []

    def _save_training_data(self, episode, episode_reward):
        map_size = f"{self.maze.grid_cols}x{self.maze.grid_rows}"
        avg_q = self.q_history[-1] if self.q_history else 0.0
        data = {
            'Episode': episode,
            'Reward': episode_reward,
            'Average_Q': avg_q,
            'Step_Count': self.step_count
        }
        self.training_data.append(data)
        df = pd.DataFrame([data])  # 只保存当前轮次的数据
        file_path = f"training_data_QLearning_{map_size}.csv"
        # 如果文件存在则追加数据，不写表头；否则创建新文件并写表头
        if os.path.exists(file_path):
            df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            df.to_csv(file_path, index=False)
        print(f"QLearning 轮次 {episode}/200, 奖励: {episode_reward:.2f}, 平均 Q 值: {avg_q:.4f}, 步数: {self.step_count}")

    def _generate_training_charts(self):
        map_size = f"{self.maze.grid_cols}x{self.maze.grid_rows}"
        algo_name = "QLearning"

        # Q值曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.q_history, label="Average Q Value", color='blue')
        plt.xlabel("Training Steps")
        plt.ylabel("Average Q Value")
        plt.title(f"{algo_name} Q Value Curve ({map_size})")
        plt.legend()
        plt.grid(True)
        q_value_file = f"q_value_{algo_name}_{map_size}.png"
        plt.savefig(q_value_file)
        plt.close()
        os.system(f"start {q_value_file}")

        # 奖励和步数曲线
        csv_file = f"training_data_{algo_name}_{map_size}.csv"
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)

            # 奖励曲线
            plt.figure(figsize=(10, 6))
            plt.plot(df['Episode'], df['Reward'], label="Reward", color='green')
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title(f"{algo_name} Reward Curve ({map_size})")
            plt.legend()
            plt.grid(True)
            reward_file = f"reward_{algo_name}_{map_size}.png"
            plt.savefig(reward_file)
            plt.close()
            os.system(f"start {reward_file}")

            # 步数曲线
            plt.figure(figsize=(10, 6))
            plt.plot(df['Episode'], df['Step_Count'], label="Step Count", color='orange')
            plt.xlabel("Episode")
            plt.ylabel("Step Count")
            plt.title(f"{algo_name} Step Count Curve ({map_size})")
            plt.legend()
            plt.grid(True)
            steps_file = f"steps_{algo_name}_{map_size}.png"
            plt.savefig(steps_file)
            plt.close()
            os.system(f"start {steps_file}")
        else:
            print(f"未找到 {algo_name} 的训练数据文件: {csv_file}")

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

class DeepQNetworkController(BaseAIController):
    def __init__(self, maze, player, game_manager):
        super().__init__(maze, player, game_manager)
        self.input_dim = 13
        self.output_dim = 4
        self.batch_size = 128
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.memory = deque(maxlen=10000)
        self.reward_history = []
        self.loss_history = []
        self.training_data = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.autograd.set_detect_anomaly(True)
        self.policy_net = DQN(self.input_dim, self.output_dim).to(self.device)
        self.target_net = DQN(self.input_dim, self.output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0005)
        self.loss_fn = nn.SmoothL1Loss()
        self.update_counter = 0
        self._initialize_optimizer()
        param_count = sum(p.numel() for p in self.policy_net.parameters() if p.requires_grad)
        print(f"DQN 优化器初始化完成，参数数量：{param_count}")

    def _initialize_optimizer(self):
        for param in self.policy_net.parameters():
            param.requires_grad = True
        self.optimizer.zero_grad()
        dummy_input = torch.zeros(1, self.input_dim, device=self.device, requires_grad=True)
        dummy_output = self.policy_net(dummy_input).clone()
        dummy_loss = dummy_output.mean()
        dummy_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        for name, param in self.policy_net.named_parameters():
            if not param.requires_grad:
                print(f"警告: 参数 {name} 不需要梯度")
        print("DQN 优化器状态已强制初始化")

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
        if action == 0: dy = -1
        elif action == 1: dx = 1
        elif action == 2: dy = 1
        elif action == 3: dx = -1
        new_grid_x = (self.player.x // self.maze.cell_size) + dx
        new_grid_y = (self.player.y // self.maze.cell_size) + dy
        valid_move = False
        if 0 <= new_grid_x < self.maze.grid_cols and 0 <= new_grid_y < self.maze.grid_rows:
            if self.maze.maze_layout[new_grid_y][new_grid_x] == 0:
                if not self.is_training:
                    self.player.x = new_grid_x * self.maze.cell_size + self.maze.cell_size // 2
                    self.player.y = new_grid_y * self.maze.cell_size + self.maze.cell_size // 2
                valid_move = True
                self.path.append((new_grid_x, new_grid_y))
                self.step_count += 1
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
        state = state.detach()
        next_state = next_state.detach()
        self.memory.append((state, action, reward, next_state, done))

    def _replay_experience(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states = torch.stack([s[0] for s in batch]).to(self.device).requires_grad_(True)
        actions = torch.LongTensor([s[1] for s in batch]).to(self.device)
        rewards = torch.FloatTensor([s[2] for s in batch]).to(self.device)
        next_states = torch.stack([s[3] for s in batch]).to(self.device)
        dones = torch.BoolTensor([s[4] for s in batch]).to(self.device)

        self.policy_net.train()
        policy_output = self.policy_net(states).clone()
        current_q = policy_output.gather(1, actions.unsqueeze(1)).clone()

        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards.unsqueeze(1) + (1 - dones.float().unsqueeze(1)) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)
        self.loss_history.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.update_counter += 1

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
            self._record_success()
            self._save_model()
        if self.update_counter % 10 == 0 and self.update_counter > 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def reset_path(self):
        self.path = []
        self.is_success = False
        self.step_count = 0
        self.start_time = time.time()
        self.player.x = self.maze.start_pos[0] * self.maze.cell_size + self.maze.cell_size // 2
        self.player.y = self.maze.start_pos[1] * self.maze.cell_size + self.maze.cell_size // 2
        self.training_data = []

    def _save_training_data(self, episode, episode_reward):
        map_size = f"{self.maze.grid_cols}x{self.maze.grid_rows}"
        loss = self.loss_history[-1] if self.loss_history else 0.0
        data = {
            'Episode': episode,
            'Reward': episode_reward,
            'Loss': loss,
            'Step_Count': self.step_count
        }
        self.training_data.append(data)
        df = pd.DataFrame([data])  # 只保存当前轮次的数据
        file_path = f"training_data_DQN_{map_size}.csv"
        # 如果文件存在则追加数据，不写表头；否则创建新文件并写表头
        if os.path.exists(file_path):
            df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            df.to_csv(file_path, index=False)
        print(f"DQN 轮次 {episode}/200, 奖励: {episode_reward:.2f}, 损失: {loss:.4f}, 步数: {self.step_count}")

    def _generate_training_charts(self):
        map_size = f"{self.maze.grid_cols}x{self.maze.grid_rows}"
        algo_name = "DQN"
        csv_file = f"training_data_{algo_name}_{map_size}.csv"

        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)

            # 奖励曲线
            plt.figure(figsize=(10, 6))
            plt.plot(df['Episode'], df['Reward'], label="Reward", color='green')
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title(f"{algo_name} Reward Curve ({map_size})")
            plt.legend()
            plt.grid(True)
            reward_file = f"reward_{algo_name}_{map_size}.png"
            plt.savefig(reward_file)
            plt.close()
            os.system(f"start {reward_file}")

            # 损失曲线
            plt.figure(figsize=(10, 6))
            plt.plot(df['Episode'], df['Loss'], label="Loss", color='red')
            plt.xlabel("Episode")
            plt.ylabel("Loss")
            plt.title(f"{algo_name} Loss Curve ({map_size})")
            plt.legend()
            plt.grid(True)
            loss_file = f"loss_{algo_name}_{map_size}.png"
            plt.savefig(loss_file)
            plt.close()
            os.system(f"start {loss_file}")

            # 步数曲线
            plt.figure(figsize=(10, 6))
            plt.plot(df['Episode'], df['Step_Count'], label="Step Count", color='orange')
            plt.xlabel("Episode")
            plt.ylabel("Step Count")
            plt.title(f"{algo_name} Step Count Curve ({map_size})")
            plt.legend()
            plt.grid(True)
            steps_file = f"steps_{algo_name}_{map_size}.png"
            plt.savefig(steps_file)
            plt.close()
            os.system(f"start {steps_file}")
        else:
            print(f"未找到 {algo_name} 的训练数据文件: {csv_file}")

    def _save_model(self):
        map_size = f"{self.maze.grid_cols}x{self.maze.grid_rows}"
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, f"dqn_model_{map_size}.pth")
        print(f"DQN 模型已保存为 dqn_model_{map_size}.pth")

    def _load_model(self):
        map_size = f"{self.maze.grid_cols}x{self.maze.grid_rows}"
        file_path = f"dqn_model_{map_size}.pth"
        if os.path.exists(file_path):
            checkpoint = torch.load(file_path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"已加载 DQN 模型和优化器状态: {file_path}")
        else:
            print(f"未找到 DQN 模型文件: {file_path}, 将使用新的模型")

class DDPGController(BaseAIController):
    def __init__(self, maze, player, game_manager):
        super().__init__(maze, player, game_manager)
        self.input_dim = 13
        self.action_dim = 4
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.001
        self.memory = deque(maxlen=10000)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.autograd.set_detect_anomaly(True)
        self.policy_net = ActorNetwork(self.input_dim, self.action_dim).to(self.device)
        self.target_policy_net = ActorNetwork(self.input_dim, self.action_dim).to(self.device)
        self.critic_net = CriticNetwork(self.input_dim, self.action_dim).to(self.device)
        self.target_critic_net = CriticNetwork(self.input_dim, self.action_dim).to(self.device)
        self.target_policy_net.load_state_dict(self.policy_net.state_dict())
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=0.002)
        self.loss_fn = nn.MSELoss()
        self.reward_history = []
        self.critic_loss_history = []
        self.policy_loss_history = []
        self.training_data = []
        self._initialize_optimizers()
        param_count_actor = sum(p.numel() for p in self.policy_net.parameters() if p.requires_grad)
        param_count_critic = sum(p.numel() for p in self.critic_net.parameters() if p.requires_grad)
        print(f"DDPG 优化器初始化完成，Actor 参数数量：{param_count_actor}，Critic 参数数量：{param_count_critic}")

    def _initialize_optimizers(self):
        self.policy_optimizer.zero_grad()
        dummy_state = torch.zeros(1, self.input_dim, device=self.device, requires_grad=True)
        dummy_action = self.policy_net(dummy_state).clone()
        dummy_policy_loss = dummy_action.mean()
        dummy_policy_loss.backward()
        self.policy_optimizer.step()
        self.policy_optimizer.zero_grad()

        self.critic_optimizer.zero_grad()
        dummy_critic_value = self.critic_net(dummy_state, dummy_action.detach()).clone()
        dummy_critic_loss = dummy_critic_value.mean()
        dummy_critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        print("DDPG 优化器状态已强制初始化（Actor 和 Critic）")

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
            return self.policy_net(state)

    def _move(self, action):
        dx, dy = 0, 0
        action_index = torch.argmax(action).item()
        if action_index == 0: dy = -1
        elif action_index == 1: dx = 1
        elif action_index == 2: dy = 1
        elif action_index == 3: dx = -1
        new_x = (self.player.x // self.maze.cell_size) + dx
        new_y = (self.player.y // self.maze.cell_size) + dy
        reward = -0.1
        done = False
        valid_move = False
        if 0 <= new_x < self.maze.grid_cols and 0 <= new_y < self.maze.grid_rows:
            if self.maze.maze_layout[new_y][new_x] == 0:
                if not self.is_training:
                    self.player.x = new_x * self.maze.cell_size + self.maze.cell_size // 2
                    self.player.y = new_y * self.maze.cell_size + self.maze.cell_size // 2
                self.path.append((new_x, new_y))
                self.step_count += 1
                valid_move = True
            else:
                reward = -50
                done = True
        else:
            reward = -50
            done = True
        if (self.player.x, self.player.y) == self.maze.end_pixel_pos:
            reward = 100
            done = True
        self.reward_history.append(reward)
        return reward, done, valid_move

    def _replay_experience(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states = torch.stack([s[0] for s in batch]).to(self.device).requires_grad_(True)
        actions = torch.stack([s[1] for s in batch]).to(self.device)
        rewards = torch.FloatTensor([s[2] for s in batch]).to(self.device)
        next_states = torch.stack([s[3] for s in batch]).to(self.device)
        dones = torch.FloatTensor([float(s[4]) for s in batch]).to(self.device)

        # Critic update
        with torch.no_grad():
            next_actions = self.target_policy_net(next_states)
            target_values = self.target_critic_net(next_states, next_actions).clone()
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * target_values
        current_q = self.critic_net(states, actions).clone()
        critic_loss = self.loss_fn(current_q, target_q)
        self.critic_loss_history.append(critic_loss.item())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), 1.0)
        self.critic_optimizer.step()

        # Policy update
        policy_actions = self.policy_net(states).clone()
        critic_output = self.critic_net(states, policy_actions).clone()
        policy_loss = -critic_output.mean()
        self.policy_loss_history.append(policy_loss.item())

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.policy_optimizer.step()

        # Soft update target networks
        for target_param, param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, events=None):
        if self.is_success:
            return
        state = self._get_state()
        action = self._choose_action(state)
        reward, done, valid_move = self._move(action)
        next_state = self._get_state()
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) >= self.batch_size:
            self._replay_experience()
        if done and (self.player.x, self.player.y) == self.maze.end_pixel_pos:
            self.is_success = True
            self.path_time = time.time() - self.start_time
            self._record_success()
            self._save_model()

    def _save_training_data(self, episode, episode_reward):
        map_size = f"{self.maze.grid_cols}x{self.maze.grid_rows}"
        critic_loss = self.critic_loss_history[-1] if self.critic_loss_history else 0.0
        policy_loss = self.policy_loss_history[-1] if self.policy_loss_history else 0.0
        data = {
            'Episode': episode,
            'Reward': episode_reward,
            'Critic_Loss': critic_loss,
            'Policy_Loss': policy_loss,
            'Step_Count': self.step_count
        }
        self.training_data.append(data)
        df = pd.DataFrame([data])  # 只保存当前轮次的数据
        file_path = f"training_data_DDPG_{map_size}.csv"
        # 如果文件存在则追加数据，不写表头；否则创建新文件并写表头
        if os.path.exists(file_path):
            df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            df.to_csv(file_path, index=False)
        print(f"DDPG 轮次 {episode}/200, 奖励: {episode_reward:.2f}, Critic 损失: {critic_loss:.4f}, Policy 损失: {policy_loss:.4f}, 步数: {self.step_count}")

    def _generate_training_charts(self):
        map_size = f"{self.maze.grid_cols}x{self.maze.grid_rows}"
        algo_name = "DDPG"
        csv_file = f"training_data_{algo_name}_{map_size}.csv"

        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)

            # 奖励曲线
            plt.figure(figsize=(10, 6))
            plt.plot(df['Episode'], df['Reward'], label="Reward", color='green')
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title(f"{algo_name} Reward Curve ({map_size})")
            plt.legend()
            plt.grid(True)
            reward_file = f"reward_{algo_name}_{map_size}.png"
            plt.savefig(reward_file)
            plt.close()
            os.system(f"start {reward_file}")

            # 损失曲线
            plt.figure(figsize=(10, 6))
            plt.plot(df['Episode'], df['Critic_Loss'], label="Critic Loss", color='red')
            plt.plot(df['Episode'], df['Policy_Loss'], label="Policy Loss", color='purple')
            plt.xlabel("Episode")
            plt.ylabel("Loss")
            plt.title(f"{algo_name} Critic and Policy Loss Curve ({map_size})")
            plt.legend()
            plt.grid(True)
            loss_file = f"loss_{algo_name}_{map_size}.png"
            plt.savefig(loss_file)
            plt.close()
            os.system(f"start {loss_file}")

            # 步数曲线
            plt.figure(figsize=(10, 6))
            plt.plot(df['Episode'], df['Step_Count'], label="Step Count", color='orange')
            plt.xlabel("Episode")
            plt.ylabel("Step Count")
            plt.title(f"{algo_name} Step Count Curve ({map_size})")
            plt.legend()
            plt.grid(True)
            steps_file = f"steps_{algo_name}_{map_size}.png"
            plt.savefig(steps_file)
            plt.close()
            os.system(f"start {steps_file}")
        else:
            print(f"未找到 {algo_name} 的训练数据文件: {csv_file}")

    def _save_model(self):
        map_size = f"{self.maze.grid_cols}x{self.maze.grid_rows}"
        torch.save({
            'actor': self.policy_net.state_dict(),
            'critic': self.critic_net.state_dict(),
            'target_actor': self.target_policy_net.state_dict(),
            'target_critic': self.target_critic_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, f"ddpg_model_{map_size}.pth")
        print(f"DDPG 模型已保存为 ddpg_model_{map_size}.pth")

    def _load_model(self):
        map_size = f"{self.maze.grid_cols}x{self.maze.grid_rows}"
        file_path = f"ddpg_model_{map_size}.pth"
        if os.path.exists(file_path):
            try:
                checkpoint = torch.load(file_path, map_location=self.device)
                self.policy_net.load_state_dict(checkpoint['actor'])
                self.critic_net.load_state_dict(checkpoint['critic'])
                self.target_policy_net.load_state_dict(checkpoint['target_actor'])
                self.target_critic_net.load_state_dict(checkpoint['target_critic'])
                self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
                self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
                print(f"已加载 DDPG 模型和优化器状态: {file_path}")
            except (KeyError, RuntimeError) as e:
                print(f"加载 DDPG 模型失败: {e}, 重新初始化优化器")
                self._initialize_optimizers()
        else:
            print(f"未找到 DDPG 模型文件: {file_path}, 将使用新的模型")
            self._initialize_optimizers()