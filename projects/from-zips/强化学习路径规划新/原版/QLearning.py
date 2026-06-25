import numpy as np
import random
import time

from state import GameState


class QLearning:
    # 修改初始化方法，动态适应迷宫尺寸
    def __init__(self, maze, player, game_manager):
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
        # 动态初始化Q表
        self._init_q_table()

    # 添加Q表动态更新方法
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
        if action == 0: dy = -1  # 上
        elif action == 1: dx = 1  # 右
        elif action == 2: dy = 1  # 下
        elif action == 3: dx = -1  # 左

        new_x = (self.player.x // self.maze.cell_size) + dx
        new_y = (self.player.y // self.maze.cell_size) + dy

        # 碰撞检测
        if 0 <= new_x < self.grid_cols and 0 <= new_y < self.grid_rows:
            if self.maze.maze_layout[new_y][new_x] != 1:
                self.player.x = new_x * self.maze.cell_size
                self.player.y = new_y * self.maze.cell_size
                self.path.append((new_x, new_y))
        return (new_x, new_y)

    def _get_reward(self, state):
        """改进的奖励函数：严格终点检测"""
        if state == self.maze.end_pos:
            return 100, True  # 直接返回成功
        elif self.maze.maze_layout[state[1]][state[0]] == 1:
            return -50, True  # 撞墙视为终止
        else:
            return -0.1, False

    def update(self, events=None):
        if self.is_success:
            return

        current_state = self._get_state()
        action = self._choose_action(current_state)
        new_state = self._move(action)
        reward, done = self._get_reward(new_state)

        # Q值更新
        old_value = self.q_table[current_state[0], current_state[1], action]
        next_max = np.max(self.q_table[new_state[0], new_state[1]])
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[current_state[0], current_state[1], action] = new_value

        # 衰减探索率
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

        # 严格成功检测
        if new_state == self.maze.end_pos:
            self.is_success = True
            self.path_time = time.time() - self.start_time
            self.game_manager.current_state = GameState.SUCCESS
            self._record_success()






























