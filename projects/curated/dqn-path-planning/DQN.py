
import matplotlib.pyplot as plt
import random
import numpy as np
from Agent import DQNAgent
from env import Environment, final_states, obstacle_width
import os
import tkinter as tk
from tkinter import ttk, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import queue
import time
from collections import deque
import torch
from heapq import heappush, heappop

# 设置 Matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 超参数
TARGET_UPDATE = 3
num_episodes = 300
hidden = 128
gamma = 0.99
replay_buffer_size = 100000
batch_size = 256
eps_stop = 0.1
epsilon = eps = 0.6
Start_epsilon_decaying = 0
End_epsilon_decaying = num_episodes // 2
epsilon_decaying = (epsilon - eps_stop) / (End_epsilon_decaying - Start_epsilon_decaying)

n_actions = 8
state_space_dim = 2
starting_position = [10, 0]
target_position = [90, 100]
env = Environment(starting_position, target_position, 100, 100, n_actions)
env.max_episode_steps = 5000

class DQN_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DQN 训练")
        self.training = False
        self.testing = False
        self.queue = queue.Queue()
        self.agent = None
        self.cumulative_rewards = []
        self.num_steps = []
        self.visited_X = [starting_position[0]]
        self.visited_Y = [starting_position[1]]
        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = 2e-3
        self.last_update_time = time.time()

        # 设置 Tkinter 中文字体
        self.font = ('Noto Sans CJK SC', 12)

        # 布局
        self.setup_gui()

        # 定期检查队列
        self.root.after(200, self.check_queue)

    def setup_gui(self):
        # 主框架
        self.main_frame = ttk.Frame(self.root, padding=15)
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # 左上角参数调整
        self.param_frame = ttk.LabelFrame(self.main_frame, text="参数调整", padding=10, labelanchor="n")
        self.param_frame.grid(row=0, column=0, padx=15, pady=15, sticky="nw")
        ttk.Label(self.param_frame, text="Epsilon:", font=self.font).grid(row=0, column=0, padx=5, pady=5)
        self.epsilon_scale = ttk.Scale(self.param_frame, from_=0.01, to=1.0, value=epsilon, command=self.update_epsilon)
        self.epsilon_scale.grid(row=0, column=1, padx=5, pady=5)
        self.epsilon_label = ttk.Label(self.param_frame, text=f"{epsilon:.2f}", font=self.font)
        self.epsilon_label.grid(row=0, column=2, padx=5, pady=5)
        ttk.Label(self.param_frame, text="Gamma:", font=self.font).grid(row=1, column=0, padx=5, pady=5)
        self.gamma_scale = ttk.Scale(self.param_frame, from_=0.0, to=1.0, value=gamma, command=self.update_gamma)
        self.gamma_scale.grid(row=1, column=1, padx=5, pady=5)
        self.gamma_label = ttk.Label(self.param_frame, text=f"{gamma:.2f}", font=self.font)
        self.gamma_label.grid(row=1, column=2, padx=5, pady=5)
        ttk.Label(self.param_frame, text="学习率:", font=self.font).grid(row=2, column=0, padx=5, pady=5)
        self.lr_scale = ttk.Scale(self.param_frame, from_=1e-5, to=1e-2, value=self.learning_rate, command=self.update_lr)
        self.lr_scale.grid(row=2, column=1, padx=5, pady=5)
        self.lr_label = ttk.Label(self.param_frame, text=f"{self.learning_rate:.5f}", font=self.font)
        self.lr_label.grid(row=2, column=2, padx=5, pady=5)

        # 左下角按钮区域
        self.button_frame = ttk.LabelFrame(self.main_frame, text="控制", padding=10, labelanchor="n")
        self.button_frame.grid(row=1, column=0, padx=15, pady=15, sticky="nw")
        buttons = [
            ("更换地图", self.change_map),
            ("开始训练", self.start_training),
            ("停止训练", self.stop_training),
            ("加载模型", self.load_model),
            ("开始测试", self.start_testing),
            ("重置", self.reset),
            ("保存", self.save)
        ]
        for i, (text, cmd) in enumerate(buttons):
            ttk.Button(self.button_frame, text=text, command=cmd).grid(row=i, column=0, padx=5, pady=5, sticky="ew")

        # 中间地图区域
        self.map_frame = ttk.LabelFrame(self.main_frame, text="地图", padding=10, labelanchor="n")
        self.map_frame.grid(row=0, column=1, rowspan=2, padx=15, pady=15, sticky="nsew")
        self.main_frame.columnconfigure(1, weight=3)
        self.main_frame.rowconfigure(0, weight=3)
        self.main_frame.rowconfigure(1, weight=1)
        self.fig_map, self.ax_map = plt.subplots(figsize=(6, 6))
        self.canvas_map = FigureCanvasTkAgg(self.fig_map, master=self.map_frame)
        self.canvas_map.get_tk_widget().pack(fill="both", expand=True)
        self.update_map()

        # 右上角终端输出
        self.terminal_frame = ttk.LabelFrame(self.main_frame, text="终端输出", padding=10, labelanchor="n")
        self.terminal_frame.grid(row=0, column=2, padx=15, pady=15, sticky="ne")
        self.main_frame.columnconfigure(2, weight=1)
        self.terminal = scrolledtext.ScrolledText(self.terminal_frame, height=8, width=50, font=self.font)
        self.terminal.pack(fill="both", expand=True)
        self.terminal.insert(tk.END, "终端输出:\n")

        # 右下角奖励和步数曲线
        self.plot_frame = ttk.LabelFrame(self.main_frame, text="奖励与步数", padding=10, labelanchor="n")
        self.plot_frame.grid(row=1, column=2, padx=15, pady=15, sticky="nsew")
        self.fig_plots, (self.ax_reward, self.ax_steps) = plt.subplots(2, 1, figsize=(5, 4))
        self.canvas_plots = FigureCanvasTkAgg(self.fig_plots, master=self.plot_frame)
        self.canvas_plots.get_tk_widget().pack(fill="both", expand=True)
        self.ax_reward.set_xlabel("轮次")
        self.ax_reward.set_ylabel("累计奖励")
        self.ax_steps.set_xlabel("轮次")
        self.ax_steps.set_ylabel("步数")
        self.fig_plots.tight_layout()

    def update_epsilon(self, value):
        self.epsilon = float(value)
        self.epsilon_label.config(text=f"{self.epsilon:.2f}")
        if self.agent:
            self.agent.epsilon = self.epsilon

    def update_gamma(self, value):
        self.gamma = float(value)
        self.gamma_label.config(text=f"{self.gamma:.2f}")
        if self.agent:
            self.agent.gamma = self.gamma

    def update_lr(self, value):
        self.learning_rate = float(value)
        self.lr_label.config(text=f"{self.learning_rate:.5f}")
        if self.agent:
            for param_group in self.agent.optimizer.param_groups:
                param_group['lr'] = self.learning_rate

    def is_path_clear(self, start, target, obstacles):
        """使用 BFS 检查起点到终点是否可达"""
        grid_size = 11
        start_grid = (int(start[0] / 10), int(10 - start[1] / 10))
        target_grid = (int(target[0] / 10), int(10 - target[1] / 10))
        obstacle_set = {(x, y) for x, y in zip(obstacles[0], obstacles[1])}

        queue = deque([start_grid])
        visited = {start_grid}
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        while queue:
            x, y = queue.popleft()
            if (x, y) == target_grid:
                return True
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size and (nx, ny) not in visited and (nx, ny) not in obstacle_set:
                    queue.append((nx, ny))
                    visited.add((nx, ny))
        return False

    def astar_path(self, start, target, obstacles):
        """使用 A* 算法计算从起点到终点的最优路径，确保避开障碍物"""
        grid_size = 11  # 环境为 100x100，网格为 11x11（每格 10 单位）

        # 将实际坐标转换为网格坐标
        start_grid = (int(start[0] / 10), int(10 - start[1] / 10))
        target_grid = (int(target[0] / 10), int(10 - target[1] / 10))

        # 验证起点和终点是否在网格范围内
        if not (0 <= start_grid[0] < grid_size and 0 <= start_grid[1] < grid_size):
            self.terminal.insert(tk.END, f"错误：起点 {start} 超出网格范围\n")
            self.terminal.see(tk.END)
            return []
        if not (0 <= target_grid[0] < grid_size and 0 <= target_grid[1] < grid_size):
            self.terminal.insert(tk.END, f"错误：终点 {target} 超出网格范围\n")
            self.terminal.see(tk.END)
            return []

        # 生成障碍物网格坐标集合
        obstacle_set = set()
        for x, y in zip(obstacles[0], obstacles[1]):
            grid_x = x  # Obstacle_x 已经是网格坐标
            grid_y = y  # Obstacle_y 已经是网格坐标
            if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                obstacle_set.add((grid_x, grid_y))
            else:
                self.terminal.insert(tk.END, f"警告：障碍物 ({x}, {y}) 超出网格范围，已忽略\n")
                self.terminal.see(tk.END)

        # 验证起点和终点是否在障碍物上
        if start_grid in obstacle_set:
            self.terminal.insert(tk.END, f"错误：起点 {start} 位于障碍物上\n")
            self.terminal.see(tk.END)
            return []
        if target_grid in obstacle_set:
            self.terminal.insert(tk.END, f"错误：终点 {target} 位于障碍物上\n")
            self.terminal.see(tk.END)
            return []

        # 定义启发式函数（曼哈顿距离）
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # 初始化优先队列
        queue = [(0, start_grid, [start_grid])]
        visited = set()
        # 支持八方向移动（水平、垂直和对角线）
        directions = [
            (0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0),  # 上下左右
            (1, 1, 1.4), (1, -1, 1.4), (-1, 1, 1.4), (-1, -1, 1.4)  # 对角线
        ]

        while queue:
            cost, (x, y), path = heappop(queue)
            if (x, y) == target_grid:
                # 将网格坐标转换回实际坐标
                path_coords = [(px * 10, 100 - py * 10) for px, py in path]
                self.terminal.insert(tk.END, f"A* 路径计算成功，路径长度：{len(path_coords)}\n")
                self.terminal.see(tk.END)
                return path_coords
            if (x, y) in visited:
                continue
            visited.add((x, y))
            for dx, dy, move_cost in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size and (nx, ny) not in obstacle_set:
                    if (nx, ny) not in visited:
                        new_cost = cost + move_cost
                        heappush(queue, (new_cost + heuristic((nx, ny), target_grid), (nx, ny), path + [(nx, ny)]))

        self.terminal.insert(tk.END, "A* 算法无法找到有效路径，可能被障碍物阻挡\n")
        self.terminal.see(tk.END)
        return []

    def change_map(self):
        max_attempts = 100
        for _ in range(max_attempts):
            new_obstacle_x = [random.randint(0, 10) for _ in range(12)]
            new_obstacle_y = [random.randint(0, 10) for _ in range(12)]
            new_terminal = np.asarray([random.randint(0, 100), random.randint(0, 100)])

            start_grid = (int(starting_position[0] / 10), int(10 - starting_position[1] / 10))
            terminal_grid = (int(new_terminal[0] / 10), int(10 - new_terminal[1] / 10))
            obstacle_set = {(x, y) for x, y in zip(new_obstacle_x, new_obstacle_y)}
            if terminal_grid == start_grid or terminal_grid in obstacle_set:
                continue

            if self.is_path_clear(starting_position, new_terminal, (new_obstacle_x, new_obstacle_y)):
                env.Obstacle_x = new_obstacle_x
                env.Obstacle_y = new_obstacle_y
                env.Terminal = new_terminal
                env.reset()
                self.terminal.insert(tk.END, "地图已更换\n")
                self.terminal.see(tk.END)
                self.update_map()
                return
        self.terminal.insert(tk.END, "无法生成有效地图，请重试\n")
        self.terminal.see(tk.END)

    def start_training(self):
        if not self.training:
            self.training = True
            self.agent = DQNAgent(state_space_dim, n_actions, replay_buffer_size, batch_size, hidden, self.gamma)
            self.agent.epsilon = self.epsilon
            for param_group in self.agent.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
            threading.Thread(target=self.train, daemon=True).start()
            self.terminal.insert(tk.END, "训练开始\n")
            self.terminal.see(tk.END)

    def stop_training(self):
        self.training = False
        self.terminal.insert(tk.END, "训练停止\n")
        self.terminal.see(tk.END)

    def load_model(self):
        if self.agent and os.path.exists('../dqn_model.pth'):
            self.agent.policy_net.load_state_dict(torch.load('../dqn_model.pth'))
            self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())
            self.terminal.insert(tk.END, "模型已加载\n")
            self.terminal.see(tk.END)
        else:
            self.terminal.insert(tk.END, "无模型文件或未初始化智能体\n")
            self.terminal.see(tk.END)

    def start_testing(self):
        if not self.training and self.agent:
            self.testing = True
            threading.Thread(target=self.test, daemon=True).start()
            self.terminal.insert(tk.END, "测试开始\n")
            self.terminal.see(tk.END)

    def reset(self):
        self.cumulative_rewards = []
        self.num_steps = []
        self.visited_X = [starting_position[0]]
        self.visited_Y = [starting_position[1]]
        env.reset()
        self.update_map()
        self.update_plots()
        self.terminal.insert(tk.END, "环境已重置\n")
        self.terminal.see(tk.END)

    def save(self):
        if self.agent:
            torch.save(self.agent.policy_net.state_dict(), '../dqn_model.pth')
            np.save('../cumulative_rewards.npy', self.cumulative_rewards)
            np.save('../num_steps.npy', self.num_steps)
            np.save('../final_path.npy', np.array(list(final_states().values())))
            self.terminal.insert(tk.END, "模型和数据已保存\n")
            self.terminal.see(tk.END)

    def update_map(self):
        self.ax_map.clear()
        x_o = env.Obstacle_x
        y_o = env.Obstacle_y
        for i in range(len(x_o)):
            rect = plt.Rectangle((x_o[i] * 10, 100 - y_o[i] * 10 - 10), obstacle_width, obstacle_width,
                                 fc='blue', ec="blue")
            self.ax_map.add_patch(rect)
        self.ax_map.scatter(starting_position[0], starting_position[1], marker="s", c='red', s=100)
        self.ax_map.scatter(env.Terminal[0], env.Terminal[1], marker="s", c='green', s=100)
        self.ax_map.plot(self.visited_X, self.visited_Y, 'b-', label='DQN路径')
        if hasattr(self, 'astar_X') and self.astar_X:
            self.ax_map.plot(self.astar_X, self.astar_Y, 'orange', linestyle='--', label='A*路径')
        self.ax_map.set_xlim(0, 100)
        self.ax_map.set_ylim(0, 100)
        self.ax_map.set_xlabel('x (m)')
        self.ax_map.set_ylabel('y (m)')
        self.ax_map.grid(linestyle=':')
        self.ax_map.set_aspect('equal', adjustable='box')
        self.ax_map.legend()
        self.canvas_map.draw()

    def update_plots(self):
        self.ax_reward.clear()
        self.ax_steps.clear()
        self.ax_reward.plot(range(len(self.cumulative_rewards)), self.cumulative_rewards, 'b-', label='累计奖励')
        self.ax_steps.plot(range(len(self.num_steps)), self.num_steps, 'r-', label='步数')
        self.ax_reward.set_xlabel("轮次")
        self.ax_steps.set_xlabel("轮次")
        self.ax_reward.set_ylabel("累计奖励")
        self.ax_steps.set_ylabel("步数")
        self.ax_reward.legend()
        self.ax_steps.legend()
        self.canvas_plots.draw()

    def train(self):
        random.seed(20)
        for ep in range(num_episodes):
            if not self.training:
                break
            state = env.reset()
            done = False
            cum_reward = 0
            counter = 0
            self.visited_X = [starting_position[0]]
            self.visited_Y = [starting_position[1]]
            current_epsilon = max(eps_stop, self.epsilon - ep * epsilon_decaying)
            while not done and counter < env.max_episode_steps:
                if not self.training:
                    break
                action = self.agent.get_action(state, current_epsilon)
                next_state, next_state_flag, reward, done, _ = env.step(action)
                self.visited_X.append(env.vector_agentState[0])
                self.visited_Y.append(env.vector_agentState[1])
                cum_reward += reward
                self.agent.store_transition(state, action, next_state, reward, done)
                loss = self.agent.update_network()
                state = next_state
                counter += 1
                if counter % 10 == 0:
                    self.queue.put(('map', None))
            reached_terminal = env.isTerminal()
            if counter >= env.max_episode_steps and not done:
                cum_reward += -20
            self.queue.put(('episode', (ep, cum_reward, counter, env.shortest, env.longest, reached_terminal)))
            self.cumulative_rewards.append(cum_reward)
            self.num_steps.append(counter)
            self.queue.put(('map', None))
            if ep % TARGET_UPDATE == 0:
                self.agent.update_target_network()
            time.sleep(0.01)
        env.final()
        self.training = False
        self.queue.put(('done', None))

    def test(self):
        state = env.reset()
        done = False
        counter = 0
        self.visited_X = [starting_position[0]]
        self.visited_Y = [starting_position[1]]

        # 计算 A* 路径
        astar_path = self.astar_path(starting_position, env.Terminal, (env.Obstacle_x, env.Obstacle_y))
        self.astar_X = [x for x, y in astar_path] if astar_path else []
        self.astar_Y = [y for x, y in astar_path] if astar_path else []
        astar_steps = len(astar_path) - 1 if astar_path else 0
        astar_reached = len(astar_path) > 0 and (astar_path[-1][0] == env.Terminal[0] and astar_path[-1][1] == env.Terminal[1])

        while not done and counter < env.max_episode_steps:
            action = self.agent.get_action(state, 0.0)
            next_state, _, _, done, _ = env.step(action)
            self.visited_X.append(env.vector_agentState[0])
            self.visited_Y.append(env.vector_agentState[1])
            state = next_state
            counter += 1
            if counter % 10 == 0:
                self.queue.put(('map', None))
        if counter >= env.max_episode_steps and not done:
            done = True
        reached_terminal = env.isTerminal()
        self.queue.put(('map', None))
        self.testing = False
        self.queue.put(('test_done', (counter, reached_terminal, astar_steps, astar_reached)))

    def check_queue(self):
        try:
            if time.time() - self.last_update_time < 0.2:
                self.root.after(50, self.check_queue)
                return
            self.last_update_time = time.time()
            while True:
                msg_type, data = self.queue.get_nowait()
                if msg_type == 'episode':
                    ep, reward, steps, shortest, longest, reached_terminal = data
                    reached_text = "是" if reached_terminal else "否"
                    self.terminal.insert(tk.END, f"轮次 {ep}: 奖励 {reward:.2f}, 步数 {steps}, 最短路径 {shortest}, 最长路径 {longest}, 是否到达终点: {reached_text}\n")
                    self.terminal.see(tk.END)
                    self.update_plots()
                elif msg_type == 'map':
                    self.update_map()
                elif msg_type == 'done':
                    self.terminal.insert(tk.END, "训练完成\n")
                    self.terminal.see(tk.END)
                elif msg_type == 'test_done':
                    steps, reached_terminal, astar_steps, astar_reached = data
                    reached_text = "是" if reached_terminal else "否"
                    astar_reached_text = "是" if astar_reached else "否"
                    self.terminal.insert(tk.END, f"测试完成，DQN步数：{steps}, 是否到达终点: {reached_text}\n")
                    self.terminal.insert(tk.END, f"A*路径步数：{astar_steps}, 是否到达终点: {astar_reached_text}\n")
                    self.terminal.see(tk.END)
                    self.update_plots()
        except queue.Empty:
            pass
        self.root.after(50, self.check_queue)

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x800")
    app = DQN_GUI(root)
    root.mainloop()
