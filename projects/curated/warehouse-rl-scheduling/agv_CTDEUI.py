import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import time
import threading
import random
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import copy
import sys


default_maze_array = np.array([
    [5, 0, 0, 0, 1, 0, 1, 0, 2, 1],
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 0, 1, 0],
    [1, 0, 1, 1, 0, 0, 4, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
    [1, 1, 3, 1, 1, 1, 1, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
])

#全局变量
Target_rewards = []
maze_array = default_maze_array.copy()
row, column = maze_array.shape
num_agents = 3  # 默认代理数量
agent_paths = [[] for _ in range(num_agents)]  # 存储路径用于可视化
is_training = False
training_thread = None
should_stop_training = False
env_instance = None  # 保存训练环境实例
controller_instance = None  # 保存训练好的控制器

# 代理/目标ID和颜色（最多支持6个代理）
Agents = [3, 4, 6, 7, 8, 9]
Targets = [2, 5, 10, 11, 12, 13]
row_sep = 50  # 默认单元格大小，UI会自动调整
column_sep = 50

color_map = {
    0: '#f8fafc', 1: '#1e293b', 2: "#fbbf24", 3: "#ef4444", 4: "#3b82f6",
    5: "#22c55e", 6: "#a855f7", 7: "#f97316", 8: "#06b6d4", 9: "#ec4899",
    10: "#facc15", 11: "#fb7185", 12: "#14b8a6", 13: "#b45309"
}
path_colors = {
    0: "#fee2e2", 1: "#dbeafe", 2: "#f3e8ff", 3: "#ffedd5", 4: "#cffafe", 5: "#fce7f3"
}
agent_colors = ['#ef4444', '#3b82f6', '#a855f7', '#f97316', '#06b6d4', '#ec4899']  # 与起始颜色匹配
target_colors = ['#fbbf24', '#22c55e', '#facc15', '#fb7185', '#14b8a6', '#b45309']  # 与目标颜色匹配
target_markers = ['*', 'P', 'D', 'X', 's', 'H']  # 不同目标标记

# 设置matplotlib字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'


# --- 环境类（基于first.py修改 - 无GUI）---
class Env:
    def __init__(self):
        self.state = None
        self.step_counter = None
        self.env_info = None
        self.agent_positions = []
        self.target_positions = []
        self.maze = None
        self.reset()  # 初始化迷宫和位置

    def _initialize_positions(self):
        """根据当前迷宫数组和代理数量初始化代理和目标位置"""
        self.agent_positions = []
        self.target_positions = []
        agent_found_count = 0
        target_found_count = 0

        for i in range(num_agents):
            agent_id = Agents[i]
            target_id = Targets[i]

            # 查找代理起始位置
            agent_pos_arr = np.transpose(np.where(maze_array == agent_id))
            if len(agent_pos_arr) > 0:
                self.agent_positions.append(agent_pos_arr[0])  # 存储为numpy数组[行, 列]
                agent_found_count += 1
            else:
                # 如果未找到则分配默认位置（后续可添加日志警告）
                self.agent_positions.append(np.array([i, 0]))  # 示例默认位置

            # 查找目标位置
            target_pos_arr = np.transpose(np.where(maze_array == target_id))
            if len(target_pos_arr) > 0:
                self.target_positions.append(target_pos_arr[0])  # 存储为numpy数组[行, 列]
                target_found_count += 1
            else:
                # 如果未找到则分配默认位置
                self.target_positions.append(np.array([row - 1 - i, column - 1]))  # 示例默认位置

        # 潜在检查：是否所有代理/目标都已找到
        if agent_found_count != num_agents or target_found_count != num_agents:
            print(f"警告：在迷宫中未找到所有{num_agents}个代理的起始/目标位置。")

    def reset(self):
        """重置环境状态"""
        global maze_array, row, column
        self.maze = copy.deepcopy(maze_array)  # 使用当前全局迷宫
        row, column = self.maze.shape  # 如果迷宫变化则更新尺寸
        self._initialize_positions()

        self.state = np.concatenate(
            [pos.flatten() for pos in self.agent_positions])  # 示例状态，可能需要调整
        self.step_counter = 0
        self.env_info = {"end": False, "action": None, "target": False}  # 保持兼容性
        return self._get_local_states()  # 返回局部状态供算法使用

    def _get_local_states(self):
        """获取每个代理的局部状态"""
        local_states = []
        all_agent_positions_flat = np.concatenate([pos.flatten() for pos in self.agent_positions])
        for i in range(num_agents):
            agent_part = self.agent_positions[i].flatten()
            target_part = self.target_positions[i].flatten()
            # 状态：自身位置(2) + 目标位置(2) + 所有代理位置(2*num_agents)
            local_state = np.concatenate([agent_part, target_part, all_agent_positions_flat])

            # 维度检查（对DQN很重要）
            expected_dim = 2 + 2 + num_agents * 2
            if local_state.shape[0] != expected_dim:
                print(
                    f"警告：代理{i}状态维度不匹配。预期{expected_dim}，实际得到{local_state.shape[0]}")
                # 如何处理不匹配？填充？报错？目前仅打印警告

            local_states.append(local_state)
        return local_states

    def _calculate_new_pos(self, current_pos, action):
        # current_pos是numpy数组[行, 列]
        x, y = current_pos
        if action == 0:
            return np.array([x - 1, y])  # 上
        elif action == 1:
            return np.array([x, y + 1])  # 右
        elif action == 2:
            return np.array([x + 1, y])  # 下
        elif action == 3:
            return np.array([x, y - 1])  # 左
        elif action == 4:
            return np.array([x, y])  # 停留
        else:
            return current_pos  # 无效动作

    def step(self, actions):
        """为所有代理执行一步操作。"""
        new_positions_calculated = []
        rewards = []
        individual_dones = [False] * num_agents

        # 先计算所有代理的潜在新位置
        for i in range(num_agents):
            current_pos = self.agent_positions[i]
            action = actions[i]
            new_pos = self._calculate_new_pos(current_pos, action)
            new_positions_calculated.append(new_pos)

        # 验证移动合法性（边界、障碍物、碰撞）
        valid_moves, final_new_positions = self._validate_moves(new_positions_calculated)

        # 更新位置并计算有效移动的奖励
        for i in range(num_agents):
            if valid_moves[i] == 1:
                # 在环境状态中更新代理位置
                self.agent_positions[i] = final_new_positions[i]
                rewards.append(self._calculate_reward(i))

                # 检查是否到达目标
                if np.array_equal(self.agent_positions[i], self.target_positions[i]):
                    individual_dones[i] = True
                    rewards[i] += Target_rewards[i]  # 目标奖励
                    Target_rewards[i] = 0
                else:
                    individual_dones[i] = False  # 未到达目标时保持未完成状态
            elif valid_moves[i] == 0:
                # 惩罚无效移动（碰撞、越界、障碍物）
                rewards.append(-5)  # 碰撞/无效移动惩罚
                # 保持当前位置（移动无效时）
                self.agent_positions[i] = self.agent_positions[i]
                individual_dones[i] = False  # 无效移动时不能标记为完成
            elif valid_moves[i] == 2:
                # 惩罚严重无效移动（碰撞、越界、障碍物）
                rewards.append(-10)  # 严重碰撞/无效移动惩罚
                # 保持当前位置（移动无效时）
                self.agent_positions[i] = self.agent_positions[i]
                individual_dones[i] = False  # 无效移动时不能标记为完成
        self.step_counter += 1
        global_done = all(individual_dones) or (self.step_counter > 200)  # 最大步数限制

        # 返回局部状态、奖励、完成状态和信息
        next_local_states = self._get_local_states()
        info = {"individual_done": individual_dones, "target_reached": individual_dones}
        return next_local_states, rewards, global_done, info

    def _validate_moves(self, proposed_positions):
        """验证移动是否越界、是否与障碍物或其他代理发生碰撞。"""
        valid = [1] * num_agents
        final_positions = [None] * num_agents
        occupied_next = {}  # 跟踪代理意图移动到的位置 {位置元组: 代理索引}

        # 首先检查边界和障碍物
        for i in range(num_agents):
            x, y = proposed_positions[i]
            if not (0 <= x < row and 0 <= y < column):
                valid[i] = 0  # 越界
            elif self.maze[x, y] == 1:
                valid[i] = 0  # 障碍物碰撞
            else:
                # 临时标记意图移动到的位置
                pos_tuple = tuple(proposed_positions[i])
                if pos_tuple in occupied_next:
                    # 检测到与其他代理的位置冲突
                    colliding_agent_idx = occupied_next[pos_tuple]
                    valid[i] = 2
                    valid[colliding_agent_idx] = 2
                else:
                    occupied_next[pos_tuple] = i

        # 检查对向碰撞（交换位置） - 更复杂的处理（按需启用）
        for i in range(num_agents):
            for j in range(i + 1, num_agents):  # 避免重复检查
                # 获取两个代理的当前和意图位置
                curr_i = self.agent_positions[i]
                curr_j = self.agent_positions[j]
                prop_i = proposed_positions[i]
                prop_j = proposed_positions[j]

                # 检查代理i是否移动到代理j的当前位置，且代理j移动到代理i的当前位置
                if np.array_equal(prop_i, curr_j) and np.array_equal(prop_j, curr_i):
                    # 检测到对向碰撞！
                    valid[i] = 2
                    valid[j] = 2
                    # 从占用列表中移除这两个位置（移动无效）
                    if tuple(prop_i) in occupied_next:
                        del occupied_next[tuple(prop_i)]
                    if tuple(prop_j) in occupied_next:
                        del occupied_next[tuple(prop_j)]

        # 确定最终位置（无效移动时保持原位）
        for i in range(num_agents):
            if valid[i]:
                final_positions[i] = proposed_positions[i]
            else:
                final_positions[i] = self.agent_positions[i]  # 保持当前位置

        return valid, final_positions

    def _calculate_reward(self, agent_idx):
        """计算单个代理的奖励值。"""
        current_pos = self.agent_positions[agent_idx]
        target_pos = self.target_positions[agent_idx]
        # 基于负距离的奖励（鼓励向目标靠近）
        distance = np.linalg.norm(current_pos - target_pos)
        # 奖励缩放（距离越近奖励值越大）
        return -distance / 5.0  # 示例缩放系数
# --- 经验回放缓冲区（改编自first.py）---
class MultiAgentReplayBuffer:
    def __init__(self, capacity: int, num_agents: int, important_scale=2):
        self.num_agents = num_agents
        self.common_buffer = deque(maxlen=capacity)  # 普通经验缓冲区
        self.important_buffer = deque(maxlen=int(capacity / important_scale))  # 重要经验缓冲区（容量更小）
        self.important_scale = important_scale  # 重要经验采样比例因子

    def add_experience(self, local_states, actions, rewards, next_local_states, done, is_important=False):
        """添加经验，根据重要性选择缓冲区存储"""
        experience = (
            [s.copy() for s in local_states],  # 当前局部状态列表
            np.array(actions, dtype=np.int64),  # 动作数组
            np.array(rewards, dtype=np.float32),  # 奖励数组
            [s.copy() for s in next_local_states],  # 下一个局部状态列表
            done  # 完成标志
        )
        if is_important and len(self.important_buffer) < self.important_buffer.maxlen:
            self.important_buffer.append(experience)  # 优先存入重要缓冲区
        else:
            # 重要缓冲区满或非重要经验时存入普通缓冲区
            self.common_buffer.append(experience)

    def sample(self, batch_size):
        """采样经验批次，优先从重要缓冲区采样"""
        important_size = min(batch_size // self.important_scale, len(self.important_buffer))
        common_size = batch_size - important_size

        sampled_transitions = []
        if len(self.important_buffer) >= important_size and important_size > 0:
            sampled_transitions.extend(random.sample(self.important_buffer, important_size))  # 从重要缓冲区采样

        remaining_common_size = min(common_size, len(self.common_buffer))
        if len(self.common_buffer) >= remaining_common_size and remaining_common_size > 0:
            sampled_transitions.extend(random.sample(self.common_buffer, remaining_common_size))  # 从普通缓冲区采样

        # 确保最终批次大小符合要求
        final_transitions = sampled_transitions[:batch_size]

        if not final_transitions:
            return None  # 数据不足时返回None

        # 为每个代理重组数据
        samples_per_agent = []
        for i in range(self.num_agents):
            states_i = []
            actions_i = []
            rewards_i = []
            next_states_i = []
            dones_list = []  # 完成标志列表（全局共享）

            for transition in final_transitions:
                s_list, a_list, r_list, ns_list, d = transition
                states_i.append(s_list[i])  # 当前状态
                actions_i.append(a_list[i])  # 执行的动作
                rewards_i.append(r_list[i])  # 获得的奖励
                next_states_i.append(ns_list[i])  # 下一个状态
                dones_list.append(d)  # 全局完成标志

            # 转换为张量
            states_tensor = torch.FloatTensor(np.array(states_i))
            actions_tensor = torch.LongTensor(np.array(actions_i))
            rewards_tensor = torch.FloatTensor(np.array(rewards_i))
            next_states_tensor = torch.FloatTensor(np.array(next_states_i))
            dones_tensor = torch.BoolTensor(np.array(dones_list))  # 全局完成标志张量

            samples_per_agent.append({
                "states": states_tensor,
                "actions": actions_tensor,
                "rewards": rewards_tensor,
                "next_states": next_states_tensor,
                "dones": dones_tensor  # 全局完成标志
            })

        return samples_per_agent

    def __len__(self):
        return len(self.common_buffer) + len(self.important_buffer)  # 返回总经验数


# --- DQN智能体网络---
class AgentDQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # 定义网络层
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # 输入层到隐藏层
            nn.ReLU(),  # 激活函数
            nn.Linear(hidden_size, hidden_size),  # 隐藏层到隐藏层
            nn.ReLU(),  # 激活函数
            nn.Linear(hidden_size, output_size)  # 隐藏层到输出层（动作值）
        )

    def forward(self, x):
        # 前向传播
        return self.net(x)


# --- CTDE多智能体DQN算法（改编自first.py）---
class CTDE_MADQN:
    def __init__(self,
                 agent_input_sizes,  # 各智能体输入维度列表
                 hidden_size,
                 output_sizes,  # 各智能体输出维度（动作空间大小）列表
                 num_agents,
                 gamma=0.99,  # 折扣因子
                 lr=0.0001,  # 学习率
                 epsilon_start=1.0,  # 初始探索率
                 epsilon_decay=0.995,  # 探索率衰减系数
                 min_epsilon=0.01,  # 最小探索率
                 target_update=6000,  # 目标网络更新周期
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):  # 计算设备

        self.num_agents = num_agents
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon_start  # 当前探索率
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.target_update = target_update  # 目标网络更新计数器
        self.update_count = 0

        # 为每个智能体创建策略网络和目标网络
        self.policy_nets = nn.ModuleList([
            AgentDQN(agent_input_sizes[i], hidden_size, output_sizes[i]).to(device)
            for i in range(num_agents)
        ])
        self.target_nets = nn.ModuleList([
            AgentDQN(agent_input_sizes[i], hidden_size, output_sizes[i]).to(device)
            for i in range(num_agents)
        ])

        # 用策略网络参数初始化目标网络
        for target_net, policy_net in zip(self.target_nets, self.policy_nets):
            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval()  # 目标网络设为评估模式

        # 为每个策略网络创建优化器
        self.optimizers = [
            torch.optim.Adam(self.policy_nets[i].parameters(), lr=lr)
            for i in range(num_agents)
        ]

    def get_actions(self, local_states, training=True):
        """根据局部状态获取所有智能体的动作"""
        actions = []
        effective_epsilon = self.epsilon if training else 0.0  # 训练时使用epsilon-greedy，测试时只用贪婪策略
        for i in range(self.num_agents):
            if random.random() < effective_epsilon:
                # 探索：随机选择动作
                actions.append(random.randint(0, self.policy_nets[i].net[-1].out_features - 1))
            else:
                # 利用：选择Q值最大的动作
                state_tensor = torch.FloatTensor(local_states[i]).unsqueeze(0).to(self.device)
                self.policy_nets[i].eval()  # 设为评估模式
                with torch.no_grad():
                    q_values = self.policy_nets[i](state_tensor)
                self.policy_nets[i].train()  # 恢复训练模式
                actions.append(torch.argmax(q_values).item())
        return actions

    def update(self, batch_samples_per_agent):
        """根据经验批次更新策略网络"""
        if batch_samples_per_agent is None:  # 检查采样是否成功
            return 0.0  # 无数据时不更新

        total_loss = 0
        for i in range(self.num_agents):
            # 获取当前智能体的批次数据
            batch = batch_samples_per_agent[i]
            states = batch["states"].to(self.device)
            actions = batch["actions"].to(self.device)
            rewards = batch["rewards"].to(self.device)
            next_states = batch["next_states"].to(self.device)
            dones = batch["dones"].to(self.device)  # 全局完成标志

            # --- 计算当前Q值 ---
            # 获取采取动作的Q值：Q(s, a)
            current_q = self.policy_nets[i](states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

            # --- 计算目标Q值 ---
            # 使用目标网络获取下一个状态的最大Q值：max_a' Q_target(s', a')
            with torch.no_grad():
                next_q_values = self.target_nets[i](next_states)
                max_next_q = next_q_values.max(1)[0]  # 获取最大Q值
                # 计算目标Q值：r + gamma * max_a' Q_target(s', a') * (1 - done)
                target_q = rewards + self.gamma * max_next_q * (~dones)

            # --- 计算损失 ---
            loss = F.mse_loss(current_q, target_q)  # 均方误差损失

            # --- 优化步骤 ---
            self.optimizers[i].zero_grad()  # 梯度清零
            loss.backward()  # 反向传播

            # torch.nn.utils.clip_grad_norm_(self.policy_nets[i].parameters(), max_norm=1.0)
            self.optimizers[i].step()  # 参数更新

            total_loss += loss.item()

        # --- 更新探索率 ---
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay  # 指数衰减

        # --- 更新目标网络 ---
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            for target_net, policy_net in zip(self.target_nets, self.policy_nets):
                target_net.load_state_dict(policy_net.state_dict())  # 硬更新目标网络

        return total_loss / self.num_agents  # 返回平均损失

    def save_model(self, path_prefix):
        """保存策略网络、目标网络和优化器的状态"""
        for i in range(self.num_agents):
            torch.save({
                'policy_net_state_dict': self.policy_nets[i].state_dict(),
                'target_net_state_dict': self.target_nets[i].state_dict(),
                'optimizer_state_dict': self.optimizers[i].state_dict(),
                'epsilon': self.epsilon
            }, f"{path_prefix}_agent_{i}.pth")
        print(f"模型已保存，前缀：{path_prefix}")

    def load_model(self, path_prefix):
        """加载策略网络、目标网络和优化器的状态"""
        device = self.device  # 确保加载到正确的设备
        for i in range(self.num_agents):
            try:
                checkpoint = torch.load(f"{path_prefix}_agent_{i}.pth", map_location=device)
                self.policy_nets[i].load_state_dict(checkpoint['policy_net_state_dict'])
                self.target_nets[i].load_state_dict(checkpoint['target_net_state_dict'])
                self.optimizers[i].load_state_dict(checkpoint['optimizer_state_dict'])

                # self.epsilon = checkpoint.get('epsilon', self.epsilon)
                self.target_nets[i].eval()  # 确保目标网络处于评估模式
                self.policy_nets[i].to(device)  # 确保模型在正确设备上
                self.target_nets[i].to(device)
                print(f"已从{path_prefix}_agent_{i}.pth加载智能体{i}的模型")
            except FileNotFoundError:
                print(f"错误：未找到智能体{i}的模型文件{path_prefix}_agent_{i}.pth")
            except Exception as e:
                print(f"加载智能体{i}的模型时出错：{e}")

# --- 控制器类（CTDE_MADQN的简化封装）---
class MultiAgentController:
    def __init__(self,
                 agent_input_sizes,
                 output_sizes,
                 num_agents,
                 hidden_size=256,  # 默认隐藏层大小
                 epsilon_start=1.0,
                 epsilon_decay=0.995,
                 min_epsilon=0.01,
                 gamma=0.99,
                 lr=0.0001,
                 target_update=6000,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

        # 初始化底层CTDE_MADQN算法
        self.ctde_madqn = CTDE_MADQN(
            agent_input_sizes=agent_input_sizes,
            hidden_size=hidden_size,
            output_sizes=output_sizes,
            num_agents=num_agents,
            gamma=gamma,
            lr=lr,
            epsilon_start=epsilon_start,
            epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon,
            target_update=target_update,
            device=device
        )
        self.train_mode = True  # 默认训练模式

    def get_actions(self, local_states):
        """从底层MADQN模型获取动作"""
        return self.ctde_madqn.get_actions(local_states, self.train_mode)

    def update_policy(self, batch_samples):
        """使用底层MADQN模型更新策略"""
        return self.ctde_madqn.update(batch_samples)

    def switch_mode(self, training=True):
        """切换训练/评估模式"""
        self.train_mode = training
        # 评估模式时调整epsilon值
        if not training:
            # 设置极小的epsilon值用于测试（几乎不探索）
            self.ctde_madqn.epsilon = 0.0  # 测试时完全禁用探索
            print("控制器已切换至评估模式（epsilon=0.0）")
        else:
            # 恢复训练epsilon（保持原有值继续训练）
            print("控制器已切换至训练模式。")

    def save_model(self, path_prefix="ctde_madqn_model"):
        """使用底层MADQN的保存方法保存模型"""
        self.ctde_madqn.save_model(path_prefix)

    def load_model(self, path_prefix="ctde_madqn_model"):
        """使用底层MADQN的加载方法加载模型"""
        self.ctde_madqn.load_model(path_prefix)


# --- 用户界面类---
class AdvancedMARL_UI(ttk.Window):
    # 在类级别定义PrintRedirector
    class PrintRedirector:
        def __init__(self, text_widget, log_func):
            self.text_widget = text_widget
            self.log_func = log_func  # 引用UI的日志方法

        def write(self, message):
            # 去除空白字符并检查非空消息
            stripped_message = message.strip()
            if self.text_widget and self.text_widget.winfo_exists() and stripped_message:
                # 使用日志方法确保线程安全和格式化
                self.log_func(stripped_message)

        def flush(self):
            # 保持stdout兼容性所需
            pass

    def __init__(self):
        super().__init__(themename="superhero")
        self.title("智能仓储多AGV路径规划系统")
        self.geometry("1400x850")  # 略微增加窗口尺寸

        # 初始化UI相关属性
        self.controller = None  # 存放MultiAgentController实例
        self.env = Env()  # 存放环境实例用于交互/可视化
        self.demo_paths = None  # 存放训练/测试后记录的路径
        self.current_figure = None  # 用于保存matplotlib图形
        self.original_stdout = sys.stdout  # 提前存储原始stdout

        # 添加奖励数据存储变量
        self.episode_rewards = []  # 存储每个训练回合的平均奖励
        self.smoothed_rewards = []  # 存储平滑后的奖励用于绘图
        self.show_realtime_rewards = True  # 是否在训练中实时显示奖励曲线

        # --- 创建主布局 ---
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=BOTH, expand=YES)

        # 状态栏
        self.status_bar = ttk.Frame(self.main_frame, bootstyle="secondary")
        self.status_bar.pack(fill=X, side=BOTTOM)
        self.status_label = ttk.Label(self.status_bar, text="就绪", padding=5)
        self.status_label.pack(side=LEFT)
        self.progress_var = ttk.IntVar(value=0)
        self.progress = ttk.Progressbar(
            self.status_bar, variable=self.progress_var, length=200, bootstyle="success-striped"
        )
        self.progress.pack(side=RIGHT, padx=10, pady=5)

        # 三列布局 - 先创建，再重定向输出
        self.create_three_column_layout()

        # 在控制台控件存在后重定向stdout
        self.redirect_stdout()

        # 绑定关闭事件
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        print("UI 初始化完成.")  # 使用重定向后的print

    def redirect_stdout(self):
        """将打印语句重定向到UI控制台"""
        # 使用类级别定义的PrintRedirector重定向stdout
        # 确保console_text存在后再创建实例
        if hasattr(self, 'console_text') and self.console_text:
            sys.stdout = self.PrintRedirector(self.console_text, self.log)
        else:
            # 备用方案（理论上不应发生）
            print("错误：未找到控制台文本控件用于重定向stdout.", file=self.original_stdout)

    def restore_stdout(self):
        """恢复原始stdout"""
        sys.stdout = self.original_stdout

    def create_three_column_layout(self):
        """创建主三列布局"""
        # 左侧面板（控制面板）
        self.left_frame = ttk.LabelFrame(self.main_frame, text="控制面板", bootstyle="primary", padding=10)
        self.left_frame.pack(side=LEFT, fill=Y, padx=5, pady=5)
        self.create_left_panel()

        # 中央面板（可视化区域）
        self.center_frame = ttk.LabelFrame(self.main_frame, text="可视化区域", bootstyle="primary", padding=10)
        self.center_frame.pack(side=LEFT, fill=BOTH, expand=YES, padx=5, pady=5)
        self.create_visualization_view()  # 初始视图

        # 右侧面板（控制台输出）
        self.right_frame = ttk.LabelFrame(self.main_frame, text="控制台输出", bootstyle="primary", padding=10,
                                          width=450)  # 调整宽度
        self.right_frame.pack(side=LEFT, fill=Y, padx=5, pady=5)
        self.right_frame.pack_propagate(False)
        self.create_right_panel()  # 创建self.console_text

    def create_left_panel(self):
        """在左侧面板创建控件"""
        # --- AGV数量选择 ---
        agv_frame = ttk.Frame(self.left_frame)
        agv_frame.pack(fill=X, pady=10)
        ttk.Label(agv_frame, text="AGV数量:").pack(side=LEFT, padx=5)
        self.agv_count_var = tk.IntVar(value=num_agents)  # 使用全局默认值
        agv_combobox = ttk.Combobox(
            agv_frame, textvariable=self.agv_count_var, values=list(range(1, len(Agents) + 1)),
            width=5, state="readonly", bootstyle="primary"
        )
        agv_combobox.pack(side=LEFT, padx=5)
        agv_combobox.bind("<<ComboboxSelected>>", self.update_agv_count)

        # --- 训练参数 ---
        param_frame = ttk.LabelFrame(self.left_frame, text="训练参数", padding=10)
        param_frame.pack(fill=X, pady=10)

        # 训练回合数
        ep_frame = ttk.Frame(param_frame)
        ep_frame.pack(fill=X, pady=5)
        ttk.Label(ep_frame, text="训练回合数:").pack(side=LEFT)
        self.episodes_var = tk.IntVar(value=1000)  # 默认回合数
        ep_entry = ttk.Entry(ep_frame, textvariable=self.episodes_var, width=8)
        ep_entry.pack(side=RIGHT, padx=5)

        # 学习率
        lr_frame = ttk.Frame(param_frame)
        lr_frame.pack(fill=X, pady=5)
        ttk.Label(lr_frame, text="学习率 (x1e-4):").pack(side=LEFT)
        self.lr_var = tk.DoubleVar(value=1.0)  # 0.001表示为10.0
        lr_scale = ttk.Scale(lr_frame, from_=1.0, to=100.0, variable=self.lr_var, bootstyle="info")
        lr_scale.pack(side=LEFT, fill=X, expand=YES, padx=5)
        lr_label = ttk.Label(lr_frame, text="10.0")
        lr_label.pack(side=LEFT, padx=5)
        lr_scale.bind("<Motion>", lambda e, v=self.lr_var, l=lr_label: l.config(text=f"{v.get():.1f}"))

        # 折扣因子
        gamma_frame = ttk.Frame(param_frame)
        gamma_frame.pack(fill=X, pady=5)
        ttk.Label(gamma_frame, text="折扣因子:").pack(side=LEFT)
        self.gamma_var = tk.DoubleVar(value=0.95)
        gamma_scale = ttk.Scale(gamma_frame, from_=0.8, to=0.999, variable=self.gamma_var, bootstyle="info")
        gamma_scale.pack(side=LEFT, fill=X, expand=YES, padx=5)
        gamma_label = ttk.Label(gamma_frame, text="0.95")
        gamma_label.pack(side=LEFT, padx=5)
        gamma_scale.bind("<Motion>", lambda e, v=self.gamma_var, l=gamma_label: l.config(text=f"{v.get():.3f}"))

        # Epsilon衰减率
        eps_frame = ttk.Frame(param_frame)
        eps_frame.pack(fill=X, pady=5)
        ttk.Label(eps_frame, text="Epsilon衰减率:").pack(side=LEFT)
        self.epsilon_decay_var = tk.DoubleVar(value=0.995)
        eps_scale = ttk.Scale(eps_frame, from_=0.9, to=0.9999, variable=self.epsilon_decay_var, bootstyle="info")
        eps_scale.pack(side=LEFT, fill=X, expand=YES, padx=5)
        eps_label = ttk.Label(eps_frame, text="0.995")
        eps_label.pack(side=LEFT, padx=5)
        eps_scale.bind("<Motion>", lambda e, v=self.epsilon_decay_var, l=eps_label: l.config(text=f"{v.get():.4f}"))

        # 目标网络更新频率
        tgt_frame = ttk.Frame(param_frame)
        tgt_frame.pack(fill=X, pady=5)
        ttk.Label(tgt_frame, text="目标网络更新 (步):").pack(side=LEFT)
        self.target_update_var = tk.IntVar(value=10)
        tgt_entry = ttk.Entry(tgt_frame, textvariable=self.target_update_var, width=8)
        tgt_entry.pack(side=RIGHT, padx=5)

        # 连续成功次数设置
        success_frame = ttk.Frame(param_frame)
        success_frame.pack(fill=X, pady=5)
        ttk.Label(success_frame, text="连续成功终止阈值:").pack(side=LEFT)
        self.success_threshold_var = tk.IntVar(value=10)
        success_entry = ttk.Entry(success_frame, textvariable=self.success_threshold_var, width=8)
        success_entry.pack(side=RIGHT, padx=5)

        # 最大步数设置
        steps_frame = ttk.Frame(param_frame)
        steps_frame.pack(fill=X, pady=5)
        ttk.Label(steps_frame, text="最大步数限制:").pack(side=LEFT)
        self.max_steps_var = tk.IntVar(value=500)
        steps_entry = ttk.Entry(steps_frame, textvariable=self.max_steps_var, width=8)
        steps_entry.pack(side=RIGHT, padx=5)

        # 添加提示语
        ttk.Label(param_frame, text="连续成功达到阈值将提前结束训练",
                  font=("SimHei", 8), foreground="gray").pack(fill=X, pady=(0, 5))

        # --- 操作按钮 ---
        button_frame = ttk.LabelFrame(self.left_frame, text="操作", padding=10)
        button_frame.pack(fill=X, pady=10, expand=YES)

        self.load_maze_btn = ttk.Button(button_frame, text="加载地图", bootstyle="primary", command=self.load_maze)
        self.load_maze_btn.pack(fill=X, pady=5)

        self.start_train_btn = ttk.Button(button_frame, text="开始训练", bootstyle="success",
                                          command=self.start_training)
        self.start_train_btn.pack(fill=X, pady=5)

        self.stop_train_btn = ttk.Button(button_frame, text="停止训练", bootstyle="danger", command=self.stop_training,
                                         state=DISABLED)
        self.stop_train_btn.pack(fill=X, pady=5)

        self.load_model_btn = ttk.Button(button_frame, text="加载模型", bootstyle="info",
                                         command=self.load_trained_model)
        self.load_model_btn.pack(fill=X, pady=5)

        self.run_test_btn = ttk.Button(button_frame, text="运行测试", bootstyle="warning", command=self.run_test)
        self.run_test_btn.pack(fill=X, pady=5)

        self.show_result_btn = ttk.Button(button_frame, text="显示结果图", bootstyle="secondary",
                                          command=self.show_path_result_view)
        self.show_result_btn.pack(fill=X, pady=5)

        self.reset_btn = ttk.Button(button_frame, text="重置环境", bootstyle="danger-outline",
                                    command=self.reset_simulation)
        self.reset_btn.pack(fill=X, pady=5)

        self.save_img_btn = ttk.Button(button_frame, text="保存结果图", bootstyle="warning-outline",
                                       command=self.download_path_image)
        self.save_img_btn.pack(fill=X, pady=5)

    def create_right_panel(self):
        """Creates the console output area and reward plot in the right panel."""
        # 使用垂直分割的Frame来布局
        self.right_vertical_frame = ttk.Frame(self.right_frame)
        self.right_vertical_frame.pack(fill=BOTH, expand=YES)

        # 上半部分 - 控制台输出区域
        console_frame = ttk.LabelFrame(self.right_vertical_frame, text="控制台输出", padding=5)
        console_frame.pack(fill=BOTH, expand=YES, side=TOP, padx=5, pady=5)

        self.console_text = tk.Text(
            console_frame, width=60, height=20, bg="#2b3e50", fg="white",
            font=("Consolas", 10), wrap=tk.WORD, relief="flat"
        )
        self.console_text.pack(fill=BOTH, expand=YES, side=LEFT)

        scrollbar = ttk.Scrollbar(console_frame, command=self.console_text.yview, bootstyle="light-round")
        scrollbar.pack(side=RIGHT, fill=Y)
        self.console_text.config(yscrollcommand=scrollbar.set)

        # 下半部分 - 奖励曲线面板
        reward_frame = ttk.LabelFrame(self.right_vertical_frame, text="实时奖励曲线", padding=5)
        reward_frame.pack(fill=BOTH, expand=YES, side=BOTTOM, padx=5, pady=5)

        # 创建matplotlib图形和嵌入到Tkinter
        self.reward_fig, self.reward_ax = plt.subplots(figsize=(4, 3), dpi=80, facecolor='#2b3e50')
        self.reward_canvas = FigureCanvasTkAgg(self.reward_fig, reward_frame)
        self.reward_canvas.get_tk_widget().pack(fill=BOTH, expand=YES)

        # 设置图形样式
        self.reward_ax.set_title('训练奖励曲线', color='white', fontsize=10)
        self.reward_ax.set_xlabel('回合', color='white', fontsize=8)
        self.reward_ax.set_ylabel('奖励', color='white', fontsize=8)
        self.reward_ax.tick_params(colors='white', labelsize=7)
        self.reward_ax.grid(True, linestyle='--', alpha=0.3)
        self.reward_ax.set_facecolor('#2b3e50')

        for spine in self.reward_ax.spines.values():
            spine.set_edgecolor('white')

        # 创建初始空曲线
        self.reward_line, = self.reward_ax.plot([], [], '-', color='#3b82f6', linewidth=1.5, label='平均奖励')
        self.reward_raw, = self.reward_ax.plot([], [], 'o', color='lightblue', alpha=0.3, markersize=1,
                                               label='每回合奖励')
        self.reward_ax.legend(loc='upper left', fontsize=7, facecolor='#34495e', labelcolor='white')

        # 添加文本显示最新奖励
        self.reward_text = self.reward_ax.text(0.02, 0.95, "当前奖励: 0.00", transform=self.reward_ax.transAxes,
                                               fontsize=8, color='white')

        self.reward_fig.tight_layout()
        self.reward_canvas.draw()

    def create_visualization_view(self):
        """在中央面板创建初始迷宫可视化视图"""
        # 清除现有内容
        for widget in self.center_frame.winfo_children():
            widget.destroy()

        # 标题（可选）
        # ttk.Label(self.center_frame, text="地图环境", font=("SimHei", 14, "bold")).pack(pady=5)

        # 用于绘制迷宫的画布
        self.maze_canvas = tk.Canvas(
            self.center_frame, bg="#1e2b37", highlightthickness=0
        )
        self.maze_canvas.pack(fill=BOTH, expand=YES, padx=5, pady=5)

        # 绑定调整大小事件以重绘迷宫
        self.maze_canvas.bind("<Configure>", self.redraw_maze_on_resize)

        # 初始绘制
        self.after(100, self.draw_maze)  # 稍作延迟确保画布尺寸计算完成

    def redraw_maze_on_resize(self, event=None):
        """画布大小改变时的回调函数，用于重绘迷宫"""
        self.draw_maze()

    def create_path_result_view(self):
        """创建路径规划结果的Matplotlib视图"""
        # 清除现有内容
        for widget in self.center_frame.winfo_children():
            widget.destroy()

        if not self.demo_paths:
            ttk.Label(self.center_frame, text="没有可用的路径数据.", font=("SimHei", 14)).pack(pady=20)
            return

        # 标题
        ttk.Label(self.center_frame, text="路径规划结果", font=("SimHei", 14, "bold")).pack(pady=5)

        try:
            fig, ax = plt.subplots(figsize=(8, 8), facecolor='#2b3e50')  # 使用UI背景色

            # --- 绘制迷宫背景 ---
            print(f"DEBUG create_path_result_view: 检查self.env...")
            if hasattr(self, 'env'):
                print(f"DEBUG: self.env类型: {type(self.env)}, id: {id(self.env)}")
                if hasattr(self.env, 'maze'):
                    print(f"DEBUG: self.env包含'maze'属性。迷宫是否为None: {self.env.maze is None}")
                    if self.env.maze is not None:
                        print(f"DEBUG: self.env.maze形状: {self.env.maze.shape}")
                else:
                    print(f"DEBUG: self.env不包含'maze'属性!")
            else:
                print(f"DEBUG: self不包含'env'属性!")

            ax.imshow(self.env.maze == 0, cmap='Greys', alpha=0.8)  # 显示可通行区域
            ax.imshow(self.env.maze == 1, cmap='binary', alpha=0.5)  # 半透明显示障碍物

            # --- 绘制起点/目标点 ---
            for i in range(num_agents):
                # 起点
                start_pos = self.env.agent_positions[i]  # 从环境重置状态获取初始位置
                # 在迷宫数组中查找实际起点标记
                start_marker_pos = np.transpose(np.where(maze_array == Agents[i]))
                if len(start_marker_pos) > 0:
                    start_pos = start_marker_pos[0]
                ax.scatter(start_pos[1], start_pos[0], color=agent_colors[i % len(agent_colors)],
                           marker='o', s=150, label=f'AGV {i + 1} 起点', zorder=5)

                # 目标点
                target_pos = self.env.target_positions[i]
                ax.scatter(target_pos[1], target_pos[0], color=target_colors[i % len(target_colors)],
                           marker=target_markers[i % len(target_markers)], s=200, label=f'AGV {i + 1} 目标', zorder=5)

            # --- 绘制路径 ---
            max_len = 0
            for i in range(min(num_agents, len(self.demo_paths))):
                path = self.demo_paths[i]
                if path:
                    max_len = max(max_len, len(path))
                    path_np = np.array(path)
                    # +++ 添加调试打印显示正在绘制的路径数据 +++
                    print(f"DEBUG 绘制Agent {i + 1}的路径（前3个点）: {path_np[:3]}")
                    # +++ 结束调试打印 +++
                    # 绘制y（列）vs x（行）
                    ax.plot(path_np[:, 1], path_np[:, 0], color=agent_colors[i % len(agent_colors)],
                            linewidth=2, label=f'AGV {i + 1} 路径', zorder=4)

            # --- 设置图表样式 ---
            ax.set_xticks(np.arange(column + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(row + 1) - 0.5, minor=True)
            ax.grid(which="minor", color="gray", linestyle=':', linewidth=0.5, alpha=0.5)
            ax.tick_params(which="minor", size=0)
            ax.set_xticks(np.arange(column))
            ax.set_yticks(np.arange(row))
            ax.set_xticklabels(np.arange(column))
            ax.set_yticklabels(np.arange(row))
            ax.set_xlim(-0.5, column - 0.5)
            ax.set_ylim(row - 0.5, -0.5)  # 反转y轴以符合矩阵惯例

            ax.set_title('路径规划结果', color='white', fontsize=14)
            ax.set_facecolor('#2b3e50')
            fig.patch.set_facecolor('#2b3e50')
            ax.tick_params(colors='white', which='both')
            for spine in ax.spines.values():
                spine.set_edgecolor('white')

            # 将图例放置在图表外部
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), facecolor='#34495e', labelcolor='white')

            # --- 嵌入到Tkinter中 ---
            canvas = FigureCanvasTkAgg(fig, self.center_frame)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill=BOTH, expand=YES, padx=10, pady=10)

            # 保存图表引用用于下载
            self.current_figure = fig

            # --- 添加统计信息 ---
            stats_frame = ttk.LabelFrame(self.center_frame, text="路径统计", padding=5)
            stats_frame.pack(fill=X, padx=10, pady=5, side=BOTTOM)
            ttk.Label(stats_frame, text=f"最大步数: {max_len}").pack(anchor="w", padx=5)
            for i in range(min(num_agents, len(self.demo_paths))):
                path_len = len(self.demo_paths[i]) if self.demo_paths[i] else 0
                ttk.Label(stats_frame, text=f"AGV {i + 1} 步数: {path_len}",
                          foreground=agent_colors[i % len(agent_colors)]).pack(anchor="w", padx=5)

        except Exception as e:
            print(f"创建结果图时出错: {e}")
            ttk.Label(self.center_frame, text=f"创建结果图时出错: {e}", wraplength=400).pack(pady=20)

    def draw_maze(self):
        """在画布上绘制当前迷宫状态"""
        if not hasattr(self, 'maze_canvas') or not self.maze_canvas.winfo_exists():
            return

        canvas = self.maze_canvas
        canvas.delete("all")

        c_width = canvas.winfo_width()
        c_height = canvas.winfo_height()
        if c_width <= 1 or c_height <= 1:  # 画布尚未准备好
            self.after(50, self.draw_maze)  # 稍后重试
            return

        # 根据当前画布尺寸和迷宫形状计算单元格大小
        cell_width = c_width / column
        cell_height = c_height / row

        # 绘制网格和单元格
        for r in range(row):
            for c in range(column):
                x0, y0 = c * cell_width, r * cell_height
                x1, y1 = (c + 1) * cell_width, (r + 1) * cell_height
                cell_value = self.env.maze[r, c]  # 使用环境的当前迷宫状态
                color = color_map.get(cell_value, '#dddddd')  # 未知值的默认颜色

                # 绘制单元格矩形
                canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="#555555", width=1)

                # 绘制起点/目标点标记（可选，可能造成杂乱）
                # if cell_value in Agents or cell_value in Targets:
                #     canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=str(cell_value), fill='black', font=('Arial', 8))

        # 如果有路径数据则绘制智能体路径
        if self.demo_paths:
            for agent_idx, path in enumerate(self.demo_paths):
                if path:
                    path_coords = []
                    for r_pos, c_pos in path:
                        center_x = c_pos * cell_width + cell_width / 2
                        center_y = r_pos * cell_height + cell_height / 2
                        path_coords.extend([center_x, center_y])
                    if len(path_coords) >= 4:
                        color = path_colors.get(agent_idx % len(path_colors), "gray")
                        canvas.create_line(path_coords, fill=color,
                                           width=max(2, int(min(cell_width, cell_height) * 0.1)), smooth=tk.TRUE,
                                           splinesteps=5)

        # 绘制当前智能体位置
        agent_radius = min(cell_width, cell_height) * 0.35  # 智能体大小相对于单元格的比例
        for i in range(num_agents):
            if i < len(self.env.agent_positions):
                r_pos, c_pos = self.env.agent_positions[i]
                center_x = c_pos * cell_width + cell_width / 2
                center_y = r_pos * cell_height + cell_height / 2
                x0 = center_x - agent_radius
                y0 = center_y - agent_radius
                x1 = center_x + agent_radius
                y1 = center_y + agent_radius
                color = agent_colors[i % len(agent_colors)]
                outline_color = "white" if color == "#1e293b" else "black"  # 对比色边框
                canvas.create_oval(x0, y0, x1, y1, fill=color, outline=outline_color, width=1)
                # 在椭圆内添加智能体编号文本
                canvas.create_text(center_x, center_y, text=str(i + 1), fill=outline_color,
                                   font=('Arial', int(agent_radius), 'bold'))

    # --- 按钮回调函数 ---

    def load_maze(self):
        """从文件加载迷宫数据"""
        global maze_array, row, column, num_agents  # 声明使用的全局变量

        # 获取当前工作目录作为初始目录并列出文件
        try:
            initial_dir = os.getcwd()
            print(f"当前工作目录: {initial_dir}")
            # 列出当前目录中的所有文件
            dir_files = os.listdir(initial_dir)
            txt_files = [f for f in dir_files if f.lower().endswith('.txt')]
            csv_files = [f for f in dir_files if f.lower().endswith('.csv')]
            print(f"当前目录中找到的txt文件: {txt_files}")
            print(f"当前目录中找到的csv文件: {csv_files}")
        except Exception as e:
            print(f"获取目录信息时出错: {e}")
            initial_dir = os.path.expanduser("~")  # 使用用户主目录

        # 使用更简单的文件类型配置
        file_path = filedialog.askopenfilename(
            title="选择地图文件 (txt或csv)",
            initialdir=initial_dir,
            filetypes=[
                ("所有地图文件", "*.txt *.csv"),  # 注意空格分隔多个模式
                ("文本文件", "*.txt"),
                ("CSV文件", "*.csv"),
                ("所有文件", "*")  # 使用"*"而不是"*.*"
            ]
        )

        # 如果用户取消了选择，直接返回
        if not file_path:
            print("用户取消了文件选择")
            return

        # 打印文件信息以便调试
        print(f"选择的文件: {file_path}")
        print(f"文件存在: {os.path.exists(file_path)}")

        try:
            # 尝试加载文件内容
            print("尝试使用np.loadtxt加载文件...")
            new_maze = np.loadtxt(file_path, dtype=int, delimiter=None)  # 自动检测分隔符
            print(f"成功读取文件内容，形状: {new_maze.shape}")

            if self.validate_maze(new_maze):
                maze_array = new_maze
                row, column = maze_array.shape
                print(f"成功加载地图: {os.path.basename(file_path)} ({row}x{column})")

                # 更新环境的迷宫数据
                if not hasattr(self, 'env'):
                    self.env = Env()
                self.env.maze = maze_array.copy()

                # 重置环境以更新智能体位置
                self.env.reset()

                # 更新显示
                self.create_visualization_view()
                self.draw_maze()

                # 更新状态
                self.log(f"已加载地图: {os.path.basename(file_path)}")
                self.update_status("就绪")
            else:
                messagebox.showerror("加载错误", f"地图文件格式无效或缺少足够的起点/终点 (需要 {num_agents} 个).")

        except Exception as e:
            print(f"加载地图文件失败: {e}")
            messagebox.showerror("加载错误", f"无法加载或解析文件:\n{e}")

    def validate_maze(self, maze_data_to_check):
        """根据当前智能体数量验证迷宫"""
        global num_agents  # 访问全局变量num_agents
        required_agents = Agents[:num_agents]
        required_targets = Targets[:num_agents]
        valid = True
        for i in range(num_agents):
            if np.sum(maze_data_to_check == required_agents[i]) != 1:
                print(f"错误: 地图缺少或有多个 AGV {i + 1} 起点 ({required_agents[i]})")
                valid = False
            if np.sum(maze_data_to_check == required_targets[i]) != 1:
                print(f"错误: 地图缺少或有多个 AGV {i + 1} 目标 ({required_targets[i]})")
                valid = False
        return valid

    def update_agv_count(self, event=None):
        """根据下拉框选择更新智能体数量"""
        global num_agents, agent_paths, maze_array  # 声明访问/修改的全局变量
        new_count = self.agv_count_var.get()
        if new_count != num_agents:
            print(f"AGV 数量更新为: {new_count}")
            num_agents = new_count
            agent_paths = [[] for _ in range(num_agents)]  # 重置路径存储
            # 使用新智能体数量重新验证当前迷宫
            if self.validate_maze(maze_array):
                self.reset_simulation()  # 使用新智能体数量重置环境
            else:
                print(f"警告: 当前加载的地图对 {num_agents} 个AGV无效。请加载合适的地图。")
                messagebox.showwarning("地图无效",
                                       f"当前加载的地图对 {num_agents} 个AGV无效。\n请加载包含 AGV 1-{num_agents} 起点/终点的地图。")
                # 如果验证失败，可以选择重置为默认地图？
                # maze_array = default_maze_array.copy()
                # self.reset_simulation()

    def start_training(self):
        """在单独线程中启动训练过程"""
        global is_training, training_thread, should_stop_training, controller_instance  # 声明修改的全局变量
        if is_training:
            messagebox.showwarning("训练中", "训练已经在进行中。请先停止当前训练。")
            return

        print("开始训练...")
        self.update_status("正在训练...")
        is_training = True
        should_stop_training = False
        self.demo_paths = None  # 清除旧路径
        self.current_figure = None

        # 禁用/启用按钮
        self.start_train_btn.config(state=DISABLED)
        self.stop_train_btn.config(state=NORMAL)
        self.load_maze_btn.config(state=DISABLED)
        self.load_model_btn.config(state=DISABLED)
        self.run_test_btn.config(state=DISABLED)

        self.progress_var.set(0)

        # 从UI获取参数
        episodes = self.episodes_var.get()
        lr = self.lr_var.get() / 10000.0  # 调整缩放比例
        gamma = self.gamma_var.get()
        epsilon_decay = self.epsilon_decay_var.get()
        target_update = self.target_update_var.get()
        success_threshold = self.success_threshold_var.get()  # 获取连续成功阈值
        max_steps = self.max_steps_var.get()  # 获取最大步数限制

        # 创建训练参数字典
        train_args = {
            'episodes': episodes,
            'lr': lr,
            'gamma': gamma,
            'epsilon_decay': epsilon_decay,
            'target_update': target_update,
            'batch_size': 256,  # 示例固定值
            'min_buffer_size': 2048,  # 示例固定值
            'success_threshold': success_threshold,  # 使用用户设置的阈值
            'hidden_size': 256,  # 示例固定值
            'buffer_capacity': 50000,  # 示例固定值
            'max_steps': max_steps  # 最大步数限制
        }

        # 在线程中启动训练
        training_thread = threading.Thread(target=self.run_training_thread, args=(train_args,), daemon=True)
        training_thread.start()

    def stop_training(self):
        """向训练线程发出停止信号"""
        global should_stop_training, is_training  # 声明使用的全局变量
        if not is_training:
            return
        print("请求停止训练...")
        should_stop_training = True
        self.update_status("正在停止训练...")
        # 按钮将在线程实际结束时重新启用

    def run_training_thread(self, args):
        """在单独线程中执行的实际训练循环"""
        global is_training, controller_instance, env_instance, should_stop_training, num_agents  # 声明访问/修改的全局变量

        try:
            print("训练线程启动.")
            # --- 初始化 ---
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"使用设备: {device}")

            # 用于训练的环境
            train_env = Env()  # 使用独立的Env实例进行训练
            env_instance = train_env  # 存储为全局变量以便外部访问/停止

            # 根据当前智能体数量计算输入/输出维度
            # 状态：自身位置(2) + 目标位置(2) + 所有智能体位置(2*num_agents)
            agent_input_size = 2 + 2 + num_agents * 2
            agent_input_sizes = [agent_input_size] * num_agents
            action_dims = [5] * num_agents  # 5个动作：上、右、下、左、停留

            # 控制器
            controller = MultiAgentController(
                agent_input_sizes=agent_input_sizes,
                output_sizes=action_dims,
                num_agents=num_agents,
                hidden_size=args['hidden_size'],
                epsilon_decay=args['epsilon_decay'],
                min_epsilon=0.01,  # 固定最小探索率
                gamma=args['gamma'],
                lr=args['lr'],
                target_update=args['target_update'],
                device=device
            )
            controller.switch_mode(training=True)  # 确保训练模式

            # 经验回放缓冲区
            replay_buffer = MultiAgentReplayBuffer(
                capacity=args['buffer_capacity'],
                num_agents=num_agents
            )

            # --- 训练循环 ---
            success_count = 0
            total_ep_rewards = []  # 存储每回合的平均奖励

            # 重置奖励数据
            self.episode_rewards = []
            self.smoothed_rewards = []
            for i in range(num_agents):
                Target_rewards.append(10)
            for episode in range(args['episodes']):
                if should_stop_training:  # 检查全局停止标志
                    print("训练被用户中断.")
                    break
                for i in range(num_agents):
                    Target_rewards[i] = 10
                local_states = train_env.reset()
                episode_reward_sum = np.zeros(num_agents)
                done = False
                step_count = 0
                max_episode_steps = args['max_steps']  # 使用参数值而非硬编码值

                while not done and step_count < max_episode_steps:
                    if should_stop_training: break  # 在内层循环中再次检查

                    # 获取动作
                    actions = controller.get_actions(local_states)

                    # 环境步进
                    next_local_states, rewards, done, info = train_env.step(actions)

                    # 将经验添加到缓冲区
                    # 如果任何智能体到达目标，则标记为重要经验
                    is_important = any(info["target_reached"])
                    replay_buffer.add_experience(local_states, actions, rewards, next_local_states, done, is_important)

                    # 更新状态和奖励
                    local_states = next_local_states
                    episode_reward_sum += rewards
                    step_count += 1

                    # 当缓冲区足够大时更新策略
                    if len(replay_buffer) > args['min_buffer_size']:
                        batch_samples = replay_buffer.sample(args['batch_size'])
                        if batch_samples:  # 检查采样是否成功
                            loss = controller.update_policy(batch_samples)
                            # 可选：每隔N步/回合记录损失
                            # if step_count % 100 == 0: print(f"Step {step_count}, Loss: {loss:.4f}")

                    # 定期更新UI（频率低于步进）
                    if step_count % 20 == 0:  # 每20步左右更新一次UI
                        # 在主线程中调度UI更新
                        self.after(0, self.update_training_visualization, copy.deepcopy(train_env.agent_positions))

                # --- 回合结束 ---
                if should_stop_training: break

                # 计算本回合的平均奖励
                avg_episode_reward = np.mean(episode_reward_sum)
                total_ep_rewards.append(avg_episode_reward)  # 智能体间的平均奖励

                # 保存奖励数据到类属性中
                self.episode_rewards.append(avg_episode_reward)

                # 计算平滑奖励（最后10个回合的移动平均）
                window_size = min(10, len(self.episode_rewards))
                smoothed_reward = np.mean(self.episode_rewards[-window_size:])
                self.smoothed_rewards.append(smoothed_reward)

                # 实时更新奖励曲线（每5个回合更新一次以减少UI负担）
                if self.show_realtime_rewards and (
                        episode % 5 == 0 or episode == args['episodes'] - 1 or all(info["target_reached"])):
                    self.after(0, self.update_reward_plot)

                avg_reward_last_10 = np.mean(total_ep_rewards[-10:]) if len(total_ep_rewards) >= 10 else np.mean(
                    total_ep_rewards)

                # 更新进度条
                self.after(0, lambda p=int((episode + 1) / args['episodes'] * 100): self.progress_var.set(p))

                if all(info["target_reached"]):
                    success_count += 1
                    print(
                        f"回合 {episode + 1}/{args['episodes']} 成功! 步数: {step_count}, 平均奖励: {avg_reward_last_10:.2f}, 探索率: {controller.ctde_madqn.epsilon:.3f}")
                else:
                    success_count = 0  # 重置连续成功计数
                    status = "超时" if step_count >= max_episode_steps else "失败"
                    print(
                        f"回合 {episode + 1}/{args['episodes']} {status}. 步数: {step_count}, 平均奖励: {avg_reward_last_10:.2f}, 探索率: {controller.ctde_madqn.epsilon:.3f}")

                # 根据连续成功次数检查提前终止
                if success_count >= args['success_threshold']:
                    print(f"连续 {args['success_threshold']} 次成功，提前结束训练!")
                    break

            # --- 训练结束 ---
            print("训练结束.")
            controller_instance = controller  # 存储训练好的控制器
            controller_instance.save_model("ctde_madqn_final")  # 保存最终模型

        except Exception as e:
            print(f"训练线程出错: {e}")
            import traceback
            print(traceback.format_exc())
        finally:
            # --- 清理 ---
            is_training = False  # 更新全局状态
            # should_stop_training 会在其他地方重置或在线程正常退出时自然为False
            env_instance = None  # 清除全局环境实例

            # 在主线程中调度UI更新
            self.after(0, self.finalize_training_ui)

    # 添加显示奖励曲线的函数
    def show_reward_curve(self):
        """显示训练过程中的奖励曲线"""
        if not self.episode_rewards:
            messagebox.showinfo("无数据", "没有奖励数据可显示。\n请先进行训练来收集奖励数据。")
            return

        print("显示奖励曲线...")

        # 清除当前视图
        for widget in self.center_frame.winfo_children():
            widget.destroy()

        # 创建绘图
        try:
            fig, ax = plt.subplots(figsize=(8, 6), facecolor='#2b3e50')

            # 绘制原始奖励数据（淡色）
            episodes = np.arange(1, len(self.episode_rewards) + 1)
            ax.plot(episodes, self.episode_rewards, 'o-', color='lightblue', alpha=0.3,
                    markersize=2, label='每回合奖励')

            # 绘制平滑的奖励曲线（更鲜艳）
            ax.plot(episodes, self.smoothed_rewards, '-', color='#3b82f6', linewidth=2.5,
                    label='平滑奖励 (10回合平均)')

            # 设置图表样式
            ax.set_title('训练奖励曲线', color='white', fontsize=14)
            ax.set_xlabel('训练回合', color='white')
            ax.set_ylabel('平均奖励', color='white')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_facecolor('#2b3e50')

            # 设置坐标轴颜色
            ax.tick_params(colors='white', which='both')
            for spine in ax.spines.values():
                spine.set_edgecolor('white')

            # 添加图例
            ax.legend(loc='upper left', facecolor='#34495e', labelcolor='white')

            # 嵌入到Tkinter
            canvas = FigureCanvasTkAgg(fig, self.center_frame)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill=BOTH, expand=YES, padx=10, pady=10)

            # 保存图表引用，以便可以下载
            self.current_figure = fig

            # 添加统计信息
            stats_frame = ttk.LabelFrame(self.center_frame, text="奖励统计", padding=5)
            stats_frame.pack(fill=X, padx=10, pady=5, side=BOTTOM)

            if len(self.episode_rewards) > 0:
                ttk.Label(stats_frame, text=f"训练回合数: {len(self.episode_rewards)}").pack(anchor="w", padx=5)
                ttk.Label(stats_frame, text=f"最终平均奖励: {self.smoothed_rewards[-1]:.2f}").pack(anchor="w", padx=5)
                ttk.Label(stats_frame, text=f"最高回合奖励: {max(self.episode_rewards):.2f}").pack(anchor="w", padx=5)
                ttk.Label(stats_frame, text=f"最低回合奖励: {min(self.episode_rewards):.2f}").pack(anchor="w", padx=5)

        except Exception as e:
            print(f"创建奖励曲线图时出错: {e}")
            ttk.Label(self.center_frame, text=f"创建奖励曲线图时出错: {e}", wraplength=400).pack(pady=20)

    def finalize_training_ui(self):
        """在训练完成或停止后更新UI"""
        global should_stop_training, controller_instance  # 访问全局变量
        self.update_status("训练完成" if not should_stop_training else "训练已停止")
        self.progress_var.set(
            100 if not should_stop_training else self.progress_var.get())  # 如果完全完成则设为100
        self.start_train_btn.config(state=NORMAL)
        self.stop_train_btn.config(state=DISABLED)
        self.load_maze_btn.config(state=NORMAL)
        self.load_model_btn.config(state=NORMAL)
        self.run_test_btn.config(state=NORMAL if controller_instance else DISABLED)  # 如果控制器存在则启用测试

    def update_training_visualization(self, agent_positions):
        """根据训练环境状态更新迷宫可视化"""
        if hasattr(self, 'env') and self.env:
            self.env.agent_positions = agent_positions  # 更新UI环境的智能体位置
            self.draw_maze()  # 重绘画布

    def load_trained_model(self):
        """加载之前训练好的模型"""
        global controller_instance, is_training, num_agents  # 声明访问/修改的全局变量
        if is_training:
            messagebox.showwarning("训练中", "请先停止当前训练再加载模型。")
            return

        # 询问用户模型路径前缀（例如 'ctde_madqn_final'）
        # 为简化操作，这里假设一个默认名称
        model_prefix = "ctde_madqn_final"  # 默认前缀

        # 检查模型文件是否存在（基本检查）
        model_exists = True
        for i in range(num_agents):
            if not os.path.exists(f"{model_prefix}_agent_{i}.pth"):
                model_exists = False
                break

        if not model_exists:
            print(f"模型文件未找到，前缀: {model_prefix}")
            messagebox.showerror("加载失败", f"找不到模型文件。\n请确保 '{model_prefix}_agent_*.pth' 文件存在。")
            return

        try:
            print(f"加载模型，前缀: {model_prefix}")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # 需要先初始化控制器结构
            agent_input_size = 2 + 2 + num_agents * 2
            agent_input_sizes = [agent_input_size] * num_agents
            action_dims = [5] * num_agents

            controller = MultiAgentController(
                agent_input_sizes=agent_input_sizes,
                output_sizes=action_dims,
                num_agents=num_agents,
                device=device
                # 其他参数如lr, gamma等在加载权重时无关紧要
            )

            # 加载权重和优化器状态
            controller.load_model(model_prefix)
            controller.switch_mode(training=False)  # 设置为评估模式

            controller_instance = controller  # 全局存储加载的控制器
            print("模型加载成功.")
            self.update_status("模型加载成功")
            self.run_test_btn.config(state=NORMAL)  # 启用测试

        except Exception as e:
            print(f"加载模型失败: {e}")
            messagebox.showerror("加载失败", f"加载模型时发生错误:\n{e}")
            controller_instance = None
            self.run_test_btn.config(state=DISABLED)

    def run_test(self):
        """使用加载/训练好的模型运行测试仿真"""
        global agent_paths, is_training, controller_instance  # 声明访问的全局变量

        if is_training:
            messagebox.showwarning("训练中", "请先停止训练再运行测试。")
            return
        if not controller_instance:
            messagebox.showerror("无模型", "请先训练或加载模型。")
            return

        print("开始测试运行...")
        self.update_status("正在运行测试...")
        self.demo_paths = None  # 清除之前的路径
        self.current_figure = None

        # 测试期间禁用按钮
        self.run_test_btn.config(state=DISABLED)
        self.start_train_btn.config(state=DISABLED)

        # 在单独线程中运行测试仿真以保持UI响应
        test_thread = threading.Thread(target=self.run_test_thread, daemon=True)
        test_thread.start()

    def run_test_thread(self):
        """在线程中执行测试仿真逻辑"""
        global agent_paths, controller_instance, num_agents  # 声明访问/修改的全局变量

        try:
            test_env = Env()  # 使用新的环境进行测试
            local_states = test_env.reset()
            controller_instance.switch_mode(training=False)  # 确保评估模式

            # 初始化路径记录
            recorded_paths = [[pos.copy()] for pos in test_env.agent_positions]  # 从初始位置开始记录
            max_steps = self.max_steps_var.get()  # 使用UI中设置的最大步数
            done = False
            step = 0

            while not done and step < max_steps:
                # 从加载的控制器获取动作
                actions = controller_instance.get_actions(local_states)

                # 环境步进
                next_local_states, rewards, done, info = test_env.step(actions)

                # 记录每个智能体的新位置
                for i in range(num_agents):
                    recorded_paths[i].append(test_env.agent_positions[i].copy())

                # 更新状态
                local_states = next_local_states
                step += 1

                # 在主线程中更新可视化
                self.after(0, self.update_test_visualization, copy.deepcopy(test_env.agent_positions),
                           copy.deepcopy(recorded_paths))
                time.sleep(0.05)  # 稍微减慢可视化速度

            # 测试完成
            agent_paths = recorded_paths  # 全局存储最终路径
            success = all(info["target_reached"])
            result_message = "成功" if success else ("超时" if step >= max_steps else "失败")
            print(f"测试完成: {result_message}, 步数: {step}")
            self.after(0, lambda msg=f"测试完成: {result_message}": self.update_status(msg))

            # +++ 将调试打印移到这里，在调度UI更新之前 +++
            if 'agent_paths' in globals() and agent_paths:
                print(f"DEBUG run_test_thread: 最终记录的agent_paths:")
                for i, path in enumerate(agent_paths):
                    # 对长路径限制打印长度
                    path_str = str(path)
                    if len(path_str) > 200:
                        path_str = path_str[:100] + "..." + path_str[-100:]
                    print(f"  智能体 {i + 1} 路径 (长度 {len(path)}): {path_str}")
            else:
                print(f"DEBUG run_test_thread: agent_paths为空或未定义。")
            # +++ 结束调试打印 +++

            # 测试完成后显示结果视图
            self.after(100, self.show_path_result_view)

        except Exception as e:
            print(f"测试运行时出错: {e}")
            import traceback
            print(traceback.format_exc())
            self.after(0, lambda: self.update_status("测试出错"))
        finally:
            # 在主线程中重新启用按钮
            self.after(0, self.finalize_test_ui)

    def finalize_test_ui(self):
        """Re-enables buttons after testing."""
        # No globals needed here, just modifies self attributes
        self.run_test_btn.config(state=NORMAL)
        self.start_train_btn.config(state=NORMAL)

    def update_test_visualization(self, agent_positions, current_paths):
        """Updates the maze canvas during testing."""
        if hasattr(self, 'env') and self.env:
            self.env.agent_positions = agent_positions  # Update UI env state
            self.demo_paths = current_paths  # Update paths for drawing
            self.draw_maze()  # Redraw

    def show_path_result_view(self):
        """Switches the center panel to show the Matplotlib result graph."""
        if not self.demo_paths:
            messagebox.showinfo("无结果", "没有路径结果可显示。\n请先训练并运行测试，或加载模型并运行测试。")
            return
        print("显示结果图...")
        self.create_path_result_view()  # Create the matplotlib view

    def reset_simulation(self):
        """Resets the environment, visualization, and controller state."""
        global maze_array, controller_instance, agent_paths, default_maze_array, row, column, is_training, num_agents  # Declare globals used/modified
        print("重置环境...")
        if is_training:
            messagebox.showwarning("训练中", "请先停止训练再重置。")
            return

        if 'default_maze_array' not in globals():
            # Re-define it if somehow lost (should not happen normally)
            default_maze_array = np.array([
                [5, 0, 0, 0, 1, 0, 1, 0, 2, 1],
                [0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [1, 0, 1, 1, 0, 0, 0, 0, 1, 0],
                [1, 0, 1, 1, 0, 0, 4, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
                [1, 1, 3, 1, 1, 1, 1, 0, 0, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            ])

        maze_array = default_maze_array.copy()  # Reset to default maze
        # global row, column # Already declared above
        row, column = maze_array.shape  # Update globals
        self.env.reset()  # Reset the UI's environment instance
        controller_instance = None  # Clear loaded/trained controller
        self.demo_paths = None  # Clear paths
        self.current_figure = None

        # 清空奖励数据
        self.episode_rewards = []
        self.smoothed_rewards = []

        # 重置奖励曲线
        if hasattr(self, 'reward_line') and hasattr(self, 'reward_raw') and hasattr(self, 'reward_text') and hasattr(
                self, 'reward_canvas'):
            self.reward_line.set_data([], [])
            self.reward_raw.set_data([], [])
            self.reward_text.set_text("当前奖励: 0.00")
            self.reward_ax.set_xlim(0, 10)
            self.reward_ax.set_ylim(-1, 1)
            self.reward_canvas.draw()

        # Reset UI elements
        self.agv_count_var.set(num_agents)  # Reflect current num_agents
        self.create_visualization_view()  # Show initial maze
        self.log("环境已重置为默认设置.")  # Use log method
        self.update_status("就绪")
        self.progress_var.set(0)
        self.run_test_btn.config(state=DISABLED)  # Disable test until model is ready

    def download_path_image(self):
        """Saves the current Matplotlib figure."""
        if not self.current_figure:
            messagebox.showerror("无图像", "没有可保存的结果图。\n请先运行测试并显示结果图。")
            return

        file_path = filedialog.asksaveasfilename(
            title="保存路径图",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
            defaultextension=".png"
        )
        if not file_path: return

        try:

            self.current_figure.savefig(file_path, dpi=300, bbox_inches='tight')

            print(f"结果图已保存至: {file_path}")
            self.update_status(f"图像已保存: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"保存图像失败: {e}")
            messagebox.showerror("保存失败", f"无法保存图像:\n{e}")

    # --- Logging and Status Updates ---
    def log(self, message):
        """Adds a message to the UI console (thread-safe)."""
        # Ensure message is a string and not empty/whitespace
        msg_str = str(message).strip()
        if not msg_str:
            return

        if hasattr(self, 'console_text') and self.console_text.winfo_exists():
            try:
                # Use 'after' to schedule the UI update in the main thread
                def update_console():
                    # Double-check existence inside the scheduled function
                    if hasattr(self, 'console_text') and self.console_text.winfo_exists():
                        current_time = time.strftime("%H:%M:%S")
                        self.console_text.insert(tk.END, f"[{current_time}] {msg_str}\n")
                        self.console_text.see(tk.END)  # Scroll to the end

                self.after(0, update_console)
            except tk.TclError:  # Handle case where window is destroyed between check and schedule
                # If UI is gone, log to original stdout
                print(f"[UI Closed Log] {msg_str}", file=self.original_stdout)
        else:
            # Fallback if console is not ready yet or destroyed
            print(f"[LOG - Console unavailable] {msg_str}", file=self.original_stdout)

    def update_status(self, message):
        """Updates the status bar text (thread-safe)."""
        if hasattr(self, 'status_label') and self.status_label.winfo_exists():
            # Use 'after' to schedule the UI update in the main thread
            self.after(0, lambda msg=message: self.status_label.config(
                text=msg) if self.status_label.winfo_exists() else None)

    # --- Window Closing ---
    def on_closing(self):
        """Handles window closing event."""
        global is_training, should_stop_training, env_instance, training_thread  # Declare globals used
        print("关闭应用程序...", file=self.original_stdout)  # Log shutdown start to original stdout
        if is_training:
            should_stop_training = True  # Signal training thread to stop
            print("等待训练线程结束...", file=self.original_stdout)
            # Ideally, wait for the thread to join, but might hang UI
            if training_thread and training_thread.is_alive():
                # Give the thread a moment to stop gracefully
                training_thread.join(timeout=1.0)
                if training_thread.is_alive():
                    print("Warning: Training thread did not exit cleanly.", file=self.original_stdout)

        self.restore_stdout()
        try:
            self.destroy()
        except tk.TclError as e:
            print(f"Error during destroy: {e}", file=self.original_stdout)

    # 在类的方法定义中添加更新奖励曲线的函数
    def update_reward_plot(self):
        """更新小窗口中的奖励曲线"""
        if len(self.episode_rewards) > 0:
            # 更新数据
            episodes = np.arange(1, len(self.episode_rewards) + 1)
            self.reward_line.set_data(episodes, self.smoothed_rewards)
            self.reward_raw.set_data(episodes, self.episode_rewards)

            # 自动调整坐标轴范围
            self.reward_ax.set_xlim(0, max(10, len(self.episode_rewards)))

            # 设置y轴范围，添加一些余量
            min_reward = min(self.episode_rewards) if self.episode_rewards else 0
            max_reward = max(self.episode_rewards) if self.episode_rewards else 0
            y_margin = max(1, (max_reward - min_reward) * 0.1)  # 至少1的余量或10%的范围
            self.reward_ax.set_ylim(min_reward - y_margin, max_reward + y_margin)

            # 更新最新奖励文本
            if len(self.smoothed_rewards) > 0:
                self.reward_text.set_text(f"当前奖励: {self.smoothed_rewards[-1]:.2f}")

            # 重绘曲线
            self.reward_fig.tight_layout()
            self.reward_canvas.draw()


if __name__ == "__main__":
    print("启动应用程序...")
    app = AdvancedMARL_UI()
    app.mainloop()
    print("应用程序已关闭.")