import os
import random
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from network import Qnet


class DQN:
    ''' DQN算法 '''
    def __init__(self, env, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 target_update, device):
        self.env = env
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim,
                          self.action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device
        self.encode = LabelEncoder()
        self.last_nodes_str = []

        # 自适应探索率ε相关超参数
        # self.epsilon = epsilon  # 当前探索率
        # self.epsilon_max = 1.0  # ε的上界
        # self.epsilon_min = 0.01  # ε的下界
        # self.epsilon_decay = 0.995  # 衰减因子（当表现良好时使用）
        # self.epsilon_increase = 1.05  # 增加因子（当表现下降时使用）
        self.update_interval = 10  # 每隔多少个episode更新一次ε
        # self.last_avg_reward = None  # 保存上一更新周期的平均奖励

        if os.path.exists("model_params.pth"):
            self.q_net.load_state_dict(torch.load('model_params.pth', map_location=device))
            print("模型参数已加载")

    def take_action(self, state, epsilon):  # epsilon-贪婪策略采取动作
        current_node = state[0]
        possible_actions = self.env.get_possible_actions(current_node)
        filter_actions = []

        i = 0
        for action in possible_actions:
            flag = False
            for last_node_str in self.last_nodes_str:
                if action[0] == last_node_str:
                    flag = True
                    break
            if not flag:
                sign_actions = (i, action[0], action[1], action[2], action[3])
                filter_actions.append(sign_actions)
                i = i + 1

        if not filter_actions:
            return None
        if len(filter_actions) > self.action_dim:
            filter_actions = filter_actions[:self.action_dim]

        if np.random.random() < epsilon:
            action = random.choice(filter_actions)
        else:
            state = self.env.encoder.transform(state)
            # state = self.encode.fit_transform(state)
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            q_values = self.q_net(state)

            mask = self.get_action_mask(len(filter_actions))
            mask_q_values = self.handle_invalid_action(mask, q_values)
            action_index = mask_q_values.argmax().item()
            action = filter_actions[action_index]
        return action

    # def update_epsilon_adaptive(self, current_avg_reward):
    #     """
    #     根据最近的平均奖励current_avg_reward，适应性地调整ε：
    #       - 如果当前平均奖励低于上一次记录的平均奖励，增加ε以加强探索；
    #       - 否则，按常规衰减ε。
    #     """
    #     if self.last_avg_reward is not None:
    #         if current_avg_reward < self.last_avg_reward:
    #             # 当表现下降时，增加探索率，但不超过1.0
    #             self.epsilon = min(self.epsilon_max, self.epsilon * self.epsilon_increase)
    #             # tqdm.write(
    #             #     f"平均奖励下降，从{self.last_avg_reward:.2f}降至{current_avg_reward:.2f}，增加ε到{self.epsilon:.3f}")
    #         else:
    #             # 当表现提升或稳定时，正常衰减ε
    #             self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    #             # tqdm.write(
    #             #     f"平均奖励提升或稳定，从{self.last_avg_reward:.2f}到{current_avg_reward:.2f}，衰减ε到{self.epsilon:.3f}")
    #
    #     self.last_avg_reward = current_avg_reward

    def get_epsilon(self, episode, total_episodes):
        """动态ε策略（指数退火）"""
        if episode < total_episodes / 2:
            return 1 * np.exp(-1.204 * episode / (total_episodes / 2))
        else:
            return 0.3 * np.exp(
                -1.099 * (episode - total_episodes / 2) / (total_episodes / 2)
            )

    def get_action_mask(self, length):
        """
        根据当前节点，返回：
        - 一个长度为 max_neighbors 的动作掩码向量，
          有效位置为 1，其余位置为 0；
        - 一个映射列表，将输出向量中的索引映射到对应邻居节点。
        """
        # 如果邻居数量超过 max_neighbors，则截断
        if length > self.action_dim:
            length = self.action_dim
        # 构造掩码：有效位置为1，其他位置为0（填充部分）
        mask = np.zeros(self.action_dim, dtype=np.float32)
        for i in range(length):
            mask[i] = 1.0
        return mask

    def handle_invalid_action(self, mask, q_values):
        # 对无效动作赋予一个极低的值
        mask = torch.tensor([mask], dtype=torch.float32).to(self.device)
        masked_q_values = q_values.clone()
        masked_q_values[mask == 0] = -np.inf
        return masked_q_values

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device).to(torch.int64)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
            -1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
                                                                )  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1

