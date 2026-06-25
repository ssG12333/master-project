import torch
import torch.nn as nn
import torch.nn.functional as F

class MADDPGActor(nn.Module):
    def __init__(self, local_state_dim, action_dim):
        """
        MADDPG 的 Actor 网络。
        接收单个代理的局部状态，输出确定性动作。
        local_state_dim: 局部状态空间维度。
        action_dim: 动作空间维度。
        """
        super(MADDPGActor, self).__init__()
        # Actor 网络结构：简单的全连接网络
        self.fc1 = nn.Linear(local_state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, local_state):
        """
        前向传播。
        local_state: 单个代理的局部观测状态张量 (形状: [batch_size, local_state_dim])。
        返回一个确定性动作张量 (形状: [batch_size, action_dim])。
        """
        x = F.relu(self.fc1(local_state))
        x = F.relu(self.fc2(x))
        # 使用 Tanh 激活函数将动作归一化到 [-1, 1]
        action = torch.tanh(self.fc3(x))
        return action

class MADDPGCritic(nn.Module):
    def __init__(self, global_state_dim, total_action_dim):
        """
        MADDPG 的 Critic 网络。
        接收所有代理的连接状态和连接动作，输出 Q 值。
        global_state_dim: 所有代理的连接状态空间维度。
        total_action_dim: 所有代理的连接动作空间维度。
        """
        super(MADDPGCritic, self).__init__()
        # Critic 网络结构：接收全局状态和所有代理的动作
        # 输入维度为 (global_state_dim + total_action_dim)
        self.fc1 = nn.Linear(global_state_dim + total_action_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1) # 输出 Q 值

    def forward(self, global_state, all_actions):
        """
        前向传播。
        global_state: 所有代理的连接全局观测状态张量 (形状: [batch_size, global_state_dim])。
        all_actions: 所有代理的连接动作张量 (形状: [batch_size, total_action_dim])。
        返回 Q 值张量 (形状: [batch_size, 1])。
        """
        # 将全局状态和所有代理的动作拼接起来作为 Critic 的输入
        x = torch.cat([global_state, all_actions], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

