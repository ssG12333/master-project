import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# 定义 Critic 中注意力机制的参数 (与 PPO 网络保持一致)
CRITIC_EMBEDDING_DIM = 256
CRITIC_ATTENTION_HEADS = 16
CRITIC_ATTENTION_LAYERS = 2

# 定义 SAC 中用于动作标准差的最小/最大值
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class MASACActor(nn.Module):
    def __init__(self, local_state_dim, action_dim):
        """
        MASAC 的 Actor 网络。
        接收单个代理的局部状态，输出随机策略的均值和对数标准差。
        local_state_dim: 局部状态空间维度。
        action_dim: 动作空间维度。
        """
        super(MASACActor, self).__init__()
        # Actor 网络结构：输出动作的均值和对数标准差
        self.fc1 = nn.Linear(local_state_dim, 256)
        self.fc2 = nn.Linear(256, 256)

        # 均值和对数标准差分别由不同的线性层输出
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)

    def forward(self, local_state):
        """
        前向传播。
        local_state: 单个代理的局部观测状态张量 (形状: [batch_size, local_state_dim])。
        返回动作的均值和对数标准差张量。
        """
        x = F.relu(self.fc1(local_state))
        x = F.relu(self.fc2(x))

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        # 裁剪对数标准差，以确保标准差在合理范围内，防止数值不稳定
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, local_state):
        """
        从策略中采样动作并计算对数概率。
        local_state: 单个代理的局部观测状态张量。
        返回采样动作、动作的对数概率和动作的均值。
        """
        mean, log_std = self.forward(local_state)
        std = log_std.exp()
        normal = Normal(mean, std) # 创建正态分布

        # 采样动作
        z = normal.sample()
        action = torch.tanh(z) # 使用 tanh 激活函数将动作映射到 [-1, 1]

        # 计算对数概率
        # SAC 中对数概率的计算需要考虑 tanh 变换
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6) # 1e-6 防止 log(0)
        log_prob = log_prob.sum(dim=-1, keepdim=True) # 对动作维度求和

        return action, log_prob, mean

class MASACCritic(nn.Module):
    def __init__(self, global_state_dim, total_action_dim, num_agents, local_state_dim):
        """
        MASAC 的 Critic 网络。
        接收所有代理的连接状态和连接动作，输出 Q 值。
        global_state_dim: 所有代理的连接状态空间维度。
        total_action_dim: 所有代理的连接动作空间维度。
        num_agents: 代理数量，用于 Critic 中的注意力机制。
        local_state_dim: 单个代理的局部状态维度，用于 Critic 内部的嵌入。
        """
        super(MASACCritic, self).__init__()
        self.num_agents = num_agents
        self.local_state_dim = local_state_dim

        # MASAC 通常使用两个 Q 网络以减少 Q 值过估计
        # Q1 网络
        self.q1_input_embedding = nn.Linear(local_state_dim, CRITIC_EMBEDDING_DIM)
        encoder_layer1 = nn.TransformerEncoderLayer(
            d_model=CRITIC_EMBEDDING_DIM,
            nhead=CRITIC_ATTENTION_HEADS,
            dim_feedforward=512,
            batch_first=True
        )
        self.attention_critic1 = nn.TransformerEncoder(encoder_layer1, num_layers=CRITIC_ATTENTION_LAYERS)
        # Q1 的 Critic 头，输入除了注意力输出，还需要拼接所有代理的动作
        self.q1_head = nn.Sequential(
            nn.Linear(CRITIC_EMBEDDING_DIM * num_agents + total_action_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # Q2 网络 (结构与 Q1 相同)
        self.q2_input_embedding = nn.Linear(local_state_dim, CRITIC_EMBEDDING_DIM)
        encoder_layer2 = nn.TransformerEncoderLayer(
            d_model=CRITIC_EMBEDDING_DIM,
            nhead=CRITIC_ATTENTION_HEADS,
            dim_feedforward=512,
            batch_first=True
        )
        self.attention_critic2 = nn.TransformerEncoder(encoder_layer2, num_layers=CRITIC_ATTENTION_LAYERS)
        self.q2_head = nn.Sequential(
            nn.Linear(CRITIC_EMBEDDING_DIM * num_agents + total_action_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, global_state, all_actions):
        """
        前向传播。
        global_state: 所有代理的连接全局观测状态张量 (形状: [batch_size, global_state_dim])。
        all_actions: 所有代理的连接动作张量 (形状: [batch_size, total_action_dim])。
        返回 Q1 和 Q2 的 Q 值张量 (形状: [batch_size, 1])。
        """
        batch_size = global_state.shape[0]

        # 处理 Q1 网络
        reshaped_global_state_for_embedding1 = global_state.view(batch_size * self.num_agents, self.local_state_dim)
        embedded_states1 = self.q1_input_embedding(reshaped_global_state_for_embedding1)
        embedded_states_for_attention1 = embedded_states1.view(batch_size, self.num_agents, CRITIC_EMBEDDING_DIM)
        attended_output1 = self.attention_critic1(embedded_states_for_attention1)
        flattened_attended_output1 = attended_output1.view(batch_size, -1)
        # 将注意力输出和所有代理的动作拼接
        q1_input = torch.cat([flattened_attended_output1, all_actions], dim=-1)
        q1_value = self.q1_head(q1_input)

        # 处理 Q2 网络
        reshaped_global_state_for_embedding2 = global_state.view(batch_size * self.num_agents, self.local_state_dim)
        embedded_states2 = self.q2_input_embedding(reshaped_global_state_for_embedding2)
        embedded_states_for_attention2 = embedded_states2.view(batch_size, self.num_agents, CRITIC_EMBEDDING_DIM)
        attended_output2 = self.attention_critic2(embedded_states_for_attention2)
        flattened_attended_output2 = attended_output2.view(batch_size, -1)
        # 将注意力输出和所有代理的动作拼接
        q2_input = torch.cat([flattened_attended_output2, all_actions], dim=-1)
        q2_value = self.q2_head(q2_input)

        return q1_value, q2_value

