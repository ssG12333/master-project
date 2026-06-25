import torch
import torch.nn as nn
from typing import Tuple


class CriticNetwork(nn.Module):
    '''Twin Q-network for TD3/SAC'''

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256, 128]

        # Q1 network
        q1_layers = []
        prev_dim = state_dim + action_dim

        for hidden_dim in hidden_dims:       #q1网络结构
            q1_layers.append(nn.Linear(prev_dim, hidden_dim))
            q1_layers.append(nn.LayerNorm(hidden_dim))
            q1_layers.append(nn.ReLU())
            prev_dim = hidden_dim

        q1_layers.append(nn.Linear(prev_dim, 1))  #q1网络输出层
        self.q1 = nn.Sequential(*q1_layers)

        # Q2 network
        q2_layers = []
        prev_dim = state_dim + action_dim

        for hidden_dim in hidden_dims:
            q2_layers.append(nn.Linear(prev_dim, hidden_dim))
            q2_layers.append(nn.LayerNorm(hidden_dim))
            q2_layers.append(nn.ReLU())
            prev_dim = hidden_dim

        q2_layers.append(nn.Linear(prev_dim, 1))
        self.q2 = nn.Sequential(*q2_layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)  # torch.cat([state, action], dim=-1)：将状态和动作张量在最后一个维度上拼接。
        return self.q1(sa), self.q2(sa)                 # self.q1(sa)：将拼接后的张量 sa 通过第一个 Q 网络 q1，得到第一个 Q 值。
                                                        # self.q2(sa)：将拼接后的张量 sa 通过第二个 Q 网络 q2，得到第二个 Q 值。
                                                        # 返回值是一个元组 (q1_value, q2_value)，包含两个 Q 值

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)