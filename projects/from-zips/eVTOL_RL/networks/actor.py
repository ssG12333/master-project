import torch
import torch.nn as nn


class ActorNetwork(nn.Module):
    '''Deterministic policy network'''

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = None, max_action: float = 1.0):
        super().__init__()
        self.max_action = max_action

        if hidden_dims is None:
            hidden_dims = [256, 256, 128]   #隐藏层默认值

        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim)) #全连接层，将输入特征值也就是动作特征值转化为隐藏层特征
            layers.append(nn.ReLU())                       # 添加Relu激活函数
            prev_dim = hidden_dim                          # 更新输入维度

        layers.append(nn.Linear(prev_dim, action_dim))  #创建输出层

        self.net = nn.Sequential(*layers)   #将一个包含多个层的列表 layers 组合成一个完整的神经网络模型 self.net

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * torch.tanh(self.net(state))
    # self.net(state)：将输入状态 state 通过 self.net，即通过所有组合的层
    # torch.tanh(...)：
    # 使用双曲正切函数 torch.tanh 将网络的输出值缩放到 [-1, 1] 范围内。将缩放后的输出值乘以 self.max_action，将动作值缩放到 [-max_action, max_action]