import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
from collections import deque

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from reward_model import Critic, Reward


class RewardFunction():
    def __init__(self, env, args, device):
        """
        Initialize the reward function used for upper-level optimization

        Args:
            env: Gym-like environment
            args:
            device: "cpu" or "cuda"
        """
        activation_function_list = {
            "relu": torch.relu,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "None": None
        }
        self.hidden_dim = args.hidden_dim
        self.encode_dim = args.encode_dim
        self.gamma = args.gamma
        self.lr = args.reward_lr
        self.activate_function = activation_function_list[args.activate_function]
        self.last_activate_function = activation_function_list[args.last_activate_function]
        self.device = device
        self.state_dim = env.observation_space.shape
        self.action_dim = env.action_space.shape
        self.use_advantage = getattr(args, 'use_advantage', False)
        self.use_adamw = getattr(args, 'use_adamw', False)
        self.n_samples = getattr(args, 'n_samples', 10)

        # 计算输入维度的大小（处理元组情况）
        state_dim_size = int(np.prod(self.state_dim))
        action_dim_size = int(np.prod(self.action_dim))

        self.value_function = Critic(layer_num=3, input_dim=state_dim_size, output_dim=1,\
                                    hidden_dim=self.hidden_dim,
                                    activation_function=self.activate_function, last_activation = None).to(device=self.device)

        if self.use_adamw:
            self.value_function_optimizer = optim.AdamW(self.value_function.parameters(), lr=self.lr, weight_decay=0.01)
        else:
            self.value_function_optimizer = optim.Adam(self.value_function.parameters(), lr=self.lr)

        self.reward_function = Reward(state_dim=state_dim_size, action_dim=action_dim_size, hidden_dim=self.hidden_dim, \
                                     encode_dim=self.encode_dim,\
                                     output_dim=1,\
                                     activation_function=self.activate_function,\
                                     last_activation=self.last_activate_function)\
                                    .to(self.device)

        if self.use_adamw:
            self.reward_function_optimizer = optim.AdamW(self.reward_function.parameters(), lr=self.lr, weight_decay=0.01)
        else:
            self.reward_function_optimizer = optim.Adam(self.reward_function.parameters(), lr=self.lr)

        self.D_xi = deque(maxlen=args.reward_buffer_size)
        self.n_samples = getattr(args, 'n_samples', 10)

    def observe_reward(self, state, action, next_state=None):
        """
        Give the current reward function on a given state-action pair
        Args:
            state: Current state
            action: Taken action
            next_state (optional): Next state (unused)

        Returns:
            Reward signal for the state-action pair
        """
        state = torch.Tensor(state).to(self.device)
        action = torch.Tensor(action).to(self.device)
        # 确保action是二维张量
        if action.dim() == 1:
            action = action.unsqueeze(1)
        reward = self.reward_function.forward(state, action).detach().cpu().numpy()
        return reward

    def optimize_reward(self, agent, use_advantage=False, value_network=None):
        """
        Perform the upper-level optimization step to update the reward function

        Args:
            agent: The policy agent
            use_advantage: Whether to use advantage function instead of Q value
            value_network: The value network for advantage calculation
        """
        if len(self.D_xi) < 1:
            return

        # 从经验回放中获取数据
        D_new = [step for traj in self.D_xi for step in traj]
        if len(D_new) < 1:
            return
        
        np.random.shuffle(D_new)
        
        # 首先优化value function
        states_batch = [step[0] for step in D_new]
        overline_V_batch = [step[5] for step in D_new]
        self.optimize_value_function(np.array(states_batch), np.array(overline_V_batch), use_advantage=use_advantage, value_network=value_network)
        
        # 然后计算reward function的loss
        accumulator_1 = []
        accumulator_2 = []
        
        for step in D_new:
            s, a, reward_hat, log_probs, mu, overline_V = step
            
            # 计算状态值V(s)
            if use_advantage and value_network is not None:
                V_s = value_network.forward(torch.Tensor(s).to(self.device))
            else:
                V_s = self.value_function.forward(torch.Tensor(s).to(self.device))
            
            # 计算当前动作的概率
            if isinstance(mu, np.ndarray):
                mu = torch.Tensor(mu).to(self.device)
            
            # 确保mu的形状正确
            if mu.dim() == 1:
                mu = mu.unsqueeze(0)
            
            probs = F.softmax(mu, dim=-1)
            
            # 确保a的索引在有效范围内
            a = int(a)
            if a >= probs.shape[-1]:
                a = probs.shape[-1] - 1
            
            prob_a = probs[0, a]
            
            # 计算优势函数或价值函数
            if use_advantage and value_network is not None:
                q_values_all = agent.get_q_values(torch.Tensor(s).to(self.device))
                q_a = q_values_all.gather(1, torch.LongTensor([a]).to(self.device)).squeeze()
                advantage = q_a - V_s.squeeze()
                accumulator_2.append(prob_a * advantage)
            else:
                # 确保overline_V是张量类型
                if isinstance(overline_V, np.ndarray):
                    overline_V = torch.Tensor(overline_V).to(self.device)
                accumulator_2.append(prob_a * (overline_V - V_s.squeeze()))
            
            # 计算期望奖励
            action_bs, log_probs_action_bs = agent.get_action_prob_from_mu(mu, self.n_samples)
            # 确保action_bs的形状正确且类型为float
            action_bs = action_bs.squeeze().unsqueeze(1).float()  # 从(1, 10)变为(10, 1)并转换为float
            s_expanded = torch.tensor(np.tile(s, (self.n_samples, 1)), dtype=torch.float32, device=self.device)
            reward_bs = self.reward_function(s_expanded, action_bs)
            probs_action_bs = torch.exp(log_probs_action_bs)
            reward_center = torch.sum(probs_action_bs * reward_bs, dim=0)
            accumulator_1.append(torch.Tensor(reward_hat).to(self.device) - reward_center)
        
        # 计算loss
        if len(accumulator_1) > 0 and len(accumulator_2) > 0:
            loss = torch.mean(torch.stack(accumulator_2)) * torch.mean(torch.stack(accumulator_1))
            
            # 优化reward function
            self.reward_function_optimizer.zero_grad()
            loss.backward()
            self.reward_function_optimizer.step()

    def optimize_value_function(self, states_batch, overline_V_batch, use_advantage=False, value_network=None):
        """
        优化价值函数V(s)，使其逼近标准化回报overline_V

        Args:
            states_batch: 状态批次
            overline_V_batch: 目标价值批次
            use_advantage: 是否使用优势模式
            value_network: 外部价值网络（如果为None，使用内部网络）
        """
        states_batch = torch.Tensor(states_batch).to(self.device)
        overline_V_batch = torch.Tensor(overline_V_batch).to(self.device)

        if use_advantage and value_network is not None:
            pred_batch = value_network.forward(states_batch)
        else:
            pred_batch = self.value_function.forward(states_batch)
        
        # 统一展平为一维张量进行比较，避免维度不匹配警告
        pred_batch = pred_batch.squeeze()
        overline_V_batch = overline_V_batch.squeeze()

        loss = torch.nn.functional.smooth_l1_loss(pred_batch, overline_V_batch)

        self.value_function_optimizer.zero_grad()
        loss.backward()
        self.value_function_optimizer.step()

    def store_V(self, epidata):
        """
        Calculate and store standardized return overline_V for each step in a trajectory

        Args:
            epidata: A list of trajectory steps.

        Returns:
            A new list of steps with computed overline_V added to each step.
        """
        new_epidata = []
        overline_V = 0  # 初始化overline_V为0
        for step in reversed(epidata):
            overline_V = step.reward + self.gamma * overline_V
            updated_step = step._replace(overline_V=overline_V)
            new_epidata.insert(0, updated_step)
        return new_epidata
