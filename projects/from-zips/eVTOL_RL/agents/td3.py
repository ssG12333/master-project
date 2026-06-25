import numpy as np
import torch
import torch.nn.functional as F
from .base import BaseAgent
from networks.actor import ActorNetwork
from networks.critic import CriticNetwork
from typing import Dict, Tuple, Optional

class TD3Agent(BaseAgent):
    '''TD3 agent with Prioritized Experience Replay support'''

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            max_action: float = 1.0,  # 动作输出的最大绝对值，用来把网络输出裁剪到 [-1, 1] 或其他范围
            device: str = 'cpu',
            lr_actor: float = 3e-4,   # 策略网络（Actor）的学习率
            lr_critic: float = 3e-4,  # 评价网络（Critic）的学习率
            gamma: float = 0.99,      # 折扣因子
            tau: float = 0.005,       # 软更新系数，目标网络每次只跟随主网络 0.5% 的参数，避免突变
            policy_noise: float = 0.2, # 计算目标动作时添加的高斯噪声标准差（相对 max_action）
            noise_clip: float = 0.5,  # 噪声被裁剪到的最大绝对值，防止动作被扰动得太过分。
            policy_delay: int = 2     # Actor 网络 每 2 次 critic 更新后才更新一次，减少震荡。
    ):
        super().__init__(state_dim, action_dim, device)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay

        # Networks
        self.actor = ActorNetwork(state_dim, action_dim, max_action=max_action).to(device)
        self.actor_target = ActorNetwork(state_dim, action_dim, max_action=max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())  #将主网络的参数复制给目标网络

        self.critic = CriticNetwork(state_dim, action_dim).to(device)
        self.critic_target = CriticNetwork(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.total_iterations = 0 #训练次数计数器

    def get_q_values(self, state: np.ndarray, action: np.ndarray) -> Dict[str, float]:
        """获取给定状态和动作的Q值 """
        try:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)   # unsqueeze(0)在第0维插入一个维度
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)

            with torch.no_grad():
                q1, q2 = self.critic(state_tensor, action_tensor)
                q1_val = q1.item()        # q1的值
                q2_val = q2.item()        # q2的值
                q_min = min(q1_val, q2_val)

            return {
                'q1': q1_val,
                'q2': q2_val,
                'q_min': q_min
            }
        except Exception as e:
            print(f"Warning: Failed to get Q-values: {e}")
            return {'q1': 0.0, 'q2': 0.0, 'q_min': 0.0}

    def select_action_with_analysis(self, state: np.ndarray, evaluate: bool = False) -> Dict:
        """选择动作并返回详细分析信息 """
        try:
            # 使用现有的select_action方法获取详细分析
            action, analysis = self.select_action(state, evaluate=evaluate, return_q_values=True)

            # 将现有的analysis格式转换为验证系统期望的格式
            q_values = {
                'q1': analysis['q1_clean'],
                'q2': analysis['q2_clean'],
                'q_min': min(analysis['q1_clean'], analysis['q2_clean'])
            }

            # 计算置信度（基于Q1和Q2的一致性）
            q_diff = analysis['q1_q2_diff']
            confidence = 1.0 / (1.0 + q_diff)  # Q值越一致，置信度越高

            return {
                'action': action,
                'q_values': q_values,
                'confidence': confidence,
                'detailed_analysis': analysis  # 保留原有的详细分析
            }

        except Exception as e:
            print(f"Warning: Failed to get action analysis: {e}")
            # 回退到标准动作选择
            action = self.select_action(state, evaluate)
            return {
                'action': action,
                'q_values': None,
                'confidence': 0.0,
                'detailed_analysis': None
            }
    def select_action(self, state: np.ndarray, evaluate: bool = False, return_q_values: bool = False):
        '''Select action with optional exploration noise and Q-value analysis
           输入状态numpy数组；在evaluate模式下，不添加噪声；
           默认状态下，只返回动作return_q_values: bool = False，当return_q_values: bool = True时，额外返回两个Critic的Q值和噪声分析信息
        '''
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        #state_tensor = torch.FloatTensor(state)把 np.ndarray 转成 torch.FloatTensor（CPU 上）。.unsqueeze(0)在最前面加一维，把形状从 [state_dim] → [1, state_dim]，符合网络 batch 输入格式

        with torch.no_grad():
            action_clean = self.actor(state_tensor).cpu().numpy()[0] #直接由actor网络得到的没有噪声的动作；[0]去掉batch维

            # For analysis, get Q-values with and without noise Q值分析
            if return_q_values:  #如果要分析Q值
                # Clean action Q-values 无噪声q值计算
                action_clean_tensor = torch.FloatTensor(action_clean).unsqueeze(0).to(self.device)
                q1_clean, q2_clean = self.critic(state_tensor, action_clean_tensor)
                q_min_clean = torch.min(q1_clean, q2_clean).item()  # .item转成python标量

                # Add exploration noise 添加噪声
                if not evaluate: # 进行q值分析，且不是评估阶段，加入噪声
                    noise = np.random.normal(0, 0.1 * self.max_action, size=action_clean.shape) #均值为0，标准差为0.1 * self.max_action的随机正态分布噪声
                    action_noisy = np.clip(action_clean + noise, -self.max_action, self.max_action) #裁剪动作
                else: # 进行q值分析，且是评估阶段，不开噪声
                    action_noisy = action_clean.copy()

                # Noisy action Q-values 带噪声q值计算
                action_noisy_tensor = torch.FloatTensor(action_noisy).unsqueeze(0).to(self.device)
                q1_noisy, q2_noisy = self.critic(state_tensor, action_noisy_tensor)
                q_min_noisy = torch.min(q1_noisy, q2_noisy).item()

                analysis = {
                    'q1_q2_diff': abs(q1_clean.item() - q2_clean.item()), # 两q值之差
                    'q_value_diff': q_min_clean - q_min_noisy,            # 加噪声的q值和不加噪声的q值之差
                    'action_perturbation': np.linalg.norm(action_noisy - action_clean), # 噪声大小的量化，L2范数
                    'q1_clean': q1_clean.item(),
                    'q2_clean': q2_clean.item(),
                    'q1_noisy': q1_noisy.item(),
                    'q2_noisy': q2_noisy.item()
                }
                return action_noisy if not evaluate else action_clean, analysis

            else: #不进行q值分析
                if not evaluate: #不进行q值分析时，且是训练阶段
                    noise = np.random.normal(0, 0.1 * self.max_action, size=action_clean.shape)
                    action = np.clip(action_clean + noise, -self.max_action, self.max_action)
                else: #不进行q值分析,且是evaluate阶段
                    action = action_clean
                return action

    def train(self, replay_buffer, batch_size: int = 256) -> dict:    #智能体的训练过程
        '''Train the agent with prioritized experience replay support'''
        self.total_iterations += 1

        # Check if using prioritized replay buffer
        use_per = hasattr(replay_buffer, 'update_priorities') #hasattr(object, name),返回true或False

        if use_per:
            # Sample batch with priorities and importance weights
            state, action, reward, done, next_state, indices, weights = replay_buffer.sample(batch_size)
            weights = torch.FloatTensor(weights).to(self.device)  #indices样本索引，weights重要性采样权重
        else:
            # Standard sampling
            state, action, reward, done, next_state = replay_buffer.sample(batch_size)
            weights = torch.ones(batch_size, 1).to(self.device)  #权重全设为1
            indices = None

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)

        with torch.no_grad(): # target_actor目标策略网络做预测
            # Select action with clipped noise。 self.policy_noise噪声标准差，self.noise_clip噪声上限
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute target Q value 两个target_critic目标价值网络做预测
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            y = reward + (1 - done) * self.gamma * target_q #贝尔曼方程，计算TD目标值

        # Get current Q estimates 两个critic价值网络做预测
        current_q1, current_q2 = self.critic(state, action)

        # Compute TD errors for prioritized replay
        td_errors1 = y - current_q1
        td_errors2 = y - current_q2
        td_errors = torch.min(torch.abs(td_errors1), torch.abs(td_errors2)) #用于per的td误差计算

        # Compute weighted critic loss #计算目标q值和实际q值的均方误差
        critic_loss1 = (weights * (td_errors1 ** 2)).mean()
        critic_loss2 = (weights * (td_errors2 ** 2)).mean()
        critic_loss = critic_loss1 + critic_loss2

        # Optimize critic 优化两个Critic网络
        self.critic_optimizer.zero_grad()                                        #清零梯度
        critic_loss.backward()                                                   #反向传播：计算 Critic 损失的梯度，并将其传播到 Critic 网络的参数中。这一步会计算每个参数的梯度，为后续的优化做准备
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)  # 梯度裁剪：将 Critic 网络的梯度范数限制在 1.0 以内。这一步可以防止梯度爆炸，确保训练过程的稳定性。
        self.critic_optimizer.step()

        # Update priorities if using PER
        if use_per and indices is not None:
            td_errors_numpy = td_errors.detach().cpu().numpy().flatten()
            replay_buffer.update_priorities(indices, td_errors_numpy)

        metrics = {
            'critic_loss': critic_loss.item(),
            'td_error_mean': td_errors.mean().item(),
            'td_error_std': td_errors.std().item(),
            'importance_weight_mean': weights.mean().item(),
            'importance_weight_max': weights.max().item()
        }

        # Delayed policy updates
        if self.total_iterations % self.policy_delay == 0:
            # Compute actor loss
            actor_loss = -self.critic.q1_forward(state, self.actor(state)).mean()

            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            # Update target networks 软更新三个网络
            self.soft_update()

            metrics['actor_loss'] = actor_loss.item()

        return metrics

    def calculate_td_error(self, state: np.ndarray, action: np.ndarray,
                           reward: float, next_state: np.ndarray, done: bool) -> float:
        '''Calculate TD error for a single transition使用该方法来计算单一新经验的TD error，作为其初始优先级，为per做准备'''
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            reward_tensor = torch.FloatTensor([reward]).unsqueeze(0).to(self.device)
            done_tensor = torch.FloatTensor([done]).unsqueeze(0).to(self.device)

            # 计算当前Q值
            current_q1, current_q2 = self.critic(state_tensor, action_tensor)
            current_q = torch.min(current_q1, current_q2)

            # 计算目标Q值
            next_action = self.actor_target(next_state_tensor)
            target_q1, target_q2 = self.critic_target(next_state_tensor, next_action)
            target_q = torch.min(target_q1, target_q2)
            y = reward_tensor + (1 - done_tensor) * self.gamma * target_q

            # TD误差
            td_error = torch.abs(current_q - y).item()

        return td_error

    def soft_update(self):
        '''Soft update of target networks目标策略网络和两个目标价值网络'''
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path: str):
        '''Save model'''
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_iterations': self.total_iterations
        }, path)

    def load(self, path: str):
        '''Load model'''
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.total_iterations = checkpoint['total_iterations']