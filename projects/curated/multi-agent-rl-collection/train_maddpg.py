import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import os
import random
import traceback
from collections import deque  # 用于经验回放缓冲区
from tqdm import tqdm

# 从 environment.py 导入环境类和全局常量
from environment import MultiAgentPathPlanningEnv, OBSTACLES, PURE_WALKABLE_COORDS, GOAL_COORDS, MAP_BOUNDS, MAP_ROWS, \
    MAP_COLS, DEVICE, AGENT_RADIUS
# 从 maddpg_network.py 导入 MADDPG 网络
from maddpg_network import MADDPGActor, MADDPGCritic
# 从 utils.py 导入辅助函数
from utils import get_gpu_memory_usage, save_multi_agent_path_plot, save_training_plots

# MADDPG 训练结果和模型保存目录
MADDPG_DIR = "MADDPG_Results1"
if not os.path.exists(MADDPG_DIR):
    os.makedirs(MADDPG_DIR)
    print(f"创建目录: {MADDPG_DIR}")


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        """
        经验回放缓冲区。
        capacity: 缓冲区最大容量。
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, local_state, global_state, action, reward, next_local_state, next_global_state, done):
        """
        向缓冲区添加一条经验。
        """
        self.buffer.append((local_state, global_state, action, reward, next_local_state, next_global_state, done))

    def sample(self, batch_size):
        """
        从缓冲区随机采样一个批次的经验。
        返回:
            tuple: 包含批次数据的元组 (local_states, global_states, actions, rewards, next_local_states, next_global_states, dones)。
        """
        batch = random.sample(self.buffer, batch_size)
        local_states, global_states, actions, rewards, next_local_states, next_global_states, dones = zip(*batch)
        return (np.array(local_states), np.array(global_states), np.array(actions),
                np.array(rewards), np.array(next_local_states), np.array(next_global_states), np.array(dones))

    def __len__(self):
        """
        返回缓冲区当前大小。
        """
        return len(self.buffer)


# Ornstein-Uhlenbeck 噪声 (用于探索)
class OUNoise:
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        """
        Ornstein-Uhlenbeck 噪声生成器。
        action_dimension: 动作维度。
        mu: 均值。
        theta: 均值回归率。
        sigma: 噪声强度。
        """
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        """
        重置噪声状态。
        """
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        """
        生成 Ornstein-Uhlenbeck 噪声。
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


# MADDPG 训练器
class MADDPGTrainer:
    def __init__(self, env, device, num_agents):
        """
        MADDPG 训练器。
        env: 环境实例。
        device: PyTorch 设备 (CPU/CUDA)。
        num_agents: 代理数量。
        """
        self.env = env
        self.device = device
        self.num_agents = num_agents

        # 计算局部状态维度和全局状态维度
        other_agents_info_dim = 3 * (self.num_agents - 1) if self.num_agents > 1 else 0
        self.local_state_dim = 2 + 2 + 1 + self.env.num_rays + other_agents_info_dim
        self.action_dim = 2  # 动作维度 (vx, vy)
        self.global_state_dim = self.local_state_dim * self.num_agents
        self.total_action_dim = self.action_dim * self.num_agents

        # Actor 和 Critic 网络列表
        self.actors = [MADDPGActor(self.local_state_dim, self.action_dim).to(device) for _ in range(num_agents)]
        self.critics = [MADDPGCritic(self.global_state_dim, self.total_action_dim).to(device) for _ in
                        range(num_agents)]

        # 目标 Actor 和 Critic 网络列表
        self.target_actors = [MADDPGActor(self.local_state_dim, self.action_dim).to(device) for _ in range(num_agents)]
        self.target_critics = [MADDPGCritic(self.global_state_dim, self.total_action_dim).to(device) for _ in
                               range(num_agents)]

        # 优化器列表
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=0.0001) for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=0.0001) for critic in self.critics]

        # 初始化目标网络与当前网络相同
        self.update_target_networks(tau=1.0)

        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(capacity=int(1e6))  # 缓冲区容量
        self.batch_size = 2048  # 训练批处理大小
        self.gamma = 0.99  # 折扣因子
        self.tau = 0.005  # 目标网络软更新参数
        self.noise = [OUNoise(self.action_dim) for _ in range(num_agents)]  # 每个代理一个噪声生成器

        self.best_avg_reward = -float('inf')  # 用于保存最佳模型

    def update_target_networks(self, tau=None):
        """
        更新目标网络参数。
        tau: 软更新参数 (0 < tau <= 1)。如果 tau=1，则硬更新。
        """
        if tau is None:
            tau = self.tau
        for i in range(self.num_agents):
            for target_param, param in zip(self.target_actors[i].parameters(), self.actors[i].parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            for target_param, param in zip(self.target_critics[i].parameters(), self.critics[i].parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def select_action(self, local_state, agent_idx, explore=True):
        """
        根据当前局部状态选择一个动作。
        local_state: 单个代理的局部观测状态 (NumPy 数组)。
        agent_idx: 代理索引。
        explore: 是否添加探索噪声。
        返回:
            action (np.array): 选择的动作。
        """
        self.actors[agent_idx].eval()  # 设置为评估模式
        with torch.no_grad():
            local_state_tensor = torch.FloatTensor(local_state).unsqueeze(0).to(self.device)
            action = self.actors[agent_idx](local_state_tensor).cpu().numpy().flatten()
        self.actors[agent_idx].train()  # 恢复训练模式

        if explore:
            action += self.noise[agent_idx].noise()  # 添加噪声

        # 裁剪动作到 [-1, 1] 范围
        action = np.clip(action, -1.0, 1.0)
        return action

    def train_step(self):
        """
        执行一个 MADDPG 训练步骤。
        从回放缓冲区采样，更新 Actor 和 Critic 网络。
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0  # 如果缓冲区数据不足，不进行训练

        # 从回放缓冲区采样
        local_states, global_states, actions, rewards, next_local_states, next_global_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        # 转换为 PyTorch 张量
        local_states_tensor = torch.FloatTensor(local_states).to(self.device)
        global_states_tensor = torch.FloatTensor(global_states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        # 修正: rewards_tensor 和 dones_tensor 不再需要 unsqueeze(1)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)  # 形状: [batch_size, num_agents]
        next_local_states_tensor = torch.FloatTensor(next_local_states).to(self.device)
        next_global_states_tensor = torch.FloatTensor(next_global_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)  # 形状: [batch_size, num_agents]

        critic_loss_total = 0.0
        actor_loss_total = 0.0

        # 对每个代理进行训练
        for i in range(self.num_agents):
            # --- 更新 Critic 网络 ---
            self.critic_optimizers[i].zero_grad()

            # 计算目标 Q 值
            with torch.no_grad():
                # 使用目标 Actor 网络生成下一个动作
                next_actions = [self.target_actors[j](
                    next_local_states_tensor[:, j * self.local_state_dim:(j + 1) * self.local_state_dim]) for j in
                                range(self.num_agents)]
                next_all_actions = torch.cat(next_actions, dim=1)  # 拼接所有代理的下一个动作

                # 使用目标 Critic 网络评估下一个状态-动作对的 Q 值
                target_q_next = self.target_critics[i](next_global_states_tensor, next_all_actions)
                # 修正: rewards_tensor[:, i] 现在是 (batch_size,)，unsqueeze(1) 变为 (batch_size, 1)
                target_q_value = rewards_tensor[:, i].unsqueeze(1) + self.gamma * target_q_next * (
                            1 - dones_tensor[:, i].unsqueeze(1))

            # 使用当前 Critic 网络评估当前状态-动作对的 Q 值
            current_q_value = self.critics[i](global_states_tensor, actions_tensor)

            # Critic 损失 (MSE)
            critic_loss = F.mse_loss(current_q_value, target_q_value)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), 0.5)  # 梯度裁剪
            self.critic_optimizers[i].step()
            critic_loss_total += critic_loss.item()

            # --- 更新 Actor 网络 ---
            self.actor_optimizers[i].zero_grad()

            # 使用当前 Actor 网络生成当前动作
            # 注意: Actor 损失需要 Critic 的梯度，所以这里不能用 torch.no_grad()
            current_actor_actions = [self.actors[j](
                local_states_tensor[:, j * self.local_state_dim:(j + 1) * self.local_state_dim]) if j == i
                                     else self.actors[j](
                local_states_tensor[:, j * self.local_state_dim:(j + 1) * self.local_state_dim]).detach()
                                     for j in range(self.num_agents)]
            current_all_actions = torch.cat(current_actor_actions, dim=1)

            # Actor 损失 (最大化 Q 值)
            actor_loss = -self.critics[i](global_states_tensor, current_all_actions).mean()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)  # 梯度裁剪
            self.actor_optimizers[i].step()
            actor_loss_total += actor_loss.item()

        # 软更新目标网络
        self.update_target_networks()

        return actor_loss_total / self.num_agents, critic_loss_total / self.num_agents

    def save_best_weights(self):
        """保存当前模型的最佳权重。"""
        for i in range(self.num_agents):
            torch.save(self.actors[i].state_dict(), os.path.join(MADDPG_DIR, f"best_actor_{i}.pth"))
            torch.save(self.critics[i].state_dict(), os.path.join(MADDPG_DIR, f"best_critic_{i}.pth"))
        print(f"MADDPG 模型权重已保存到: {MADDPG_DIR}")

    def load_best_weights(self):
        """加载已保存的最佳模型权重。"""
        for i in range(self.num_agents):
            actor_path = os.path.join(MADDPG_DIR, f"best_actor_{i}.pth")
            critic_path = os.path.join(MADDPG_DIR, f"best_critic_{i}.pth")
            if os.path.exists(actor_path) and os.path.exists(critic_path):
                self.actors[i].load_state_dict(torch.load(actor_path, map_location=self.device, weights_only=False))
                self.critics[i].load_state_dict(torch.load(critic_path, map_location=self.device, weights_only=False))
            else:
                print(f"警告: 未找到代理 {i} 的 MADDPG 模型文件。")
        print(f"已从 {MADDPG_DIR} 加载 MADDPG 模型权重。")


# 主训练逻辑
def main():
    num_agents = 15  # 代理数量
    max_episodes = 6000  # 最大训练 episode 数量
    save_interval_path_plot = 50  # 每隔多少个 episode 保存一次路径图
    save_interval_curve_plot = 50  # 每隔多少个 episode 保存一次训练曲线图和数据
    patience = 3000  # 早停耐心值
    max_steps_per_episode = 200  # 每个 episode 的最大步数
    train_start_steps = 10000  # 经验回放缓冲区达到此数量后开始训练

    # 实例化环境
    env = MultiAgentPathPlanningEnv(OBSTACLES, PURE_WALKABLE_COORDS, GOAL_COORDS, MAP_BOUNDS, MAP_ROWS, MAP_COLS,
                                    num_agents, DEVICE, max_steps_per_episode)
    # 实例化 MADDPG 训练器
    trainer = MADDPGTrainer(env, DEVICE, num_agents)

    # 训练过程中的统计数据
    episode_rewards = []  # 总奖励
    episode_actor_losses = []  # Actor 损失
    episode_critic_losses = []  # Critic 损失
    episode_steps_sum_all_agvs = []  # 所有 AGV 的总步数
    avg_steps_reached_agvs = []  # 到达目标的 AGV 的平均步数
    episode_reach_rates = []  # 每个 episode 的到达率

    # 每个代理的奖励和步数历史
    episode_rewards_per_agent = [[] for _ in range(num_agents)]
    episode_steps_per_agent = [[] for _ in range(num_agents)]

    best_avg_reward = -float('inf')  # 记录最佳平均奖励
    patience_counter = 0  # 早停计数器

    print("开始 MADDPG 训练...")
    with tqdm(range(max_episodes), desc="训练进度") as pbar:
        for episode in pbar:
            try:
                # 重置环境，获取初始状态
                local_states = env.reset()
                global_state = env._get_global_state()
                # 重置噪声
                for noise_gen in trainer.noise:
                    noise_gen.reset()

                per_agent_episode_reward = [0.0] * num_agents
                per_agent_episode_steps = [0] * num_agents
                per_agent_reached_goal = [False] * num_agents

                current_episode_total_reward = 0
                steps_taken_overall_sum = 0

                all_agents_done = False

                # Episode 循环
                while not all_agents_done:
                    actions = []
                    # 收集所有代理的局部状态，以便 Actor 选择动作
                    current_local_states_list = [local_states[i] for i in range(num_agents)]

                    for i, state in enumerate(current_local_states_list):
                        # 如果代理已到达目标或达到最大步数，则不再行动
                        if not per_agent_reached_goal[i] and per_agent_episode_steps[i] < max_steps_per_episode:
                            action_i = trainer.select_action(state, i, explore=True)
                            actions.append(action_i)
                        else:
                            actions.append(np.zeros(trainer.action_dim, dtype=np.float32))  # 不再移动

                    # 环境步进
                    next_local_states, next_global_state, rewards, all_agents_done, infos = env.step(actions)

                    # 存储经验到回放缓冲区
                    # MADDPG 经验是 (local_state, global_state, action, reward, next_local_state, next_global_state, done)
                    # 注意: 这里需要将所有代理的 local_state 和 next_local_state 作为一个整体存储
                    # global_state 已经是所有 local_state 的拼接

                    # 转换 local_states 和 next_local_states 为适合存储的格式
                    # 假设 local_states 是一个列表，每个元素是一个代理的 np.array
                    # 转换为一个 (num_agents * local_state_dim) 的 np.array
                    flat_local_states = np.concatenate(local_states).astype(np.float32)
                    flat_next_local_states = np.concatenate(next_local_states).astype(np.float32)
                    flat_actions = np.concatenate(actions).astype(np.float32)
                    flat_rewards = np.array(rewards).astype(np.float32)
                    flat_dones = np.array(
                        [info['reached_goal'] or info['steps'] >= max_steps_per_episode for info in infos]).astype(
                        np.float32)

                    trainer.replay_buffer.push(flat_local_states, global_state, flat_actions, flat_rewards,
                                               flat_next_local_states, next_global_state, flat_dones)

                    # 更新代理的统计信息
                    for i in range(num_agents):
                        per_agent_episode_reward[i] += rewards[i]
                        per_agent_episode_steps[i] = infos[i]['steps']
                        per_agent_reached_goal[i] = infos[i]['reached_goal']

                    current_episode_total_reward = sum(per_agent_episode_reward)
                    steps_taken_overall_sum = sum(per_agent_episode_steps)
                    local_states = next_local_states
                    global_state = next_global_state

                    # 训练网络 (当缓冲区足够大时)
                    actor_loss, critic_loss = 0.0, 0.0
                    if len(trainer.replay_buffer) >= train_start_steps and len(
                            trainer.replay_buffer) >= trainer.batch_size:
                        actor_loss, critic_loss = trainer.train_step()

                # 记录训练统计数据
                episode_rewards.append(current_episode_total_reward)
                episode_actor_losses.append(actor_loss)
                episode_critic_losses.append(critic_loss)
                episode_steps_sum_all_agvs.append(steps_taken_overall_sum)

                # 计算到达目标的 AGV 的平均步数和本轮到达率
                reached_steps_in_episode = []
                num_reached_agents_in_episode = 0
                for i in range(num_agents):
                    if per_agent_reached_goal[i]:
                        reached_steps_in_episode.append(per_agent_episode_steps[i])
                        num_reached_agents_in_episode += 1

                current_avg_steps_reached = np.mean(reached_steps_in_episode) if reached_steps_in_episode else 0
                current_reach_rate = num_reached_agents_in_episode / num_agents

                avg_steps_reached_agvs.append(current_avg_steps_reached)
                episode_reach_rates.append(current_reach_rate)

                # 更新每个代理的奖励和步数历史
                for i in range(num_agents):
                    episode_rewards_per_agent[i].append(per_agent_episode_reward[i])
                    episode_steps_per_agent[i].append(per_agent_episode_steps[i])

                # 计算过去 100 个 episode 的平均奖励，用于早停判断
                avg_reward_last_100 = np.mean(episode_rewards[-100:]) if len(
                    episode_rewards) >= 100 else current_episode_total_reward

                # 计算当前 episode 中每个 AGV 的平均奖励
                avg_reward_current_agv_episode = np.mean(per_agent_episode_reward)
                # 计算当前 episode 中每个 AGV 的平均步数
                avg_steps_current_agv_episode = np.mean(per_agent_episode_steps)

                # 获取 GPU 内存使用情况
                gpu_allocated_mb, gpu_cached_mb = get_gpu_memory_usage()

                # 更新进度条显示
                postfix_dict = {
                    '总奖励': f'{current_episode_total_reward:.2f}',
                    '平均奖励 (近100)': f'{avg_reward_last_100:.2f}',
                    'AGV平均奖励 (当前)': f'{avg_reward_current_agv_episode:.2f}',
                    '总步数 (当前)': steps_taken_overall_sum,
                    'AGV平均步数 (当前)': f'{avg_steps_current_agv_episode:.1f}',
                    '到达AGV平均步数': f'{current_avg_steps_reached:.1f}',
                    '到达率': f'{current_reach_rate:.2f}',
                    'Actor损失': f'{actor_loss:.4f}',
                    'Critic损失': f'{critic_loss:.4f}'
                }
                if DEVICE.type == 'cuda':
                    postfix_dict['GPU 内存'] = f'{gpu_allocated_mb:.1f}MB'
                pbar.set_postfix(postfix_dict)

                # 打印单个 AGV 性能详情
                print(f"--- Episode {episode + 1} 单个 AGV 性能 ---")
                for ag_idx, reward_info in enumerate(per_agent_episode_reward):
                    print(
                        f" AGV {ag_idx}: 奖励={reward_info:.2f}, 步数={per_agent_episode_steps[ag_idx]}, 到达目标={per_agent_reached_goal[ag_idx]}")
                print(f"------------------------------------")

                # 定期保存路径图
                if (episode + 1) % save_interval_path_plot == 0:
                    current_episode_paths = []
                    for agent_data in env.agents:
                        current_episode_paths.append({
                            'path': agent_data['path_history'],
                            'goal': agent_data['goal']
                        })
                    save_multi_agent_path_plot(OBSTACLES, GOAL_COORDS, MAP_BOUNDS, current_episode_paths,
                                               os.path.join(MADDPG_DIR, f"path_episode_{episode + 1}.png"),
                                               AGENT_RADIUS)
                    print(f"已保存 Episode {episode + 1} 路径图: path_episode_{episode + 1}.png")

                # 定期保存训练曲线图和数据
                if (episode + 1) % save_interval_curve_plot == 0:
                    save_training_plots(list(range(len(episode_rewards))), episode_actor_losses, episode_critic_losses,
                                        episode_rewards, episode_steps_sum_all_agvs,
                                        episode_rewards_per_agent, episode_steps_per_agent,
                                        avg_steps_reached_agvs, episode_reach_rates,
                                        os.path.join(MADDPG_DIR, f"training_curves_episode_{episode + 1}.png"))
                    print(f"已保存 Episode {episode + 1} 训练曲线图: training_curves_episode_{episode + 1}.png")

                    # 保存当前训练结果到 CSV
                    results_df = pd.DataFrame({
                        'episode': list(range(len(episode_rewards))),
                        'total_reward': episode_rewards,
                        'actor_loss': episode_actor_losses,
                        'critic_loss': episode_critic_losses,
                        'total_steps_sum_all_agvs': episode_steps_sum_all_agvs,
                        'avg_reward_per_agv_per_episode': np.mean(np.array(episode_rewards_per_agent), axis=0),
                        'avg_steps_per_agv_per_episode': np.mean(np.array(episode_steps_per_agent), axis=0),
                        'avg_steps_reached_agvs': avg_steps_reached_agvs,
                        'episode_reach_rates': episode_reach_rates
                    })
                    results_df.to_csv(os.path.join(MADDPG_DIR, f"training_results_episode_{episode + 1}.csv"),
                                      index=False)
                    print(f"已保存 Episode {episode + 1} 训练结果数据: training_results_episode_{episode + 1}.csv")

                # 早停逻辑
                if avg_reward_last_100 > best_avg_reward:
                    best_avg_reward = avg_reward_last_100
                    trainer.save_best_weights()
                    patience_counter = 0
                    print(f"\n检测到最佳平均奖励，正在保存 MADDPG 模型权重。")
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"\n在 {episode + 1} 个 episode 后达到耐心上限，提前停止 MADDPG 训练。")
                    break

            except Exception as e:
                print(f"\nMADDPG 训练过程中发生错误: {e}")
                traceback.print_exc()
                break

    print("MADDPG 训练结束。")

    # 训练结束后保存最终训练曲线图和结果数据
    save_training_plots(list(range(len(episode_rewards))), episode_actor_losses, episode_critic_losses,
                        episode_rewards, episode_steps_sum_all_agvs,
                        episode_rewards_per_agent, episode_steps_per_agent,
                        avg_steps_reached_agvs, episode_reach_rates,
                        os.path.join(MADDPG_DIR, "final_training_curves.png"))
    print(f"最终 MADDPG 训练曲线图已保存到: {os.path.join(MADDPG_DIR, 'final_training_curves.png')}")

    # 将训练结果保存为 CSV (最终保存，确保完整性)
    results_df = pd.DataFrame({
        'episode': list(range(len(episode_rewards))),
        'total_reward': episode_rewards,
        'actor_loss': episode_actor_losses,
        'critic_loss': episode_critic_losses,
        'total_steps_sum_all_agvs': episode_steps_sum_all_agvs,
        'avg_reward_per_agv_per_episode': np.mean(np.array(episode_rewards_per_agent), axis=0),
        'avg_steps_per_agv_per_episode': np.mean(np.array(episode_steps_per_agent), axis=0),
        'avg_steps_reached_agvs': avg_steps_reached_agvs,
        'episode_reach_rates': episode_reach_rates
    })
    results_df.to_csv(os.path.join(MADDPG_DIR, "training_results.csv"), index=False)
    print(f"MADDPG 训练结果数据已保存到: {os.path.join(MADDPG_DIR, 'training_results.csv')}")

    print("\n开始最终 MADDPG 测试...")
    # MADDPG 测试时，不需要探索噪声
    # 加载最佳模型权重
    trainer.load_best_weights()
    local_states = env.reset()
    global_state = env._get_global_state()
    test_paths = [[] for _ in range(num_agents)]
    test_total_rewards = [0] * num_agents
    test_steps = [0] * num_agents
    test_reached_goals = [False] * num_agents

    for i, agent_data in enumerate(env.agents):
        test_paths[i].append(agent_data['pos'].copy())

    all_test_dones = False

    while not all_test_dones:
        actions = []
        for i, state in enumerate(local_states):
            action = trainer.select_action(state, i, explore=False)  # 测试时不探索
            actions.append(action)

        next_local_states, next_global_state, rewards, all_test_dones, infos = env.step(actions)

        for i in range(num_agents):
            test_paths[i].append(env.agents[i]['pos'].copy())
            test_total_rewards[i] += rewards[i]
            test_steps[i] = infos[i]['steps']
            test_reached_goals[i] = infos[i]['reached_goal']
        local_states = next_local_states
        global_state = next_global_state

    print(f"--- 最终 MADDPG 测试单个 AGV 性能 ---")
    final_test_paths_data = []
    for i in range(num_agents):
        final_test_paths_data.append({
            'path': test_paths[i],
            'goal': env.agents[i]['goal']
        })
        print(
            f" AGV {i}: 奖励={test_total_rewards[i]:.2f}, 步数={test_steps[i]}, 到达目标={test_reached_goals[i]}")
    print(f"-----------------------------")

    save_multi_agent_path_plot(OBSTACLES, GOAL_COORDS, MAP_BOUNDS, final_test_paths_data,
                               os.path.join(MADDPG_DIR, "final_test_paths.png"), AGENT_RADIUS)
    print(f"最终 MADDPG 测试路径图已保存到: {os.path.join(MADDPG_DIR, 'final_test_paths.png')}")
    print("MADDPG 测试结束。")


if __name__ == "__main__":
    main()

