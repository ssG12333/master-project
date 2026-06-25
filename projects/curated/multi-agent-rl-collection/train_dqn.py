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
    MAP_COLS, DEVICE, AGENT_RADIUS, CELL_SIZE
# 从 dqn_network.py 导入 DQN 网络
from dqn_network import DQNNetwork
# 从 utils.py 导入辅助函数
from utils import get_gpu_memory_usage, save_multi_agent_path_plot, save_training_plots

# DDQN 训练结果和模型保存目录
DDQN_DIR = "DDQN_Results1"
if not os.path.exists(DDQN_DIR):
    os.makedirs(DDQN_DIR)
    print(f"创建目录: {DDQN_DIR}")


# 经验回放缓冲区 (与 MADDPG 相同)
class ReplayBuffer:
    def __init__(self, capacity):
        """
        经验回放缓冲区。
        capacity: 缓冲区最大容量。
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, local_state, action_idx, reward, next_local_state, done):
        """
        向缓冲区添加一条经验。
        注意: 对于独立 DQN，我们只存储局部状态和离散动作索引。
        """
        self.buffer.append((local_state, action_idx, reward, next_local_state, done))

    def sample(self, batch_size):
        """
        从缓冲区随机采样一个批次的经验。
        返回:
            tuple: 包含批次数据的元组 (local_states, action_indices, rewards, next_local_states, dones)。
        """
        batch = random.sample(self.buffer, batch_size)
        local_states, action_indices, rewards, next_local_states, dones = zip(*batch)
        return (np.array(local_states), np.array(action_indices), np.array(rewards),
                np.array(next_local_states), np.array(dones))

    def __len__(self):
        """
        返回缓冲区当前大小。
        """
        return len(self.buffer)


# DDQN 训练器
class DDQNTrainer:
    def __init__(self, env, device, num_agents):
        """
        DDQN 训练器。
        env: 环境实例。
        device: PyTorch 设备 (CPU/CUDA)。
        num_agents: 代理数量。
        """
        self.env = env
        self.device = device
        self.num_agents = num_agents

        # 计算局部状态维度
        other_agents_info_dim = 3 * (self.num_agents - 1) if self.num_agents > 1 else 0
        self.local_state_dim = 2 + 2 + 1 + self.env.num_rays + other_agents_info_dim

        # 定义离散动作空间
        # 9 个离散动作: 停止，以及 8 个方向 (0, 45, 90, ..., 315 度)
        self.num_discrete_actions = 9
        self.discrete_actions = self._define_discrete_actions(env.max_speed)
        self.action_dim = 2  # 环境的实际动作维度 (vx, vy)

        # 当前 Q 网络和目标 Q 网络列表 (每个代理一个)
        self.q_networks = [DQNNetwork(self.local_state_dim, self.num_discrete_actions).to(device) for _ in
                           range(num_agents)]
        self.target_q_networks = [DQNNetwork(self.local_state_dim, self.num_discrete_actions).to(device) for _ in
                                  range(num_agents)]

        # 优化器列表
        self.optimizers = [optim.Adam(q_net.parameters(), lr=0.0001) for q_net in self.q_networks]

        # 初始化目标网络与当前网络相同
        self.update_target_networks(tau=1.0)

        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(capacity=int(1e6))  # 缓冲区容量
        self.batch_size = 2048  # 训练批处理大小
        self.gamma = 0.99  # 折扣因子
        self.tau = 0.005  # 目标网络软更新参数 (用于软更新，但 DDQN 通常是硬更新)
        self.update_target_interval = 200  # 目标网络更新频率 (每隔多少步硬更新一次)

        self.epsilon = 1.0  # 探索率
        self.epsilon_decay = 0.995  # 探索率衰减
        self.epsilon_min = 0.01  # 最小探索率

        self.best_avg_reward = -float('inf')  # 用于保存最佳模型

    def _define_discrete_actions(self, max_speed):
        """
        定义离散动作空间。
        返回一个列表，每个元素是 (vx, vy) 向量。
        """
        actions = []
        # 动作 0: 停止
        actions.append(np.array([0.0, 0.0], dtype=np.float32))

        # 动作 1-8: 8 个方向的移动
        for i in range(8):
            angle = np.deg2rad(i * 45)  # 0, 45, 90, ..., 315 度
            vx = max_speed * np.cos(angle)
            vy = max_speed * np.sin(angle)
            actions.append(np.array([vx, vy], dtype=np.float32))
        return actions

    def update_target_networks(self, tau=None):
        """
        更新目标 Q 网络参数。
        DDQN 通常是硬更新 (tau=1.0)，但也可以选择软更新。
        """
        if tau is None:
            tau = self.tau  # 默认使用软更新参数

        # DDQN 通常是硬更新，这里可以根据 update_target_interval 进行硬更新
        # 如果是软更新，则使用下面的循环
        for i in range(self.num_agents):
            for target_param, param in zip(self.target_q_networks[i].parameters(), self.q_networks[i].parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def select_action(self, local_state, agent_idx, explore=True):
        """
        根据当前局部状态选择一个离散动作索引。
        local_state: 单个代理的局部观测状态 (NumPy 数组)。
        agent_idx: 代理索引。
        explore: 是否使用 epsilon-greedy 策略。
        返回:
            action_idx (int): 选择的离散动作索引。
            continuous_action (np.array): 对应的连续动作 (vx, vy)。
        """
        if explore and random.random() < self.epsilon:
            # 随机选择一个离散动作索引
            action_idx = random.randrange(self.num_discrete_actions)
        else:
            # 从 Q 网络中选择 Q 值最高的动作
            self.q_networks[agent_idx].eval()  # 设置为评估模式
            with torch.no_grad():
                local_state_tensor = torch.FloatTensor(local_state).unsqueeze(0).to(self.device)
                q_values = self.q_networks[agent_idx](local_state_tensor)
                action_idx = q_values.argmax(dim=1).item()
            self.q_networks[agent_idx].train()  # 恢复训练模式

        # 将离散动作索引映射到连续动作 (vx, vy)
        continuous_action = self.discrete_actions[action_idx]
        return action_idx, continuous_action

    def train_step(self, step_count):
        """
        执行一个 DDQN 训练步骤。
        从回放缓冲区采样，更新 Q 网络。
        step_count: 当前训练的总步数 (用于目标网络更新频率)。
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0  # 如果缓冲区数据不足，不进行训练

        # 从回放缓冲区采样
        local_states, action_indices, rewards, next_local_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        # 转换为 PyTorch 张量
        local_states_tensor = torch.FloatTensor(local_states).to(self.device)
        action_indices_tensor = torch.LongTensor(action_indices).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_local_states_tensor = torch.FloatTensor(next_local_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        q_loss_total = 0.0

        # 对每个代理进行训练
        for i in range(self.num_agents):
            self.optimizers[i].zero_grad()

            # 从当前 Q 网络获取 Q 值
            current_q_values = self.q_networks[i](
                local_states_tensor[:, i * self.local_state_dim:(i + 1) * self.local_state_dim])
            # 收集被选择动作的 Q 值
            q_value = current_q_values.gather(1, action_indices_tensor[:, i].unsqueeze(1)).squeeze(1)

            # 计算目标 Q 值 (DDQN 核心)
            with torch.no_grad():
                # 使用当前 Q 网络选择下一个状态的最佳动作
                next_q_values_current = self.q_networks[i](
                    next_local_states_tensor[:, i * self.local_state_dim:(i + 1) * self.local_state_dim])
                next_best_action_indices = next_q_values_current.argmax(dim=1).unsqueeze(1)

                # 使用目标 Q 网络评估下一个状态中最佳动作的 Q 值
                next_q_values_target = self.target_q_networks[i](
                    next_local_states_tensor[:, i * self.local_state_dim:(i + 1) * self.local_state_dim])
                next_q_value = next_q_values_target.gather(1, next_best_action_indices).squeeze(1)

                # 计算目标值
                target_q_value = rewards_tensor[:, i] + self.gamma * next_q_value * (1 - dones_tensor[:, i])

            # Q 损失 (MSE)
            q_loss = F.mse_loss(q_value, target_q_value)
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_networks[i].parameters(), 0.5)  # 梯度裁剪
            self.optimizers[i].step()
            q_loss_total += q_loss.item()

        # 周期性硬更新目标网络
        if step_count % self.update_target_interval == 0:
            self.update_target_networks(tau=1.0)  # 硬更新

        # 衰减 epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return q_loss_total / self.num_agents

    def save_best_weights(self):
        """保存当前模型的最佳权重。"""
        for i in range(self.num_agents):
            torch.save(self.q_networks[i].state_dict(), os.path.join(DDQN_DIR, f"best_q_network_{i}.pth"))
        print(f"DDQN 模型权重已保存到: {DDQN_DIR}")

    def load_best_weights(self):
        """加载已保存的最佳模型权重。"""
        for i in range(self.num_agents):
            model_path = os.path.join(DDQN_DIR, f"best_q_network_{i}.pth")
            if os.path.exists(model_path):
                self.q_networks[i].load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
                # 加载时也更新目标网络，确保一致性
                self.target_q_networks[i].load_state_dict(
                    torch.load(model_path, map_location=self.device, weights_only=False))
            else:
                print(f"警告: 未找到代理 {i} 的 DDQN 模型文件。")
        print(f"已从 {DDQN_DIR} 加载 DDQN 模型权重。")


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
    # 实例化 DDQN 训练器
    trainer = DDQNTrainer(env, DEVICE, num_agents)

    # 训练过程中的统计数据
    episode_rewards = []  # 总奖励
    episode_q_losses = []  # Q 损失
    episode_steps_sum_all_agvs = []  # 所有 AGV 的总步数
    avg_steps_reached_agvs = []  # 到达目标的 AGV 的平均步数
    episode_reach_rates = []  # 每个 episode 的到达率

    # 每个代理的奖励和步数历史
    episode_rewards_per_agent = [[] for _ in range(num_agents)]
    episode_steps_per_agent = [[] for _ in range(num_agents)]

    best_avg_reward = -float('inf')  # 记录最佳平均奖励
    patience_counter = 0  # 早停计数器
    total_steps_trained = 0  # 记录总训练步数，用于目标网络更新

    print("开始 DDQN 训练...")
    with tqdm(range(max_episodes), desc="训练进度") as pbar:
        for episode in pbar:
            try:
                # 重置环境，获取初始状态
                local_states = env.reset()
                global_state = env._get_global_state()  # DDQN 不直接使用 global_state，但为了兼容保留

                per_agent_episode_reward = [0.0] * num_agents
                per_agent_episode_steps = [0] * num_agents
                per_agent_reached_goal = [False] * num_agents

                current_episode_total_reward = 0
                steps_taken_overall_sum = 0

                all_agents_done = False

                # Episode 循环
                while not all_agents_done:
                    actions_discrete_indices = []  # 存储离散动作索引
                    actions_continuous = []  # 存储连续动作 (vx, vy)

                    for i, state in enumerate(local_states):
                        # 如果代理已到达目标或达到最大步数，则不再行动
                        if not per_agent_reached_goal[i] and per_agent_episode_steps[i] < max_steps_per_episode:
                            action_idx_i, continuous_action_i = trainer.select_action(state, i, explore=True)
                            actions_discrete_indices.append(action_idx_i)
                            actions_continuous.append(continuous_action_i)
                        else:
                            actions_discrete_indices.append(0)  # 停止动作的索引
                            actions_continuous.append(np.zeros(trainer.action_dim, dtype=np.float32))  # 停止

                    # 环境步进
                    next_local_states, next_global_state, rewards, all_agents_done, infos = env.step(actions_continuous)

                    # 存储经验到回放缓冲区
                    # 存储所有代理的局部状态和选择的离散动作索引
                    flat_local_states = np.concatenate(local_states).astype(np.float32)
                    flat_next_local_states = np.concatenate(next_local_states).astype(np.float32)
                    flat_actions_discrete_indices = np.array(actions_discrete_indices).astype(np.int64)  # 动作索引是整数
                    flat_rewards = np.array(rewards).astype(np.float32)
                    flat_dones = np.array(
                        [info['reached_goal'] or info['steps'] >= max_steps_per_episode for info in infos]).astype(
                        np.float32)

                    trainer.replay_buffer.push(flat_local_states, flat_actions_discrete_indices, flat_rewards,
                                               flat_next_local_states, flat_dones)

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
                    q_loss = 0.0
                    if len(trainer.replay_buffer) >= train_start_steps and len(
                            trainer.replay_buffer) >= trainer.batch_size:
                        q_loss = trainer.train_step(total_steps_trained)
                        total_steps_trained += 1  # 只有在进行训练步骤时才增加总步数

                # 记录训练统计数据
                episode_rewards.append(current_episode_total_reward)
                episode_q_losses.append(q_loss)
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
                    'Q损失': f'{q_loss:.4f}',
                    'Epsilon': f'{trainer.epsilon:.4f}'  # 显示 epsilon
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
                                               os.path.join(DDQN_DIR, f"path_episode_{episode + 1}.png"), AGENT_RADIUS)
                    print(f"已保存 Episode {episode + 1} 路径图: path_episode_{episode + 1}.png")

                # 定期保存训练曲线图和数据
                if (episode + 1) % save_interval_curve_plot == 0:
                    # 对于 DDQN，只有 Q 损失，所以 policy_loss 和 value_loss 传入 None
                    save_training_plots(list(range(len(episode_rewards))), None, episode_q_losses,
                                        episode_rewards, episode_steps_sum_all_agvs,
                                        episode_rewards_per_agent, episode_steps_per_agent,
                                        avg_steps_reached_agvs, episode_reach_rates,
                                        os.path.join(DDQN_DIR, f"training_curves_episode_{episode + 1}.png"))
                    print(f"已保存 Episode {episode + 1} 训练曲线图: training_curves_episode_{episode + 1}.png")

                    # 保存当前训练结果到 CSV
                    results_df = pd.DataFrame({
                        'episode': list(range(len(episode_rewards))),
                        'total_reward': episode_rewards,
                        'q_loss': episode_q_losses,
                        'total_steps_sum_all_agvs': episode_steps_sum_all_agvs,
                        'avg_reward_per_agv_per_episode': np.mean(np.array(episode_rewards_per_agent), axis=0),
                        'avg_steps_per_agv_per_episode': np.mean(np.array(episode_steps_per_agent), axis=0),
                        'avg_steps_reached_agvs': avg_steps_reached_agvs,
                        'episode_reach_rates': episode_reach_rates
                    })
                    results_df.to_csv(os.path.join(DDQN_DIR, f"training_results_episode_{episode + 1}.csv"),
                                      index=False)
                    print(f"已保存 Episode {episode + 1} 训练结果数据: training_results_episode_{episode + 1}.csv")

                # 早停逻辑
                if avg_reward_last_100 > best_avg_reward:
                    best_avg_reward = avg_reward_last_100
                    trainer.save_best_weights()
                    patience_counter = 0
                    print(f"\n检测到最佳平均奖励，正在保存 DDQN 模型权重。")
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"\n在 {episode + 1} 个 episode 后达到耐心上限，提前停止 DDQN 训练。")
                    break

            except Exception as e:
                print(f"\nDDQN 训练过程中发生错误: {e}")
                traceback.print_exc()
                break

    print("DDQN 训练结束。")

    # 训练结束后保存最终训练曲线图和结果数据
    save_training_plots(list(range(len(episode_rewards))), None, episode_q_losses,
                        episode_rewards, episode_steps_sum_all_agvs,
                        episode_rewards_per_agent, episode_steps_per_agent,
                        avg_steps_reached_agvs, episode_reach_rates,
                        os.path.join(DDQN_DIR, "final_training_curves.png"))
    print(f"最终 DDQN 训练曲线图已保存到: {os.path.join(DDQN_DIR, 'final_training_curves.png')}")

    # 将训练结果保存为 CSV (最终保存，确保完整性)
    results_df = pd.DataFrame({
        'episode': list(range(len(episode_rewards))),
        'total_reward': episode_rewards,
        'q_loss': episode_q_losses,
        'total_steps_sum_all_agvs': episode_steps_sum_all_agvs,
        'avg_reward_per_agv_per_episode': np.mean(np.array(episode_rewards_per_agent), axis=0),
        'avg_steps_per_agv_per_episode': np.mean(np.array(episode_steps_per_agent), axis=0),
        'avg_steps_reached_agvs': avg_steps_reached_agvs,
        'episode_reach_rates': episode_reach_rates
    })
    results_df.to_csv(os.path.join(DDQN_DIR, "training_results.csv"), index=False)
    print(f"DDQN 训练结果数据已保存到: {os.path.join(DDQN_DIR, 'training_results.csv')}")

    print("\n开始最终 DDQN 测试...")
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
        actions_continuous = []
        for i, state in enumerate(local_states):
            _, continuous_action = trainer.select_action(state, i, explore=False)  # 测试时不探索
            actions_continuous.append(continuous_action)

        next_local_states, next_global_state, rewards, all_test_dones, infos = env.step(actions_continuous)

        for i in range(num_agents):
            test_paths[i].append(env.agents[i]['pos'].copy())
            test_total_rewards[i] += rewards[i]
            test_steps[i] = infos[i]['steps']
            test_reached_goals[i] = infos[i]['reached_goal']
        local_states = next_local_states
        global_state = next_global_state

    print(f"--- 最终 DDQN 测试单个 AGV 性能 ---")
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
                               os.path.join(DDQN_DIR, "final_test_paths.png"), AGENT_RADIUS)
    print(f"最终 DDQN 测试路径图已保存到: {os.path.join(DDQN_DIR, 'final_test_paths.png')}")
    print("DDQN 测试结束。")


if __name__ == "__main__":
    main()

