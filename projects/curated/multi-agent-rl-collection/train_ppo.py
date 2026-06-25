import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import pandas as pd
import os
import random
import traceback
from tqdm import tqdm

# 从 environment.py 导入环境类和全局常量
from environment import MultiAgentPathPlanningEnv, OBSTACLES, PURE_WALKABLE_COORDS, GOAL_COORDS, MAP_BOUNDS, MAP_ROWS, MAP_COLS, DEVICE, AGENT_RADIUS
# 从 ppo_network.py 导入 PPO 网络
from ppo_network import PPONetwork
# 从 utils.py 导入辅助函数
from utils import get_gpu_memory_usage, save_multi_agent_path_plot, save_training_plots

# PPO 训练结果和模型保存目录
PPO_DIR = "PPO_Results1"
if not os.path.exists(PPO_DIR):
    os.makedirs(PPO_DIR)
    print(f"创建目录: {PPO_DIR}")

# PPO 训练器 (去中心化 Actor, 中心化 Critic)
class PPOTrainer(nn.Module):
    def __init__(self, env, device):
        """
        PPO 训练器。
        env: 环境实例。
        device: PyTorch 设备 (CPU/CUDA)。
        """
        super(PPOTrainer, self).__init__()
        self.env = env
        self.device = device

        # 计算局部状态维度和全局状态维度
        # 局部状态维度: 当前位置(2) + 目标位置(2) + 距离目标(1) + 射线投射距离(num_rays) + 其他代理相对信息(3 * (num_agents - 1))
        # 确保 env.num_agents 至少为 1，否则 3 * (self.env.num_agents - 1) 会是负数
        other_agents_info_dim = 3 * (self.env.num_agents - 1) if self.env.num_agents > 1 else 0
        self.local_state_dim = 2 + 2 + 1 + self.env.num_rays + other_agents_info_dim
        self.global_state_dim = self.local_state_dim * self.env.num_agents
        self.action_dim = 2 # 动作维度 (vx, vy)

        # 实例化 PPO 网络
        self.model = PPONetwork(self.local_state_dim, self.global_state_dim, self.action_dim, self.env.num_agents).to(device)
        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0003)
        # PPO 算法参数
        self.gamma = 0.99 # 折扣因子
        self.lam = 0.95 # GAE 参数
        self.clip_param = 0.2 # PPO 裁剪参数
        self.value_loss_coef = 0.5 # 价值损失系数
        self.entropy_coef = 0.02 # 熵系数，鼓励探索
        self.max_grad_norm = 0.5 # 梯度裁剪最大范数
        self.ppo_epochs = 10 # PPO 训练迭代次数
        self.batch_size = 2048 # 训练批处理大小

        # 用于保存最佳模型权重
        self.best_weights = self.model.state_dict()

    def select_action(self, local_state):
        """
        根据当前局部状态从策略中选择一个动作 (仅 Actor)。
        local_state: 单个代理的局部观测状态 (NumPy 数组)。
        返回:
            action (np.array): 采样得到的动作。
            log_prob (torch.Tensor): 动作的对数概率。
        """
        # 将 NumPy 数组转换为 PyTorch 张量，并添加到设备上
        local_state_tensor = torch.FloatTensor(local_state).unsqueeze(0).to(self.device)
        # 获取策略分布
        policy = self.model.get_action_policy(local_state_tensor)
        # 从策略中采样动作
        action = policy.sample()
        # 计算采样动作的对数概率
        log_prob = policy.log_prob(action).sum(dim=-1) # 对动作维度求和
        # 返回 CPU 上的 NumPy 动作和对数概率张量
        return action.squeeze(0).cpu().numpy(), log_prob.squeeze(0)

    def compute_gae(self, rewards, values, next_value, dones):
        """
        计算广义优势估计 (GAE) 和回报。
        rewards: 序列中的奖励 (NumPy 数组)。
        values: 序列中相应状态的价值估计 (NumPy 数组)。
        next_value: 序列中最后一个状态之后状态的价值估计 (用于引导) (标量)。
        dones: 序列中每个时间步是否终止 (NumPy 数组)。
        返回:
            advantages (np.array): 优势估计。
            returns (np.array): GAE 回报。
        """
        advantages = []
        gae = 0
        # 从后向前迭代计算 GAE
        for i in reversed(range(len(rewards))):
            # 计算 TD 误差
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae) # 插入到列表开头
            next_value = values[i] # 更新下一个价值
        advantages = np.array(advantages, dtype=np.float32)
        returns = advantages + values # 计算回报
        return advantages, returns

    def train_step(self, trajectories_per_agent):
        """
        执行一个 PPO 训练步骤。
        trajectories_per_agent: 列表，每个元素是单个代理在一个 episode 中的轨迹。
        每个轨迹元素包含 (local_state, global_state, action, log_prob_old, reward, done, value)。
        返回:
            policy_loss_avg (float): 平均策略损失。
            value_loss_avg (float): 平均价值损失。
        """
        # 收集所有代理的所有轨迹数据
        all_local_states, all_global_states, all_actions, all_log_probs_old, all_rewards, all_dones, all_values = \
            [], [], [], [], [], [], []

        for agent_trajectory in trajectories_per_agent:
            for step_data in agent_trajectory:
                all_local_states.append(step_data[0])
                all_global_states.append(step_data[1])
                all_actions.append(step_data[2])
                all_log_probs_old.append(step_data[3])
                all_rewards.append(step_data[4])
                all_dones.append(step_data[5])
                all_values.append(step_data[6])

        # 将 NumPy 数组转换为 PyTorch 张量并移动到设备
        local_states_tensor = torch.FloatTensor(np.array(all_local_states)).to(self.device)
        global_states_tensor = torch.FloatTensor(np.array(all_global_states)).to(self.device)
        actions_tensor = torch.FloatTensor(np.array(all_actions)).to(self.device)
        log_probs_old_tensor = torch.FloatTensor(np.array(all_log_probs_old)).to(self.device)
        rewards_tensor = torch.FloatTensor(np.array(all_rewards)).to(self.device)
        dones_tensor = torch.FloatTensor(np.array(all_dones)).to(self.device)
        values_tensor = torch.FloatTensor(np.array(all_values)).to(self.device).flatten()

        # 计算所有轨迹的 GAE 和回报
        all_advantages = []
        all_returns = []

        # 遍历每个代理的轨迹来计算 GAE
        start_idx = 0
        for agent_trajectory in trajectories_per_agent:
            agent_rewards = np.array([t[4] for t in agent_trajectory])
            agent_values = np.array([t[6] for t in agent_trajectory]).flatten()
            agent_dones = np.array([t[5] for t in agent_trajectory])

            # 获取轨迹最后一个状态的下一个状态的价值估计（用于引导）
            # 如果轨迹结束是因为达到目标或最大步数，则下一个价值为 0
            last_global_state_of_trajectory = agent_trajectory[-1][1]
            with torch.no_grad(): # 不需要计算梯度
                last_value_estimate = self.model.get_value(
                    torch.FloatTensor(last_global_state_of_trajectory).unsqueeze(0).to(self.device))

            next_value_for_gae = last_value_estimate.item() * (1 - agent_dones[-1])

            adv, ret = self.compute_gae(agent_rewards, agent_values, next_value_for_gae, agent_dones)

            all_advantages.extend(adv)
            all_returns.extend(ret)
            start_idx += len(agent_trajectory)

        advantages_tensor = torch.FloatTensor(np.array(all_advantages)).to(self.device)
        returns_tensor = torch.FloatTensor(np.array(all_returns)).to(self.device)

        policy_loss_total, value_loss_total = 0, 0

        # PPO 训练迭代
        for _ in range(self.ppo_epochs):
            # 随机打乱索引，用于小批量训练
            indices = np.arange(len(local_states_tensor))
            np.random.shuffle(indices)
            for start in range(0, len(local_states_tensor), self.batch_size):
                batch_indices = indices[start:start + self.batch_size]
                # 获取当前批次的数据
                batch_local_states = local_states_tensor[batch_indices]
                batch_global_states = global_states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_log_probs_old = log_probs_old_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                # 计算当前策略的对数概率和价值
                policy = self.model.get_action_policy(batch_local_states)
                value = self.model.get_value(batch_global_states)

                log_probs = policy.log_prob(batch_actions).sum(dim=-1)
                entropy = policy.entropy().mean() # 熵

                # PPO 策略损失
                ratio = torch.exp(log_probs - batch_log_probs_old) # 重要性采样比率
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean() # 最小化裁剪后的目标

                # 价值损失 (MSE)
                value_loss = nn.MSELoss()(value.squeeze(), batch_returns)

                # 总损失
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                # 优化步骤
                self.optimizer.zero_grad() # 清除梯度
                loss.backward() # 反向传播
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm) # 梯度裁剪
                self.optimizer.step() # 更新网络参数

                policy_loss_total += policy_loss.item()
                value_loss_total += value_loss.item()

        # 返回平均损失
        num_batches = (len(local_states_tensor) + self.batch_size - 1) // self.batch_size # 计算批次数量
        return policy_loss_total / (self.ppo_epochs * num_batches), \
               value_loss_total / (self.ppo_epochs * num_batches)

    def save_best_weights(self):
        """保存当前模型的最佳权重。"""
        self.best_weights = self.model.state_dict()
        torch.save(self.best_weights, os.path.join(PPO_DIR, "best_model.pth"))
        print(f"模型权重已保存到: {os.path.join(PPO_DIR, 'best_model.pth')}")

    def load_best_weights(self):
        """加载已保存的最佳模型权重。"""
        model_path = os.path.join(PPO_DIR, "best_model.pth")
        if os.path.exists(model_path):
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=False))
            print(f"已从 {model_path} 加载模型权重。")
        else:
            print(f"警告: 未找到模型文件 '{model_path}'。")


# 主训练逻辑
def main():
    num_agents = 15 # 代理数量
    max_episodes = 6000 # 最大训练 episode 数量
    save_interval_path_plot = 50 # 每隔多少个 episode 保存一次路径图
    save_interval_curve_plot = 50 # 每隔多少个 episode 保存一次训练曲线图和数据
    patience = 3000 # 早停耐心值
    max_steps_per_episode = 200 # 每个 episode 的最大步数

    # 实例化环境
    env = MultiAgentPathPlanningEnv(OBSTACLES, PURE_WALKABLE_COORDS, GOAL_COORDS, MAP_BOUNDS, MAP_ROWS, MAP_COLS,
                                    num_agents, DEVICE, max_steps_per_episode)
    # 实例化 PPO 训练器
    trainer = PPOTrainer(env, DEVICE)

    # 训练过程中的统计数据
    episode_rewards = [] # 总奖励
    episode_policy_losses = [] # 策略损失
    episode_value_losses = [] # 价值损失
    episode_steps_sum_all_agvs = [] # 所有 AGV 的总步数
    avg_steps_reached_agvs = [] # 到达目标的 AGV 的平均步数
    episode_reach_rates = [] # 每个 episode 的到达率

    # 每个代理的奖励和步数历史
    episode_rewards_per_agent = [[] for _ in range(num_agents)]
    episode_steps_per_agent = [[] for _ in range(num_agents)]

    best_avg_reward = -float('inf') # 记录最佳平均奖励
    patience_counter = 0 # 早停计数器

    print("开始 PPO 训练...")
    with tqdm(range(max_episodes), desc="训练进度") as pbar:
        for episode in pbar:
            try:
                # 重置环境，获取初始状态
                local_states = env.reset()
                global_state = env._get_global_state()

                # 存储每个代理的轨迹
                trajectories_per_agent = [[] for _ in range(num_agents)]
                per_agent_episode_reward = [0.0] * num_agents
                per_agent_episode_steps = [0] * num_agents
                per_agent_reached_goal = [False] * num_agents

                current_episode_total_reward = 0
                steps_taken_overall_sum = 0 # 初始化为总步数

                all_agents_done = False # 标记所有代理是否都已完成

                # Episode 循环
                while not all_agents_done:
                    actions = []
                    log_probs = []
                    values = []

                    for i, state in enumerate(local_states):
                        # 如果代理已到达目标或达到最大步数，则不再行动
                        if not per_agent_reached_goal[i] and per_agent_episode_steps[i] < max_steps_per_episode:
                            # 修正: 移除多余的 'i' 参数
                            action_i, log_prob_i = trainer.select_action(state)
                            with torch.no_grad():
                                # 注意: 这里获取 value 应该用 global_state，因为 Critic 是中心化的
                                value_i = trainer.model.get_value(
                                    torch.FloatTensor(global_state).unsqueeze(0).to(DEVICE)).item()
                            actions.append(action_i)
                            log_probs.append(log_prob_i.item())
                            values.append(value_i)
                        else:
                            # 对于已完成的代理，动作设为零，对数概率和价值设为无效值
                            actions.append(np.zeros(2, dtype=np.float32))
                            log_probs.append(0.0)
                            values.append(0.0)

                    # 环境步进
                    next_local_states, next_global_state, rewards, all_agents_done, infos = env.step(actions)

                    # 收集轨迹数据
                    for i in range(num_agents):
                        # 仅将未完成的代理的经验添加到轨迹中
                        if not per_agent_reached_goal[i] and per_agent_episode_steps[i] < max_steps_per_episode:
                            trajectories_per_agent[i].append(
                                (local_states[i], global_state, actions[i], log_probs[i], rewards[i],
                                 infos[i]['steps'] >= max_steps_per_episode or infos[i]['reached_goal'], values[i]))

                        per_agent_episode_reward[i] += rewards[i]
                        per_agent_episode_steps[i] = infos[i]['steps']
                        per_agent_reached_goal[i] = infos[i]['reached_goal']

                    current_episode_total_reward = sum(per_agent_episode_reward)
                    steps_taken_overall_sum = sum(per_agent_episode_steps) # 所有 AGV 的步数总和
                    local_states = next_local_states
                    global_state = next_global_state

                # 过滤掉空的轨迹 (例如，代理在第一步就完成了)
                active_trajectories_for_training = [t for t in trajectories_per_agent if len(t) > 0]

                policy_loss, value_loss = 0.0, 0.0
                if active_trajectories_for_training:
                    policy_loss, value_loss = trainer.train_step(active_trajectories_for_training)
                else:
                    # 如果没有活动轨迹（例如，所有代理都在第一步就完成了），则损失为 0
                    pass

                # 记录训练统计数据
                episode_rewards.append(current_episode_total_reward)
                episode_policy_losses.append(policy_loss)
                episode_value_losses.append(value_loss)
                episode_steps_sum_all_agvs.append(steps_taken_overall_sum) # 存储总步数

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
                    '策略损失': f'{policy_loss:.4f}',
                    '价值损失': f'{value_loss:.4f}'
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
                                               os.path.join(PPO_DIR, f"path_episode_{episode + 1}.png"), AGENT_RADIUS)
                    print(f"已保存 Episode {episode + 1} 路径图: path_episode_{episode + 1}.png")

                # 定期保存训练曲线图和数据
                if (episode + 1) % save_interval_curve_plot == 0:
                    save_training_plots(list(range(len(episode_rewards))), episode_policy_losses, episode_value_losses,
                                        episode_rewards, episode_steps_sum_all_agvs,
                                        episode_rewards_per_agent, episode_steps_per_agent,
                                        avg_steps_reached_agvs, episode_reach_rates,
                                        os.path.join(PPO_DIR, f"training_curves_episode_{episode + 1}.png"))
                    print(f"已保存 Episode {episode + 1} 训练曲线图: training_curves_episode_{episode + 1}.png")

                    # 保存当前训练结果到 CSV
                    results_df = pd.DataFrame({
                        'episode': list(range(len(episode_rewards))),
                        'total_reward': episode_rewards,
                        'policy_loss': episode_policy_losses,
                        'value_loss': episode_value_losses,
                        'total_steps_sum_all_agvs': episode_steps_sum_all_agvs,
                        'avg_reward_per_agv_per_episode': np.mean(np.array(episode_rewards_per_agent), axis=0),
                        'avg_steps_per_agv_per_episode': np.mean(np.array(episode_steps_per_agent), axis=0),
                        'avg_steps_reached_agvs': avg_steps_reached_agvs,
                        'episode_reach_rates': episode_reach_rates
                    })
                    results_df.to_csv(os.path.join(PPO_DIR, f"training_results_episode_{episode + 1}.csv"), index=False)
                    print(f"已保存 Episode {episode + 1} 训练结果数据: training_results_episode_{episode + 1}.csv")


                # 早停逻辑
                if avg_reward_last_100 > best_avg_reward:
                    best_avg_reward = avg_reward_last_100
                    trainer.save_best_weights() # 保存最佳模型
                    patience_counter = 0 # 重置耐心计数器
                    print(f"\n检测到最佳平均奖励，正在保存模型权重。")
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"\n在 {episode + 1} 个 episode 后达到耐心上限，提前停止训练。")
                    break # 提前停止训练

            except Exception as e:
                print(f"\n训练过程中发生错误: {e}")
                traceback.print_exc() # 打印详细错误信息
                break # 发生错误时停止训练

    print("训练结束。")

    # 训练结束后保存最终训练曲线图和结果数据
    save_training_plots(list(range(len(episode_rewards))), episode_policy_losses, episode_value_losses,
                        episode_rewards, episode_steps_sum_all_agvs,
                        episode_rewards_per_agent, episode_steps_per_agent,
                        avg_steps_reached_agvs, episode_reach_rates,
                        os.path.join(PPO_DIR, "final_training_curves.png"))
    print(f"最终训练曲线图已保存到: {os.path.join(PPO_DIR, 'final_training_curves.png')}")

    # 将训练结果保存为 CSV (最终保存，确保完整性)
    results_df = pd.DataFrame({
        'episode': list(range(len(episode_rewards))),
        'total_reward': episode_rewards,
        'policy_loss': episode_policy_losses,
        'value_loss': episode_value_losses,
        'total_steps_sum_all_agvs': episode_steps_sum_all_agvs,
        'avg_reward_per_agv_per_episode': np.mean(np.array(episode_rewards_per_agent), axis=0),
        'avg_steps_per_agv_per_episode': np.mean(np.array(episode_steps_per_agent), axis=0),
        'avg_steps_reached_agvs': avg_steps_reached_agvs,
        'episode_reach_rates': episode_reach_rates
    })
    results_df.to_csv(os.path.join(PPO_DIR, "training_results.csv"), index=False)
    print(f"训练结果数据已保存到: {os.path.join(PPO_DIR, 'training_results.csv')}")

    print("\n开始最终测试...")
    model_path = os.path.join(PPO_DIR, "best_model.pth")
    if os.path.exists(model_path):
        trainer.load_best_weights() # 加载最佳模型权重
        local_states = env.reset() # 重置环境进行测试
        global_state = env._get_global_state()
        test_paths = [[] for _ in range(num_agents)]
        test_total_rewards = [0] * num_agents
        test_steps = [0] * num_agents
        test_reached_goals = [False] * num_agents

        # 记录初始位置
        for i, agent_data in enumerate(env.agents):
            test_paths[i].append(agent_data['pos'].copy())

        all_test_dones = False

        # 测试循环
        while not all_test_dones:
            actions = []
            for state in local_states:
                action, _ = trainer.select_action(state) # 测试时只选择动作，不计算对数概率
                actions.append(action)

            next_local_states, next_global_state, rewards, all_test_dones, infos = env.step(actions)

            # 记录测试结果
            for i in range(num_agents):
                test_paths[i].append(env.agents[i]['pos'].copy())
                test_total_rewards[i] += rewards[i]
                test_steps[i] = infos[i]['steps']
                test_reached_goals[i] = infos[i]['reached_goal']
            local_states = next_local_states
            global_state = next_global_state

        print(f"--- 最终测试单个 AGV 性能 ---")
        final_test_paths_data = []
        for i in range(num_agents):
            final_test_paths_data.append({
                'path': test_paths[i],
                'goal': env.agents[i]['goal']
            })
            print(
                f" AGV {i}: 奖励={test_total_rewards[i]:.2f}, 步数={test_steps[i]}, 到达目标={test_reached_goals[i]}")
        print(f"-----------------------------")

        # 保存最终测试路径图
        save_multi_agent_path_plot(OBSTACLES, GOAL_COORDS, MAP_BOUNDS, final_test_paths_data,
                                   os.path.join(PPO_DIR, "final_test_paths.png"), AGENT_RADIUS)
        print(f"最终测试路径图已保存到: {os.path.join(PPO_DIR, 'final_test_paths.png')}")
        print("测试结束。")
    else:
        print(
            f"警告: 未找到最佳模型文件 '{model_path}'。跳过最终测试。请确保训练过程成功保存了模型。")


if __name__ == "__main__":
    main()

