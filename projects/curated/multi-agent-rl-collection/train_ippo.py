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
from environment import MultiAgentPathPlanningEnv, OBSTACLES, PURE_WALKABLE_COORDS, GOAL_COORDS, MAP_BOUNDS, MAP_ROWS, \
    MAP_COLS, DEVICE, AGENT_RADIUS
# 从 ippo_network.py 导入 IPPO 网络
from ippo_network import IPPONetwork
# 从 utils.py 导入辅助函数
from utils import get_gpu_memory_usage, save_multi_agent_path_plot, save_training_plots

# IPPO 训练结果和模型保存目录
IPPO_DIR = "IPPO_Results1"
if not os.path.exists(IPPO_DIR):
    os.makedirs(IPPO_DIR)
    print(f"创建目录: {IPPO_DIR}")


# IPPO 训练器 (去中心化 Actor, 去中心化 Critic)
class IPPOTrainer(nn.Module):
    def __init__(self, env, device):
        """
        IPPO 训练器。
        env: 环境实例。
        device: PyTorch 设备 (CPU/CUDA)。
        """
        super(IPPOTrainer, self).__init__()
        self.env = env
        self.device = device
        self.num_agents = env.num_agents

        # 计算局部状态维度
        other_agents_info_dim = 3 * (self.num_agents - 1) if self.num_agents > 1 else 0
        self.local_state_dim = 2 + 2 + 1 + self.env.num_rays + other_agents_info_dim
        self.action_dim = 2  # 动作维度 (vx, vy)

        # 每个代理都有自己的 IPPONetwork (Actor 和 Critic)
        self.models = [IPPONetwork(self.local_state_dim, self.action_dim).to(device) for _ in range(self.num_agents)]
        # 每个代理都有自己的优化器
        self.optimizers = [optim.Adam(model.parameters(), lr=0.0003) for model in self.models]

        # PPO 算法参数
        self.gamma = 0.99  # 折扣因子
        self.lam = 0.95  # GAE 参数
        self.clip_param = 0.2  # PPO 裁剪参数
        self.value_loss_coef = 0.5  # 价值损失系数
        self.entropy_coef = 0.02  # 熵系数，鼓励探索
        self.max_grad_norm = 0.5  # 梯度裁剪最大范数
        self.ppo_epochs = 10  # PPO 训练迭代次数
        self.batch_size = 2048  # 训练批处理大小

        # 用于保存最佳模型权重 (每个代理一个文件)
        self.best_weights = [model.state_dict() for model in self.models]

    def select_action(self, local_state, agent_idx):
        """
        根据当前局部状态从策略中选择一个动作。
        local_state: 单个代理的局部观测状态 (NumPy 数组)。
        agent_idx: 代理索引。
        返回:
            action (np.array): 采样得到的动作。
            log_prob (float): 动作的对数概率 (Python 浮点数)。
            value (float): 状态价值 (Python 浮点数)。
        """
        model = self.models[agent_idx]
        local_state_tensor = torch.FloatTensor(local_state).unsqueeze(0).to(self.device)

        policy = model.get_action_policy(local_state_tensor)
        action = policy.sample()
        log_prob = policy.log_prob(action).sum(dim=-1)  # This is a tensor on device
        value = model.get_value(local_state_tensor)  # This is a tensor on device

        # Ensure all returned values are on CPU and are Python scalars/NumPy arrays
        return action.squeeze(0).cpu().numpy(), log_prob.squeeze(0).cpu().item(), value.squeeze(0).cpu().item()

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
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            next_value = values[i]
        advantages = np.array(advantages, dtype=np.float32)
        returns = advantages + values
        return advantages, returns

    def train_step(self, trajectories_per_agent):
        """
        执行一个 IPPO 训练步骤。
        trajectories_per_agent: 列表，每个元素是单个代理在一个 episode 中的轨迹。
        每个轨迹元素包含 (local_state, action, log_prob_old, reward, done, value)。
        """
        policy_loss_total = 0.0
        value_loss_total = 0.0

        for i in range(self.num_agents):  # 对每个代理独立训练
            agent_trajectory = trajectories_per_agent[i]
            if not agent_trajectory:  # 如果代理没有收集到任何经验，跳过
                continue

            # 收集当前代理的轨迹数据
            local_states = np.array([t[0] for t in agent_trajectory])
            actions = np.array([t[1] for t in agent_trajectory])
            log_probs_old = np.array([t[2] for t in agent_trajectory])  # 这里的 t[2] 现在已经是 float
            rewards = np.array([t[3] for t in agent_trajectory])
            dones = np.array([t[4] for t in agent_trajectory])
            values = np.array([t[5] for t in agent_trajectory]).flatten()

            # 将 NumPy 数组转换为 PyTorch 张量并移动到设备
            local_states_tensor = torch.FloatTensor(local_states).to(self.device)
            actions_tensor = torch.FloatTensor(actions).to(self.device)
            log_probs_old_tensor = torch.FloatTensor(log_probs_old).to(self.device)
            rewards_tensor = torch.FloatTensor(rewards).to(self.device)
            dones_tensor = torch.FloatTensor(dones).to(self.device)
            values_tensor = torch.FloatTensor(values).to(self.device)

            # 计算 GAE 和回报
            with torch.no_grad():
                # 获取轨迹最后一个状态的下一个状态的价值估计
                last_local_state_of_trajectory = agent_trajectory[-1][0]
                # last_value_estimate 已经是 float，不需要 .item()
                last_value_estimate = self.models[i].get_value(
                    torch.FloatTensor(last_local_state_of_trajectory).unsqueeze(0).to(self.device)).item()
                next_value_for_gae = last_value_estimate * (1 - dones[-1])

            advantages, returns = self.compute_gae(rewards, values, next_value_for_gae, dones)

            advantages_tensor = torch.FloatTensor(advantages).to(self.device)
            returns_tensor = torch.FloatTensor(returns).to(self.device)

            # PPO 训练迭代
            for _ in range(self.ppo_epochs):
                indices = np.arange(len(local_states_tensor))
                np.random.shuffle(indices)
                for start in range(0, len(local_states_tensor), self.batch_size):
                    batch_indices = indices[start:start + self.batch_size]
                    batch_local_states = local_states_tensor[batch_indices]
                    batch_actions = actions_tensor[batch_indices]
                    batch_log_probs_old = log_probs_old_tensor[batch_indices]
                    batch_advantages = advantages_tensor[batch_indices]
                    batch_returns = returns_tensor[batch_indices]

                    model = self.models[i]  # 当前代理的模型
                    optimizer = self.optimizers[i]  # 当前代理的优化器

                    policy = model.get_action_policy(batch_local_states)
                    value = model.get_value(batch_local_states)

                    log_probs = policy.log_prob(batch_actions).sum(dim=-1)
                    entropy = policy.entropy().mean()

                    # PPO 策略损失
                    ratio = torch.exp(log_probs - batch_log_probs_old)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # 价值损失 (MSE)
                    value_loss = nn.MSELoss()(value.squeeze(), batch_returns)

                    # 总损失
                    loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                    # 优化步骤
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                    optimizer.step()

                    policy_loss_total += policy_loss.item()
                    value_loss_total += value_loss.item()

        # 返回平均损失 (所有代理的平均值)
        num_active_agents = sum(1 for t in trajectories_per_agent if len(t) > 0)
        if num_active_agents == 0:
            return 0.0, 0.0  # 如果没有活动代理，返回0

        # 确保 num_batches_per_agent 不为零
        num_batches_per_agent = 1
        if trajectories_per_agent and len(trajectories_per_agent[0]) > 0:
            num_batches_per_agent = (len(trajectories_per_agent[0]) + self.batch_size - 1) // self.batch_size

        return policy_loss_total / (self.ppo_epochs * num_batches_per_agent * num_active_agents), \
               value_loss_total / (self.ppo_epochs * num_batches_per_agent * num_active_agents)

    def save_best_weights(self):
        """保存当前模型的最佳权重。"""
        for i, model in enumerate(self.models):
            self.best_weights[i] = model.state_dict()
            torch.save(self.best_weights[i], os.path.join(IPPO_DIR, f"best_model_agent_{i}.pth"))
        print(f"IPPO 模型权重已保存到: {IPPO_DIR}")

    def load_best_weights(self):
        """加载已保存的最佳模型权重。"""
        for i, model in enumerate(self.models):
            model_path = os.path.join(IPPO_DIR, f"best_model_agent_{i}.pth")
            if os.path.exists(model_path):
                model.load_state_dict(
                    torch.load(model_path, map_location=self.device, weights_only=False))
            else:
                print(f"警告: 未找到代理 {i} 的 IPPO 模型文件 '{model_path}'。")
        print(f"已从 {IPPO_DIR} 加载 IPPO 模型权重。")


# 主训练逻辑
def main():
    num_agents = 15  # 代理数量
    max_episodes = 6000  # 最大训练 episode 数量
    save_interval_path_plot = 50  # 每隔多少个 episode 保存一次路径图
    save_interval_curve_plot = 50  # 每隔多少个 episode 保存一次训练曲线图和数据
    patience = 3000  # 早停耐心值
    max_steps_per_episode = 200  # 每个 episode 的最大步数

    # 实例化环境
    env = MultiAgentPathPlanningEnv(OBSTACLES, PURE_WALKABLE_COORDS, GOAL_COORDS, MAP_BOUNDS, MAP_ROWS, MAP_COLS,
                                    num_agents, DEVICE, max_steps_per_episode)
    # 实例化 IPPO 训练器
    trainer = IPPOTrainer(env, DEVICE)

    # 训练过程中的统计数据
    episode_rewards = []  # 总奖励
    episode_policy_losses = []  # 策略损失
    episode_value_losses = []  # 价值损失
    episode_steps_sum_all_agvs = []  # 所有 AGV 的总步数
    avg_steps_reached_agvs = []  # 到达目标的 AGV 的平均步数
    episode_reach_rates = []  # 每个 episode 的到达率

    # 每个代理的奖励和步数历史
    episode_rewards_per_agent = [[] for _ in range(num_agents)]
    episode_steps_per_agent = [[] for _ in range(num_agents)]

    best_avg_reward = -float('inf')  # 记录最佳平均奖励
    patience_counter = 0  # 早停计数器

    print("开始 IPPO 训练...")
    with tqdm(range(max_episodes), desc="训练进度") as pbar:
        for episode in pbar:
            try:
                # 重置环境，获取初始状态
                local_states = env.reset()
                global_state = env._get_global_state()  # IPPO 训练时实际上不直接用 global_state，但为了兼容环境接口保留

                # 存储每个代理的轨迹
                trajectories_per_agent = [[] for _ in range(num_agents)]
                per_agent_episode_reward = [0.0] * num_agents
                per_agent_episode_steps = [0] * num_agents
                per_agent_reached_goal = [False] * num_agents

                current_episode_total_reward = 0
                steps_taken_overall_sum = 0

                all_agents_done = False

                # Episode 循环
                while not all_agents_done:
                    actions = []
                    current_values = []  # 存储当前步的价值估计
                    current_log_probs = []  # 存储当前步的对数概率

                    for i, state in enumerate(local_states):
                        # 如果代理已到达目标或达到最大步数，则不再行动
                        if not per_agent_reached_goal[i] and per_agent_episode_steps[i] < max_steps_per_episode:
                            action_i, log_prob_i, value_i = trainer.select_action(state, i)
                            actions.append(action_i)
                            current_values.append(value_i)  # value_i 已经是 float
                            current_log_probs.append(log_prob_i)  # log_prob_i 已经是 float
                        else:
                            actions.append(np.zeros(trainer.action_dim, dtype=np.float32))
                            current_values.append(0.0)  # 无效价值
                            current_log_probs.append(0.0)  # 无效对数概率

                    # 环境步进
                    next_local_states, next_global_state, rewards, all_agents_done, infos = env.step(actions)

                    # 收集轨迹数据
                    for i in range(num_agents):
                        if not per_agent_reached_goal[i] and per_agent_episode_steps[i] < max_steps_per_episode:
                            # IPPO 轨迹只包含局部信息
                            trajectories_per_agent[i].append(
                                (local_states[i], actions[i], current_log_probs[i], rewards[i],  # 使用 current_log_probs
                                 infos[i]['steps'] >= max_steps_per_episode or infos[i]['reached_goal'],
                                 current_values[i]))  # 记录当前步的价值

                        per_agent_episode_reward[i] += rewards[i]
                        per_agent_episode_steps[i] = infos[i]['steps']
                        per_agent_reached_goal[i] = infos[i]['reached_goal']

                    current_episode_total_reward = sum(per_agent_episode_reward)
                    steps_taken_overall_sum = sum(per_agent_episode_steps)
                    local_states = next_local_states
                    global_state = next_global_state  # 仍然更新 global_state，尽管 IPPO 不直接使用

                # 训练网络
                policy_loss, value_loss = trainer.train_step(trajectories_per_agent)

                # 记录训练统计数据
                episode_rewards.append(current_episode_total_reward)
                episode_policy_losses.append(policy_loss)
                episode_value_losses.append(value_loss)
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
                                               os.path.join(IPPO_DIR, f"path_episode_{episode + 1}.png"), AGENT_RADIUS)
                    print(f"已保存 Episode {episode + 1} 路径图: path_episode_{episode + 1}.png")

                # 定期保存训练曲线图和数据
                if (episode + 1) % save_interval_curve_plot == 0:
                    save_training_plots(list(range(len(episode_rewards))), episode_policy_losses, episode_value_losses,
                                        episode_rewards, episode_steps_sum_all_agvs,
                                        episode_rewards_per_agent, episode_steps_per_agent,
                                        avg_steps_reached_agvs, episode_reach_rates,
                                        os.path.join(IPPO_DIR, f"training_curves_episode_{episode + 1}.png"))
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
                    results_df.to_csv(os.path.join(IPPO_DIR, f"training_results_episode_{episode + 1}.csv"),
                                      index=False)
                    print(f"已保存 Episode {episode + 1} 训练结果数据: training_results_episode_{episode + 1}.csv")

                # 早停逻辑
                if avg_reward_last_100 > best_avg_reward:
                    best_avg_reward = avg_reward_last_100
                    trainer.save_best_weights()
                    patience_counter = 0
                    print(f"\n检测到最佳平均奖励，正在保存 IPPO 模型权重。")
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"\n在 {episode + 1} 个 episode 后达到耐心上限，提前停止 IPPO 训练。")
                    break

            except Exception as e:
                print(f"\nIPPO 训练过程中发生错误: {e}")
                traceback.print_exc()
                break

    print("IPPO 训练结束。")

    # 训练结束后保存最终训练曲线图和结果数据
    save_training_plots(list(range(len(episode_rewards))), episode_policy_losses, episode_value_losses,
                        episode_rewards, episode_steps_sum_all_agvs,
                        episode_rewards_per_agent, episode_steps_per_agent,
                        avg_steps_reached_agvs, episode_reach_rates,
                        os.path.join(IPPO_DIR, "final_training_curves.png"))
    print(f"最终 IPPO 训练曲线图已保存到: {os.path.join(IPPO_DIR, 'final_training_curves.png')}")

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
    results_df.to_csv(os.path.join(IPPO_DIR, "training_results.csv"), index=False)
    print(f"IPPO 训练结果数据已保存到: {os.path.join(IPPO_DIR, 'training_results.csv')}")

    print("\n开始最终 IPPO 测试...")
    # 加载最佳模型权重
    trainer.load_best_weights()
    local_states = env.reset()
    global_state = env._get_global_state()  # 测试时也需要 global_state 来更新环境
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
            action, _, _ = trainer.select_action(state, i)  # 测试时只选择动作
            actions.append(action)

        next_local_states, next_global_state, rewards, all_test_dones, infos = env.step(actions)

        for i in range(num_agents):
            test_paths[i].append(env.agents[i]['pos'].copy())
            test_total_rewards[i] += rewards[i]
            test_steps[i] = infos[i]['steps']
            test_reached_goals[i] = infos[i]['reached_goal']
        local_states = next_local_states
        global_state = next_global_state

    print(f"--- 最终 IPPO 测试单个 AGV 性能 ---")
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
                               os.path.join(IPPO_DIR, "final_test_paths.png"), AGENT_RADIUS)
    print(f"最终 IPPO 测试路径图已保存到: {os.path.join(IPPO_DIR, 'final_test_paths.png')}")
    print("IPPO 测试结束。")


if __name__ == "__main__":
    main()

