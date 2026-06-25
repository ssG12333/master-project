import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import seaborn as sns

# 从 environment.py 导入必要的常量
# 确保 environment.py 在同一目录下或者在 Python 路径中
try:
    from environment import CELL_SIZE, AGENT_RADIUS, MAP_BOUNDS, OBSTACLES, GOAL_COORDS, DEVICE
except ImportError:
    print("警告: 无法从 environment.py 导入常量。请确保 environment.py 文件存在且可访问。")
    # 提供默认值以防止程序崩溃，但在实际运行中需要 environment.py
    CELL_SIZE = 1.0
    AGENT_RADIUS = 0.4 * CELL_SIZE
    MAP_BOUNDS = [0, 35, 0, 35] # 假设一个默认的地图边界
    OBSTACLES = []
    GOAL_COORDS = []
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 获取 GPU 内存使用信息
def get_gpu_memory_usage():
    """
    获取当前和总 GPU 内存使用量 (MB)。
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(DEVICE) / (1024 ** 2) # 已分配内存
        cached = torch.cuda.memory_reserved(DEVICE) / (1024 ** 2) # 缓存内存
        return allocated, cached
    return 0, 0 # 如果 CUDA 不可用，返回 0

# 保存多代理路径图
def save_multi_agent_path_plot(obstacles, goal_coords, map_bounds, agents_paths, filename, agent_radius=AGENT_RADIUS):
    """
    绘制并保存多代理路径图。
    obstacles: 障碍物段列表。
    goal_coords: 目标坐标列表。
    map_bounds: 地图边界。
    agents_paths: 字典列表，每个字典包含代理路径和目标。
    filename: 保存图像的文件名。
    agent_radius: 代理的显示半径。
    """
    plt.figure(figsize=(15, 10), dpi=200)
    ax = plt.gca()

    # 绘制网格线
    rows = int((map_bounds[3] - map_bounds[2]) / CELL_SIZE)
    cols = int((map_bounds[1] - map_bounds[0]) / CELL_SIZE)
    for x in range(cols + 1):
        ax.axvline(x * CELL_SIZE, color='lightgray', linestyle='-', linewidth=0.5)
    for y in range(rows + 1):
        ax.axhline(y * CELL_SIZE, color='lightgray', linestyle='-', linewidth=0.5)

    # 绘制障碍物
    for obs in obstacles:
        plt.plot([obs[0][0], obs[1][0]], [obs[0][1], obs[1][1]], 'k-', linewidth=2, zorder=1)

    # 绘制目标位置
    for goal in goal_coords:
        goal_circle = plt.Circle(goal, agent_radius * 1.5, color='blue', alpha=0.5, zorder=2)
        ax.add_patch(goal_circle)
        plt.plot(goal[0], goal[1], 'bx', markersize=6, markeredgewidth=2, zorder=3)

    # 绘制代理路径和起点/目标点
    cmap = plt.colormaps['hsv'] # 使用 HSV 颜色映射
    colors = [cmap(i / len(agents_paths)) for i in range(len(agents_paths))] # 为每个代理分配不同颜色

    for i, path_data in enumerate(agents_paths):
        path = np.array(path_data['path'])
        start_pos = path[0]
        goal_pos = path_data['goal']

        color = colors[i]

        plt.plot(path[:, 0], path[:, 1], color=color, linewidth=1.5, label=f'AGV {i} Path', zorder=3, alpha=0.8)
        plt.plot(start_pos[0], start_pos[1], 'o', markersize=5, color=color, markeredgecolor='black',
                 label=f'AGV {i} Start', zorder=4)
        plt.plot(goal_pos[0], goal_pos[1], 'D', markersize=5, color=color, markeredgecolor='black',
                 label=f'AGV {i} Goal', zorder=4)

    ax.set_aspect('equal', adjustable='box') # 保持宽高比
    plt.xlim(map_bounds[0], map_bounds[1]) # 设置 X 轴范围
    plt.ylim(map_bounds[2], map_bounds[3]) # 设置 Y 轴范围
    plt.xlabel('X Coordinate', fontsize=12) # X 轴标签
    plt.ylabel('Y Coordinate', fontsize=12) # Y 轴标签
    plt.title('Multi-Agent Path Planning', fontsize=14, pad=20) # 标题
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7) # 显示网格
    plt.tight_layout() # 调整布局
    plt.savefig(filename, bbox_inches='tight') # 保存图像
    plt.close() # 关闭图形，释放内存

# 保存训练过程中各种性能曲线
def save_training_plots(episodes, policy_loss, value_loss,
                        total_episode_rewards, total_episode_steps_sum,
                        all_agents_rewards_history, all_agents_steps_history,
                        avg_steps_reached_agvs, episode_reach_rates, # 新增参数
                        filename):
    """
    绘制并保存训练性能曲线。
    episodes: episode 编号列表。
    policy_loss: 策略损失列表。
    value_loss: 价值损失列表。
    total_episode_rewards: 每个 episode 中所有代理的总奖励列表。
    total_episode_steps_sum: 每个 episode 中所有代理的总步数列表。
    all_agents_rewards_history: 列表的列表，all_agents_rewards_history[agv_idx][episode_num]
    all_agents_steps_history: 列表的列表，all_agents_steps_history[agv_idx][episode_num]
    avg_steps_reached_agvs: 每个 episode 中到达目标的 AGV 的平均步数列表。
    episode_reach_rates: 每个 episode 的到达率列表。
    filename: 保存图像的文件名。
    """
    # 确定需要绘制的图表数量，根据传入的损失数据决定
    num_plots = 6 # 默认有 6 个图 (总奖励, 总步数, 成功率, 到达AGV平均步数, AGV平均奖励, AGV平均步数)
    if policy_loss is not None and len(policy_loss) > 0:
        num_plots += 1 # 如果有策略损失，增加一个图
    if value_loss is not None and len(value_loss) > 0:
        num_plots += 1 # 如果有价值损失，增加一个图

    # 计算子图的行数和列数，使其尽可能接近正方形
    rows = int(np.ceil(np.sqrt(num_plots)))
    cols = int(np.ceil(num_plots / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows)) # 调整 figsize
    axes = axes.flatten() # 将 axes 展平，方便迭代
    fig.suptitle('Training Performance Metrics', fontsize=18)

    plot_idx = 0

    # 图 1: 策略损失 (如果存在)
    if policy_loss is not None and len(policy_loss) > 0:
        sns.lineplot(x=episodes, y=policy_loss, ax=axes[plot_idx], label='Policy Loss', color='blue')
        axes[plot_idx].set_title('Policy Loss')
        axes[plot_idx].set_xlabel('Episodes')
        axes[plot_idx].set_ylabel('Loss')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, linestyle='--', alpha=0.6)
        plot_idx += 1

    # 图 2: 价值损失 (如果存在)
    if value_loss is not None and len(value_loss) > 0:
        sns.lineplot(x=episodes, y=value_loss, ax=axes[plot_idx], label='Value Loss', color='green')
        axes[plot_idx].set_title('Value Loss')
        axes[plot_idx].set_xlabel('Episodes')
        axes[plot_idx].set_ylabel('Loss')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, linestyle='--', alpha=0.6)
        plot_idx += 1

    # 从历史记录中计算每个 AGV 每个 episode 的平均奖励
    rewards_np = np.array(all_agents_rewards_history) # 形状 (num_agents, num_episodes)
    if rewards_np.shape[1] > 0: # 确保有数据
        avg_reward_per_agv_per_episode = np.mean(rewards_np, axis=0)
    else:
        avg_reward_per_agv_per_episode = []

    # 图 3: Total Episode Reward (Sum of all AGVs)
    sns.lineplot(x=episodes, y=total_episode_rewards, ax=axes[plot_idx], label='Total Episode Reward', color='purple')
    axes[plot_idx].set_title('Total Episode Reward (Sum of all AGVs)')
    axes[plot_idx].set_xlabel('Episodes')
    axes[plot_idx].set_ylabel('Reward')
    axes[plot_idx].legend()
    axes[plot_idx].grid(True, linestyle='--', alpha=0.6)
    plot_idx += 1

    # 图 4: Average Reward per AGV per Episode
    if len(avg_reward_per_agv_per_episode) > 0:
        sns.lineplot(x=episodes, y=avg_reward_per_agv_per_episode, ax=axes[plot_idx], label='Average Reward per AGV',
                     color='orange')
        axes[plot_idx].set_title('Average Reward per AGV per Episode')
        axes[plot_idx].set_xlabel('Episodes')
        axes[plot_idx].set_ylabel('Reward')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, linestyle='--', alpha=0.6)
    plot_idx += 1

    # 从历史记录中计算每个 AGV 每个 episode 的平均步数
    steps_np = np.array(all_agents_steps_history) # 形状 (num_agents, num_episodes)
    if steps_np.shape[1] > 0: # 确保有数据
        avg_steps_per_agv_per_episode = np.mean(steps_np, axis=0)
    else:
        avg_steps_per_agv_per_episode = []

    # 图 5: Total Episode Steps (Sum of all AGVs)
    sns.lineplot(x=episodes, y=total_episode_steps_sum, ax=axes[plot_idx], label='Total Episode Steps (Sum of all AGVs)',
                 color='brown')
    axes[plot_idx].set_title('Total Episode Steps (Sum of all AGVs)')
    axes[plot_idx].set_xlabel('Episodes')
    axes[plot_idx].set_ylabel('Steps')
    axes[plot_idx].legend()
    axes[plot_idx].grid(True, linestyle='--', alpha=0.6)
    plot_idx += 1

    # 图 6: Average Steps per AGV per Episode
    if len(avg_steps_per_agv_per_episode) > 0:
        sns.lineplot(x=episodes, y=avg_steps_per_agv_per_episode, ax=axes[plot_idx], label='Average Steps per AGV',
                     color='teal')
        axes[plot_idx].set_title('Average Steps per AGV per Episode')
        axes[plot_idx].set_xlabel('Episodes')
        axes[plot_idx].set_ylabel('Steps')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, linestyle='--', alpha=0.6)
    plot_idx += 1

    # 图 7: Average Steps for Reached AGVs (新增)
    if len(avg_steps_reached_agvs) > 0:
        sns.lineplot(x=episodes, y=avg_steps_reached_agvs, ax=axes[plot_idx], label='Avg Steps for Reached AGVs',
                     color='darkred')
        axes[plot_idx].set_title('Average Steps for Reached AGVs')
        axes[plot_idx].set_xlabel('Episodes')
        axes[plot_idx].set_ylabel('Steps')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, linestyle='--', alpha=0.6)
    plot_idx += 1

    # 图 8: Episode Reach Rate (新增)
    if len(episode_reach_rates) > 0:
        sns.lineplot(x=episodes, y=episode_reach_rates, ax=axes[plot_idx], label='Episode Reach Rate', color='darkgreen')
        axes[plot_idx].set_title('Episode Reach Rate')
        axes[plot_idx].set_xlabel('Episodes')
        axes[plot_idx].set_ylabel('Reach Rate')
        axes[plot_idx].set_ylim(0, 1.05) # 到达率在 0 到 1 之间
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, linestyle='--', alpha=0.6)
    plot_idx += 1

    # 隐藏未使用的子图
    for i in range(plot_idx, len(axes)):
        fig.delaxes(axes[i])

    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局，为总标题留出空间
    plt.savefig(filename, dpi=150) # 保存图像
    plt.close(fig) # 关闭图形，释放内存

