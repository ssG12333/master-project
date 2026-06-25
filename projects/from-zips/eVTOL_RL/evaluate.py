import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List

from envs.evtol_env import EVTOLEnvironment
from envs.obstacles import TrainingStage
from agents.td3 import TD3Agent


class Evaluator:
    '''Evaluation suite for trained agents'''

    def __init__(self, agent, env, output_dir, config = None):
        self.agent = agent
        self.env = env
        self.output_dir = output_dir
        self.config = config or {}
        os.makedirs(output_dir, exist_ok=True)

    def evaluate_performance(self, num_episodes: int = 100) -> Dict:
        '''Evaluate agent performance'''
        print(f"\nEvaluating agent performance over {num_episodes} episodes...")

        metrics = {
            'success_rate': 0,
            'avg_reward': 0,
            'avg_steps': 0,
            'collision_rate': 0,
            'avg_path_length': 0,
            'avg_safety_coverage': 0
        }

        successes = []
        rewards = []
        steps = []
        collisions = []
        path_lengths = []
        safety_coverages = []

        for ep in range(num_episodes):
            if (ep + 1) % 20 == 0:    # 每20个回合显示一次训练进度
                print(f"  Progress: {ep + 1}/{num_episodes} episodes")

            state = self.env.reset(seed=1000 + ep)    # seed=1000 + ep：为每个回合设置不同的随机种子1000 + ep 确保每个回合的初始条件都不同但可重复
            episode_reward = 0
            episode_steps = 0
            trajectory = []

            for _ in range(self.env.max_steps):
                action = self.agent.select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, info = self.env.step(action)

                episode_reward += reward
                episode_steps += 1
                trajectory.append(info['position'].copy())

                if terminated or truncated:
                    # Check success
                    if info['distance_to_goal'] < self.env.target_radius:
                        successes.append(1)
                    else:
                        successes.append(0)

                    # Check collision
                    d_safe = self.config.get('environment', {}).get('collision_distance_threshold', 7.5)
                    if terminated and info['distance_to_goal'] >= self.env.target_radius:
                        min_dist, _ = self.env.obstacle_gen.get_nearest_obstacle(self.env.dynamics.position)
                        if min_dist < d_safe:
                            collisions.append(1)
                        else:
                            collisions.append(0)
                    else:
                        collisions.append(0)
                    break
                state = next_state

            # Calculate path metrics
            trajectory = np.array(trajectory)
            if len(trajectory) > 1:
                path_length = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
            else:
                path_length = 0.0

            # Calculate safety coverage
            safety_distances = []
            for pos in trajectory:
                min_dist, _ = self.env.obstacle_gen.get_nearest_obstacle(pos)
                safety_distances.append(min_dist)
            safety_threshold = self.config.get('environment', {}).get('safety_distance_threshold', 20.0)
            safety_coverage = np.mean(np.array(safety_distances) > safety_threshold) if safety_distances else 0.0

            rewards.append(episode_reward)
            steps.append(episode_steps)
            path_lengths.append(path_length)
            safety_coverages.append(safety_coverage)

        metrics['success_rate'] = np.mean(successes)
        metrics['avg_reward'] = np.mean(rewards)
        metrics['avg_steps'] = np.mean(steps)
        metrics['collision_rate'] = np.mean(collisions)
        metrics['avg_path_length'] = np.mean(path_lengths)
        metrics['avg_safety_coverage'] = np.mean(safety_coverages)

        # Calculate additional statistics
        metrics.update({
            'reward_std': np.std(rewards),
            'steps_std': np.std(steps),
            'path_length_std': np.std(path_lengths),
            'safety_coverage_std': np.std(safety_coverages)
        })

        print(f"\n" + "=" * 60)
        print(f"EVALUATION RESULTS ({num_episodes} episodes)")
        print("=" * 60)
        print(f"Success Rate:        {metrics['success_rate']:.2%}")
        print(f"Collision Rate:      {metrics['collision_rate']:.2%}")
        print(f"Average Reward:      {metrics['avg_reward']:.1f} ± {metrics['reward_std']:.1f}")
        print(f"Average Steps:       {metrics['avg_steps']:.1f} ± {metrics['steps_std']:.1f}")
        print(f"Average Path Length: {metrics['avg_path_length']:.1f}m ± {metrics['path_length_std']:.1f}m")
        print(f"Safety Coverage:     {metrics['avg_safety_coverage']:.2%} ± {metrics['safety_coverage_std']:.2%}")
        print("=" * 60)

        # Save results to file
        results_file = os.path.join(self.output_dir, 'evaluation_results.txt')
        with open(results_file, 'w') as f:
            f.write(f"Evaluation Results ({num_episodes} episodes)\n")
            f.write("=" * 50 + "\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")

        print(f"Results saved to: {results_file}")

        return metrics

    def visualize_trajectory(self, episode_seed: int = 42):
        '''Visualize sample trajectory'''
        print(f"\nGenerating trajectory visualization (seed={episode_seed})...")

        state = self.env.reset(seed=episode_seed)
        trajectory = []
        actions = []
        rewards = []
        velocities = []

        for step in range(self.env.max_steps):
            action = self.agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, info = self.env.step(action)

            trajectory.append(info['position'].copy())
            actions.append(action.copy())
            rewards.append(reward)
            velocities.append(self.env.dynamics.velocity.copy())

            state = next_state

            if terminated or truncated:
                success = info['distance_to_goal'] < self.env.target_radius
                print(f"  Episode completed in {step + 1} steps")
                print(f"  Result: {'Success' if success else 'Failed'}")
                print(f"  Final distance to goal: {info['distance_to_goal']:.1f}m")
                break

        trajectory = np.array(trajectory)
        actions = np.array(actions)
        rewards = np.array(rewards)
        velocities = np.array(velocities)

        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 15))

        # 3D trajectory 3D轨迹图
        ax1 = fig.add_subplot(231, projection='3d')   # 231：总共有2行3列，第一个子图。在2行3列的网格布局中创建第1个子图
        ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', linewidth=2, label='Trajectory') # b-：蓝色实线格式，linewidth=2：线宽为2磅，使轨迹更醒目。label='Trajectory'：设置图例标签为"Trajectory
        ax1.scatter(*self.env.start_position, color='green', s=100, label='Start')
        ax1.scatter(*self.env.target_position, color='red', s=100, label='Target')

        # Plot obstacles if any
        if hasattr(self.env, 'obstacle_gen') and self.env.obstacle_gen.obstacles:
            max_show = min(len(self.env.obstacle_gen.obstacles), 30)  # 最多显示30个
            for obs in self.env.obstacle_gen.obstacles[:max_show]:
                # Draw cylinder for obstacle 绘制圆柱体来表示障碍物
                theta = np.linspace(0, 2 * np.pi, 20)
                x_cyl = obs['position'][0] + obs['radius'] * np.cos(theta)
                y_cyl = obs['position'][1] + obs['radius'] * np.sin(theta)
                z_bottom = np.zeros_like(x_cyl)
                z_top = np.full_like(x_cyl, obs['height'])
                ax1.plot(x_cyl, y_cyl, z_bottom, 'r-', alpha=0.3)  # 绘制底部圆环（红色实线）
                ax1.plot(x_cyl, y_cyl, z_top, 'r-', alpha=0.3)    # 绘制顶部圆环（红色实线）

        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectory')
        ax1.legend()  # 显示图例

        # XY plane view   XY平面俯视图，用于从顶部视角分析智能体在水平面上的运动轨迹和避障行为
        ax2 = fig.add_subplot(232)
        ax2.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Path')
        ax2.scatter(self.env.start_position[0], self.env.start_position[1],
                    color='green', s=100, label='Start', zorder=5)
        ax2.scatter(self.env.target_position[0], self.env.target_position[1],
                    color='red', s=100, label='Target', zorder=5)

        # Plot obstacles in XY plane
        if hasattr(self.env, 'obstacle_gen') and self.env.obstacle_gen.obstacles:
            for obs in self.env.obstacle_gen.obstacles:
                circle = plt.Circle((obs['position'][0], obs['position'][1]),
                                    obs['radius'], color='red', alpha=0.3)
                ax2.add_patch(circle)

        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('XY Plane View')
        ax2.legend()
        ax2.axis('equal')
        ax2.grid(True, alpha=0.3)  # 显示网格线，透明度为0.3

        # Altitude profile
        ax3 = fig.add_subplot(233)
        time_steps = np.arange(len(trajectory))
        ax3.plot(time_steps, trajectory[:, 2], 'b-', linewidth=2, label='Altitude')
        ax3.axhline(y=self.env.target_position[2], color='r', linestyle='--', alpha=0.7, label='Target Altitude')
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Altitude (m)')
        ax3.set_title('Altitude Profile')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Control inputs
        ax4 = fig.add_subplot(234)
        time_steps = np.arange(len(trajectory))  # 确保 time_steps 在这里定义

        ax4.plot(time_steps[:len(actions)], actions[:, 0], label='T_Lift (a0)', color='blue', alpha=0.8)  # 垂直推力 (a0)
        ax4.plot(time_steps[:len(actions)], actions[:, 1], label='T_Cruise (a1)', color='red', alpha=0.8)  # 巡航推力 (a1)
        ax4.plot(time_steps[:len(actions)], actions[:, 2], label='Alpha (a2)', color='green', alpha=0.8)  # 攻角 (a2)
        ax4.plot(time_steps[:len(actions)], actions[:, 3], label='Phi (a3)', color='orange', alpha=0.8)  # 滚转角 (a3)

        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('Control Input (normalized)')
        ax4.set_title('Control Inputs')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Reward progression
        ax5 = fig.add_subplot(235)
        cumulative_rewards = np.cumsum(rewards)  # cumsum：积累求和
        ax5.plot(time_steps[:len(rewards)], rewards, 'g-', alpha=0.6, label='Step Reward')
        ax5.plot(time_steps[:len(rewards)], cumulative_rewards, 'r-', linewidth=2, label='Cumulative Reward')
        ax5.set_xlabel('Time Steps')
        ax5.set_ylabel('Reward')
        ax5.set_title('Reward Progression')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Velocity profile
        ax6 = fig.add_subplot(236)
        speeds = np.linalg.norm(velocities, axis=1)   # 沿着行的方向计算L2范数，speed = √(vₓ² + vᵧ² + v_z²)
        ax6.plot(time_steps[:len(velocities)], speeds, 'purple', linewidth=2, label='Speed')
        ax6.plot(time_steps[:len(velocities)], velocities[:, 0], label='Vx', alpha=0.7)
        ax6.plot(time_steps[:len(velocities)], velocities[:, 1], label='Vy', alpha=0.7)
        ax6.plot(time_steps[:len(velocities)], velocities[:, 2], label='Vz', alpha=0.7)
        ax6.set_xlabel('Time Steps')
        ax6.set_ylabel('Velocity (m/s)')
        ax6.set_title('Velocity Profile')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()   # 自动调整布局

        # Save visualization
        viz_path = os.path.join(self.output_dir, f'trajectory_seed_{episode_seed}.png')
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"Trajectory visualization saved to: {viz_path}")

        # Show plot
        plt.show()

        return trajectory, actions, rewards

    def generate_multiple_trajectories(self, num_trajectories: int = 5):     # 默认生成 5 个轨迹可视化
        '''Generate and save multiple trajectory visualizations'''
        print(f"\nGenerating {num_trajectories} trajectory visualizations...")

        for i in range(num_trajectories):
            seed = 100 + i * 10
            print(f"  Generating trajectory {i + 1}/{num_trajectories} (seed={seed})")

            try:
                self.visualize_trajectory(episode_seed=seed)
                plt.close('all')  # Close plots to save memory
            except Exception as e:
                print(f"    Warning: Failed to generate trajectory {i + 1}: {e}")

        print(f"All trajectory visualizations saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained eVTOL agent')
    parser.add_argument('--checkpoint', type=str, required=True,            # --checkpoint：模型检查点的路径，required=True必须需要的
                        help='Path to model checkpoint')
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--stage', type=str, default='COMPLEX_NAV',
                        choices=['BASIC_FLIGHT', 'SIMPLE_NAV', 'COMPLEX_NAV', 'GENERALIZATION'],
                        help='Evaluation stage')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Output directory')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate trajectory visualizations')
    parser.add_argument('--num_viz', type=int, default=5,
                        help='Number of trajectory visualizations to generate')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cpu/cuda/auto)')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("=" * 60)
    print("eVTOL Agent Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Stage: {args.stage}")
    print(f"Device: {device}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 60)

    # Initialize environment 初始化环境
    stage = TrainingStage[args.stage]
    # Load config from config dictionary
    config = None
    config_path = 'config/default.yaml'
    if os.path.exists(config_path):
        import yaml
        with open(config_path,'r') as f:
            config = yaml.safe_load(f)
        print(f'Using config from: {config_path}')
    else:
        print(f'Config file not found, using default paameters')

    env = EVTOLEnvironment(stage,config)

    print(f"Environment initialized for stage: {stage.name}")
    print(f"Area size: {env.area_size}")
    print(f"Target radius: {env.target_radius}")     #

    # Initialize agent 初始化智能体
    agent = TD3Agent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        device=device
    )

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return

    agent.load(args.checkpoint)
    print(f"✓ Model loaded from: {args.checkpoint}")

    # Initialize evaluator
    evaluator = Evaluator(agent, env, args.output_dir, config)

    # Run performance evaluation
    metrics = evaluator.evaluate_performance(args.num_episodes)

    # Generate visualizations if requested
    if args.visualize:
        evaluator.generate_multiple_trajectories(args.num_viz)

    print(f"\n✓ Evaluation complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()