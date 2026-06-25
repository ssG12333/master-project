import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# 导入你的模块
from envs.evtol_env import EVTOLEnvironment
from envs.obstacles import TrainingStage
from agents.td3 import TD3Agent
from utils.replay_buffer import ReplayBuffer


class QuickTestTrainer:
    '''快速测试训练器 - 专门用于验证修改效果'''

    def __init__(self):
        # 使用你修改后的配置
        self.config = {
            'training': {
                'max_steps': 100000,  # 10万步快速测试
                'warmup_steps': 2000,  # 减少预热步数
                'batch_size': 128,  # 减小批大小
                'buffer_size': 50000,  # 减小缓冲区
                'save_freq': 20,  # 更频繁保存
            },
            'agent_params': {
                'lr_actor': 3e-5,  # 使用建议的学习率
                'lr_critic': 3e-5,
                'gamma': 0.99,
                'tau': 0.005,
                'policy_noise': 0.2,
                'noise_clip': 0.5,
                'policy_delay': 2
            },
            'environment': {
                'target_radius': 15.0,
                'safety_distance_threshold': 25.0,
                'collision_distance_threshold': 10.0
            }
        }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 初始化环境
        self.env = EVTOLEnvironment(TrainingStage.BASIC_FLIGHT, self.config)

        # 初始化智能体
        agent_params = self.config['agent_params'].copy()
        lr_actor = float(agent_params.pop('lr_actor'))
        lr_critic = float(agent_params.pop('lr_critic'))

        self.agent = TD3Agent(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim,
            max_action=1.0,
            device=self.device,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            **agent_params
        )

        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(
            max_size=self.config['training']['buffer_size'],
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim
        )

        # 创建输出目录
        self.output_dir = "quick_test_results"
        os.makedirs(self.output_dir, exist_ok=True)

        # TensorBoard记录
        self.writer = SummaryWriter(log_dir=os.path.join(self.output_dir, 'tensorboard'))

        # 训练记录
        self.episode_rewards = []
        self.critic_losses = []
        self.episode_steps_list = []

        print("快速测试训练器初始化完成！")
        print(f"状态维度: {self.env.state_dim}, 动作维度: {self.env.action_dim}")
        print(f"总训练步数: {self.config['training']['max_steps']}")

    def run_episode(self):
        '''运行单个episode'''
        state = self.env.reset()
        episode_reward = 0
        episode_steps = 0
        actions = []

        max_steps = 800  # BASIC_FLIGHT最大步数

        while episode_steps < max_steps:
            # 选择动作
            if len(self.replay_buffer) < self.config['training']['warmup_steps']:
                action = np.random.uniform(-1, 1, self.env.action_dim)
            else:
                action = self.agent.select_action(state, evaluate=False)

            actions.append(action)

            # 执行动作
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # 存储经验
            self.replay_buffer.add(state, action, reward, next_state, done)

            episode_reward += reward
            episode_steps += 1
            state = next_state

            if done:
                break

        return episode_reward, episode_steps, actions

    def train(self):
        '''运行训练'''
        print("\n开始快速测试训练...")
        print("=" * 50)

        total_steps = 0
        episode = 0

        warmup_steps = self.config['training']['warmup_steps']
        max_steps = self.config['training']['max_steps']

        # 训练循环
        while total_steps < max_steps:
            # 运行一个episode
            episode_reward, episode_steps, actions = self.run_episode()

            total_steps += episode_steps
            episode += 1

            # 记录episode结果
            self.episode_rewards.append(episode_reward)
            self.episode_steps_list.append(episode_steps)

            # 训练智能体
            training_metrics = {}
            if len(self.replay_buffer) > warmup_steps:
                training_metrics = self.agent.train(self.replay_buffer, self.config['training']['batch_size'])

                if training_metrics and 'critic_loss' in training_metrics:
                    self.critic_losses.append(training_metrics['critic_loss'])

            # 记录到TensorBoard
            self.writer.add_scalar('Episode/Reward', episode_reward, episode)
            self.writer.add_scalar('Episode/Steps', episode_steps, episode)
            if training_metrics and 'critic_loss' in training_metrics:
                self.writer.add_scalar('Training/Critic_Loss', training_metrics['critic_loss'], episode)

            # 每5个episode打印进度
            if episode % 5 == 0:
                avg_reward = np.mean(self.episode_rewards[-5:])
                current_critic_loss = training_metrics.get('critic_loss', 0) if training_metrics else 0

                print(f"Episode {episode:3d} | "
                      f"奖励: {episode_reward:8.1f} | "
                      f"平均奖励: {avg_reward:8.1f} | "
                      f"步数: {episode_steps:3d} | "
                      f"Critic损失: {current_critic_loss:10.2f} | "
                      f"总步数: {total_steps:5d}")

            # 每20个episode保存检查点
            if episode % 20 == 0:
                self.save_checkpoint(episode)
                self.plot_progress()

            if total_steps >= max_steps:
                break

        print("\n训练完成！")
        self.final_analysis()

    def plot_progress(self):
        '''绘制训练进度'''
        if len(self.episode_rewards) < 2:
            return

        plt.figure(figsize=(12, 4))

        # 奖励曲线
        plt.subplot(1, 2, 1)
        episodes = range(len(self.episode_rewards))
        plt.plot(episodes, self.episode_rewards, 'b-', alpha=0.7, linewidth=1)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True, alpha=0.3)

        # Critic损失曲线（如果有数据）
        if self.critic_losses:
            plt.subplot(1, 2, 2)
            # 使用移动平均平滑损失曲线
            window = max(1, len(self.critic_losses) // 10)
            if window > 1:
                smoothed_losses = np.convolve(self.critic_losses, np.ones(window) / window, mode='valid')
                loss_episodes = range(window - 1, len(self.critic_losses))
                plt.plot(loss_episodes, smoothed_losses, 'r-', alpha=0.7, linewidth=1)
            else:
                plt.plot(range(len(self.critic_losses)), self.critic_losses, 'r-', alpha=0.7, linewidth=1)
            plt.title('Critic Loss (Smoothed)')
            plt.xlabel('Training Step')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_progress.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def final_analysis(self):
        '''最终分析'''
        print("\n" + "=" * 60)
        print("快速测试最终分析")
        print("=" * 60)

        if not self.episode_rewards:
            print("没有训练数据！")
            return

        # 基本统计
        final_reward = self.episode_rewards[-1]
        avg_reward_last_10 = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else final_reward
        best_reward = max(self.episode_rewards)
        improvement = final_reward - self.episode_rewards[0] if self.episode_rewards else 0

        print(f"📊 性能指标:")
        print(f"  最终奖励: {final_reward:8.1f}")
        print(f"  最近10轮平均奖励: {avg_reward_last_10:8.1f}")
        print(f"  历史最佳奖励: {best_reward:8.1f}")
        print(f"  奖励改善: {improvement:8.1f}")
        print(f"  总episode数: {len(self.episode_rewards)}")
        print(f"  总训练步数: {sum(self.episode_steps_list)}")

        if self.critic_losses:
            final_loss = self.critic_losses[-1]
            avg_loss_last_10 = np.mean(self.critic_losses[-10:]) if len(self.critic_losses) >= 10 else final_loss
            print(f"  Critic损失: {final_loss:10.2f}")
            print(f"  最近Critic损失: {avg_loss_last_10:10.2f}")

        # 诊断建议
        print(f"\n💡 诊断建议:")
        if final_reward > -100:
            print("  ✅ 优秀！奖励已经接近正值，继续训练")
        elif final_reward > -500:
            print("  ⚠️  良好！奖励在改善，但还有优化空间")
        elif final_reward > -1000:
            print("  🔄 一般！需要进一步调整奖励函数")
        else:
            print("  ❌ 需要大幅调整！建议检查奖励函数和网络架构")

        if self.critic_losses and final_loss > 10000:
            print("  ⚠️  Critic损失仍然偏高，建议降低学习率")
        elif self.critic_losses and final_loss < 1000:
            print("  ✅ Critic损失在合理范围内")

        # 保存最终图表
        self.plot_progress()
        print(f"\n📈 训练图表已保存到: {self.output_dir}/training_progress.png")
        print(f"📊 TensorBoard日志在: {self.output_dir}/tensorboard")
        print(f"💾 模型检查点在: {self.output_dir}/checkpoints")

    def save_checkpoint(self, episode_num):
        '''保存检查点'''
        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        path = os.path.join(checkpoint_dir, f'checkpoint_ep{episode_num}.pt')
        self.agent.save(path)

    def close(self):
        '''清理资源'''
        self.writer.close()


def main():
    '''主函数'''
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    print("🚀 eVTOL快速测试训练")
    print("基于修改后的配置验证训练效果")
    print("=" * 50)

    try:
        # 创建训练器
        trainer = QuickTestTrainer()

        # 运行训练
        trainer.train()

        # 最终分析
        trainer.final_analysis()

    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保资源被清理
        if 'trainer' in locals():
            trainer.close()

    print("\n快速测试完成！")


if __name__ == "__main__":
    main()