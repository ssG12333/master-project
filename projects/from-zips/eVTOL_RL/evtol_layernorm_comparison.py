import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import json
from datetime import datetime

# 导入你的现有模块
from envs.evtol_env import EVTOLEnvironment
from envs.obstacles import TrainingStage
from agents.td3 import TD3Agent
from utils.replay_buffer import ReplayBuffer
from networks.critic import CriticNetwork


class ActorNoLayerNorm(nn.Module):
    '''不带LayerNorm的Actor网络 - 用于对比测试'''

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = None, max_action: float = 1.0):
        super().__init__()
        self.max_action = max_action

        if hidden_dims is None:
            hidden_dims = [256, 256, 128]

        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())  # 没有LayerNorm
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))
        self.net = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * torch.tanh(self.net(state))


class PracticalLayerNormComparison:
    '''实际训练环境下的LayerNorm对比测试'''

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 创建测试目录
        self.test_dir = f"layernorm_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.test_dir, exist_ok=True)

        print(f"测试目录: {self.test_dir}")

        # 保存配置
        with open(os.path.join(self.test_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    def create_agent_with_custom_actor(self, config, use_layernorm: bool):
        '''创建使用自定义Actor的TD3智能体'''

        # 创建环境获取维度
        env = EVTOLEnvironment(TrainingStage.BASIC_FLIGHT, config)
        state_dim = env.state_dim
        action_dim = env.action_dim

        # 复制agent参数
        agent_params = config['agent_params'].copy()
        lr_actor = float(agent_params.pop('lr_actor'))
        lr_critic = float(agent_params.pop('lr_critic'))

        # 创建自定义的TD3Agent
        class CustomTD3Agent(TD3Agent):
            def __init__(self, state_dim, action_dim, max_action, device, lr_actor, lr_critic, use_layernorm, **kwargs):
                super().__init__(state_dim, action_dim, max_action, device, lr_actor, lr_critic, **kwargs)

                # 替换Actor网络
                if use_layernorm:
                    # 使用原始带LayerNorm的Actor
                    from networks.actor import ActorNetwork
                    self.actor = ActorNetwork(state_dim, action_dim, max_action=max_action).to(device)
                    self.actor_target = ActorNetwork(state_dim, action_dim, max_action=max_action).to(device)
                else:
                    # 使用不带LayerNorm的Actor
                    self.actor = ActorNoLayerNorm(state_dim, action_dim, max_action=max_action).to(device)
                    self.actor_target = ActorNoLayerNorm(state_dim, action_dim, max_action=max_action).to(device)

                self.actor_target.load_state_dict(self.actor.state_dict())
                self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

                print(f"创建{'带' if use_layernorm else '不带'}LayerNorm的智能体")

        agent = CustomTD3Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=1.0,
            device=self.device,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            use_layernorm=use_layernorm,
            **agent_params
        )

        return agent, env

    def run_training_comparison(self, total_steps: int = 50000):
        '''运行训练对比测试'''
        print(f"\n开始实际训练对比测试，总步数: {total_steps}")

        # 创建两个训练器
        trainer_with_ln = SingleTrainer(self.config, use_layernorm=True,
                                        log_dir=os.path.join(self.test_dir, 'with_layernorm'))
        trainer_without_ln = SingleTrainer(self.config, use_layernorm=False,
                                           log_dir=os.path.join(self.test_dir, 'without_layernorm'))

        # 运行训练
        print("\n训练带LayerNorm的智能体...")
        results_with_ln = trainer_with_ln.train(total_steps)

        print("\n训练不带LayerNorm的智能体...")
        results_without_ln = trainer_without_ln.train(total_steps)

        # 保存结果
        comparison_results = {
            'with_layernorm': results_with_ln,
            'without_layernorm': results_without_ln,
            'training_steps': total_steps,
            'timestamp': datetime.now().isoformat()
        }

        with open(os.path.join(self.test_dir, 'training_comparison_results.json'), 'w') as f:
            json.dump(comparison_results, f, indent=2)

        return comparison_results

    def analyze_training_results(self, results):
        '''分析训练结果'''
        print("\n" + "=" * 80)
        print("实际训练对比结果分析")
        print("=" * 80)

        ln_results = results['with_layernorm']
        no_ln_results = results['without_layernorm']

        # 最终性能指标
        ln_final_reward = ln_results['episode_rewards'][-1] if ln_results['episode_rewards'] else 0
        no_ln_final_reward = no_ln_results['episode_rewards'][-1] if no_ln_results['episode_rewards'] else 0

        ln_avg_reward = np.mean(ln_results['episode_rewards'][-10:]) if len(
            ln_results['episode_rewards']) >= 10 else ln_final_reward
        no_ln_avg_reward = np.mean(no_ln_results['episode_rewards'][-10:]) if len(
            no_ln_results['episode_rewards']) >= 10 else no_ln_final_reward

        print("\n📊 训练性能对比:")
        print(f"带LayerNorm:")
        print(f"  最终奖励: {ln_final_reward:.1f}")
        print(f"  最近10轮平均奖励: {ln_avg_reward:.1f}")
        print(f"  总训练步数: {ln_results['total_steps']}")
        print(f"  总episode数: {len(ln_results['episode_rewards'])}")
        print(f"  Critic损失: {ln_results['final_critic_loss']:.4f}")

        print(f"\n不带LayerNorm:")
        print(f"  最终奖励: {no_ln_final_reward:.1f}")
        print(f"  最近10轮平均奖励: {no_ln_avg_reward:.1f}")
        print(f"  总训练步数: {no_ln_results['total_steps']}")
        print(f"  总episode数: {len(no_ln_results['episode_rewards'])}")
        print(f"  Critic损失: {no_ln_results['final_critic_loss']:.4f}")

        # 奖励改善程度
        if len(ln_results['episode_rewards']) > 10 and len(no_ln_results['episode_rewards']) > 10:
            ln_improvement = ln_results['episode_rewards'][-1] - ln_results['episode_rewards'][0]
            no_ln_improvement = no_ln_results['episode_rewards'][-1] - no_ln_results['episode_rewards'][0]

            print(f"\n📈 奖励改善程度:")
            print(f"  带LayerNorm: {ln_improvement:.1f}")
            print(f"  不带LayerNorm: {no_ln_improvement:.1f}")

            if no_ln_improvement > ln_improvement:
                print("✅ 不带LayerNorm的学习效果更好！")
            elif ln_improvement > no_ln_improvement:
                print("⚠️  带LayerNorm的学习效果更好")
            else:
                print("➖ 两者效果相似")

    def plot_training_comparison(self, results):
        '''绘制训练对比图表'''
        ln_results = results['with_layernorm']
        no_ln_results = results['without_layernorm']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 修复1: 确保数据长度一致
        min_episodes = min(len(ln_results['episode_rewards']), len(no_ln_results['episode_rewards']))

        # 奖励曲线对比
        episodes = range(min_episodes)

        axes[0, 0].plot(episodes, ln_results['episode_rewards'][:min_episodes],
                        label='With LayerNorm', alpha=0.7, linewidth=2, color='blue')
        axes[0, 0].plot(episodes, no_ln_results['episode_rewards'][:min_episodes],
                        label='Without LayerNorm', alpha=0.7, linewidth=2, color='red')
        axes[0, 0].set_title('Training Rewards Comparison')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Critic损失对比
        if ln_results['critic_losses'] and no_ln_results['critic_losses']:
            # 修复2: 确保损失数据长度一致
            min_loss_steps = min(len(ln_results['critic_losses']), len(no_ln_results['critic_losses']))
            steps = range(min_loss_steps)

            # 使用移动平均平滑损失曲线
            window = max(1, min_loss_steps // 50)
            ln_loss_smooth = self.moving_average(ln_results['critic_losses'][:min_loss_steps], window)
            no_ln_loss_smooth = self.moving_average(no_ln_results['critic_losses'][:min_loss_steps], window)

            # 修复3: 确保平滑后的数据长度一致
            min_smooth_length = min(len(ln_loss_smooth), len(no_ln_loss_smooth))
            smooth_steps = range(min_smooth_length)

            axes[0, 1].plot(smooth_steps, ln_loss_smooth[:min_smooth_length],
                            label='With LayerNorm', alpha=0.7, linewidth=2, color='blue')
            axes[0, 1].plot(smooth_steps, no_ln_loss_smooth[:min_smooth_length],
                            label='Without LayerNorm', alpha=0.7, linewidth=2, color='red')
            axes[0, 1].set_title('Critic Loss Comparison (Smoothed)')
            axes[0, 1].set_xlabel('Training Step')
            axes[0, 1].set_ylabel('Critic Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # 动作统计对比
        if ln_results['action_stats'] and no_ln_results['action_stats']:
            # 修复4: 确保动作统计数据长度一致
            min_action_episodes = min(len(ln_results['action_stats']), len(no_ln_results['action_stats']))
            action_episodes = range(min_action_episodes)

            ln_action_norms = [stat['avg_norm'] for stat in ln_results['action_stats'][:min_action_episodes]]
            no_ln_action_norms = [stat['avg_norm'] for stat in no_ln_results['action_stats'][:min_action_episodes]]

            axes[1, 0].plot(action_episodes, ln_action_norms,
                            label='With LayerNorm', alpha=0.7, linewidth=2, color='blue')
            axes[1, 0].plot(action_episodes, no_ln_action_norms,
                            label='Without LayerNorm', alpha=0.7, linewidth=2, color='red')
            axes[1, 0].set_title('Action Magnitude Comparison')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Action Norm')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # 成功率对比（如果有验证数据）
        if 'validation_success' in ln_results and 'validation_success' in no_ln_results:
            if ln_results['validation_success'] and no_ln_results['validation_success']:
                # 修复5: 确保验证数据长度一致
                min_validations = min(len(ln_results['validation_success']), len(no_ln_results['validation_success']))
                validations = range(min_validations)

                axes[1, 1].plot(validations, ln_results['validation_success'][:min_validations],
                                label='With LayerNorm', alpha=0.7, linewidth=2, color='blue', marker='o')
                axes[1, 1].plot(validations, no_ln_results['validation_success'][:min_validations],
                                label='Without LayerNorm', alpha=0.7, linewidth=2, color='red', marker='s')
                axes[1, 1].set_title('Validation Success Rate Comparison')
                axes[1, 1].set_xlabel('Validation Cycle')
                axes[1, 1].set_ylabel('Success Rate')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            else:
                # 如果没有验证数据，显示提示信息
                axes[1, 1].text(0.5, 0.5, 'No validation data available',
                                horizontalalignment='center', verticalalignment='center',
                                transform=axes[1, 1].transAxes, fontsize=12)
                axes[1, 1].set_title('Validation Success Rate Comparison')
        else:
            # 如果没有验证数据，显示提示信息
            axes[1, 1].text(0.5, 0.5, 'No validation data available',
                            horizontalalignment='center', verticalalignment='center',
                            transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].set_title('Validation Success Rate Comparison')

        plt.tight_layout()
        plt.savefig(os.path.join(self.test_dir, 'training_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def moving_average(self, data, window_size):
        '''计算移动平均'''
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    def run_complete_test(self, total_steps: int = 50000):
        '''运行完整的实际训练对比测试'''
        print("开始实际训练环境下的LayerNorm对比测试...")

        results = self.run_training_comparison(total_steps)
        self.analyze_training_results(results)
        self.plot_training_comparison(results)

        print(f"\n测试完成！")
        print(f"所有结果保存在: {self.test_dir}")
        print("查看 training_comparison.png 获取可视化对比结果")


class SingleTrainer:
    '''单个训练器，用于对比测试'''

    def __init__(self, config: Dict, use_layernorm: bool, log_dir: str):
        self.config = config
        self.use_layernorm = use_layernorm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_dir = log_dir

        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        # 创建智能体和环境
        self.comparison = PracticalLayerNormComparison(config)
        self.agent, self.env = self.comparison.create_agent_with_custom_actor(config, use_layernorm)

        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(
            max_size=config['training']['buffer_size'],
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim
        )

        # 训练结果记录
        self.results = {
            'episode_rewards': [],
            'episode_steps': [],
            'critic_losses': [],
            'actor_losses': [],
            'action_stats': [],
            'validation_success': [],
            'total_steps': 0,
            'final_critic_loss': 0
        }

        print(f"初始化{'带' if use_layernorm else '不带'}LayerNorm的训练器")

    def train(self, total_steps: int):
        '''训练智能体'''
        steps = 0
        episode = 0

        warmup_steps = self.config['training']['warmup_steps']
        train_steps_per_cycle = min(1000, total_steps // 10)  # 简化训练周期

        while steps < total_steps:
            # 运行一个episode
            episode_reward, episode_steps, action_stats = self.run_episode()

            steps += episode_steps
            episode += 1

            # 记录episode结果
            self.results['episode_rewards'].append(episode_reward)
            self.results['episode_steps'].append(episode_steps)
            self.results['action_stats'].append(action_stats)

            # 训练（简化版）
            if len(self.replay_buffer) > warmup_steps:
                training_metrics = self.agent.train(self.replay_buffer, self.config['training']['batch_size'])

                if training_metrics:
                    self.results['critic_losses'].append(training_metrics.get('critic_loss', 0))
                    if 'actor_loss' in training_metrics:
                        self.results['actor_losses'].append(training_metrics['actor_loss'])

            # 记录到TensorBoard
            self.writer.add_scalar('Episode/Reward', episode_reward, episode)
            self.writer.add_scalar('Episode/Steps', episode_steps, episode)

            # 每10个episode打印进度
            if episode % 10 == 0:
                avg_reward = np.mean(self.results['episode_rewards'][-10:])
                print(f"Episode {episode}: 平均奖励={avg_reward:.1f}, 总步数={steps}")

            if steps >= total_steps:
                break

        self.results['total_steps'] = steps
        if self.results['critic_losses']:
            self.results['final_critic_loss'] = self.results['critic_losses'][-1]

        self.writer.close()
        return self.results

    def run_episode(self):
        '''运行单个episode'''
        state = self.env.reset()
        episode_reward = 0
        episode_steps = 0
        actions = []

        max_steps = 800  # BASIC_FLIGHT阶段

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

        # 计算动作统计
        if actions:
            action_array = np.array(actions)
            action_stats = {
                'avg_norm': np.mean(np.linalg.norm(action_array, axis=1)),
                'avg_std': np.mean(np.std(action_array, axis=0)),
                'diversity': np.std(np.linalg.norm(action_array, axis=1))
            }
        else:
            action_stats = {'avg_norm': 0, 'avg_std': 0, 'diversity': 0}

        return episode_reward, episode_steps, action_stats


def main():
    '''主函数'''

    # 使用你提供的配置
    config = {
        'training': {
            'max_steps': 1000000,
            'max_episode_steps': 1500,
            'warmup_steps': 10000,
            'batch_size': 256,
            'buffer_size': 1000000,
            'save_freq': 100,
            'train_steps_per_cycle': 10000,
            'validation_episodes_per_cycle': 100,
            'use_prioritized_replay': True,
            'per_alpha': 0.6,
            'per_beta': 0.4,
            'per_beta_increment': 0.001
        },
        'agent_params': {
            'lr_actor': 1e-4,
            'lr_critic': 1e-4,
            'gamma': 0.99,
            'tau': 0.005,
            'policy_noise': 0.2,
            'noise_clip': 0.5,
            'policy_delay': 2
        },
        'stage_progression': {
            'BASIC_FLIGHT': 0.9,
            'SIMPLE_NAV': 0.9,
            'COMPLEX_NAV': 0.7,
            'GENERALIZATION': None
        }
    }

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 创建对比测试器
    tester = PracticalLayerNormComparison(config)

    # 运行完整测试（建议先用50000步快速测试）
    tester.run_complete_test(total_steps=50000)


if __name__ == "__main__":
    main()