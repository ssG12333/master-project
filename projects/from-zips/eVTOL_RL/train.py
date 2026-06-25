import argparse
import os
import numpy as np
import torch
import yaml

from trainers import CurriculumTrainer


def main():
    parser = argparse.ArgumentParser(description='Train eVTOL Agent with Prioritized Experience Replay')

    # Basic arguments   基本参数
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='output/evtol_td3_per',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cpu/cuda/auto)')

    # Override config arguments  配置覆盖参数，这些参数用于覆盖文件中的设置，default=None表示默认不覆盖
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr_actor', type=float, default=None)
    parser.add_argument('--lr_critic', type=float, default=None)
    parser.add_argument('--use_per', action='store_true', default=None,  # action='store_true'：表示这是一个开关，不需要跟值。有参数就是 True，没有就是 False
                        help='Enable Prioritized Experience Replay')
    parser.add_argument('--no_per', action='store_true', default=None,
                        help='Disable Prioritized Experience Replay')
    parser.add_argument('--per_alpha', type=float, default=None,
                        help='PER alpha parameter (0=uniform, 1=greedy)')
    parser.add_argument('--per_beta', type=float, default=None,
                        help='PER beta parameter (importance sampling)')

    args = parser.parse_args()   # 解析命令行参数，将结果存储在 args 变量中

    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)   # 如果配置文件（defult.yaml，中等优先级）存在就加载，否则使用内置的默认配置
    else:
        # Default configuration   使用下面的默认配置（最低优先级）
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
                'lr_actor': 3e-4,
                'lr_critic': 3e-4,
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

    # 扁平化配置以保持向后兼容，把嵌套的配置结构扁平化，让一些常用的配置项可以直接访问
    flat_config = {
        'output_dir': args.output_dir,
        **config.get('training', {}),   # **：把一个字典的所有键值对展开到另一个字典中
        'agent_params': config.get('agent_params', {}),
        'stage_progression': config.get('stage_progression', {}),
        'environment': config.get('environment',{})
    }

    # Override config with command line arguments  命令行参数覆盖配置,如果命令行提供了这些参数，就覆盖配置文件中的值。（命令行最高优先级）
    if args.max_steps is not None:
        flat_config['max_steps'] = args.max_steps
    if args.batch_size is not None:
        flat_config['batch_size'] = args.batch_size
    if args.lr_actor is not None:
        flat_config['agent_params']['lr_actor'] = args.lr_actor
    if args.lr_critic is not None:
        flat_config['agent_params']['lr_critic'] = args.lr_critic

    # PER相关参数
    if args.use_per:
        flat_config['use_prioritized_replay'] = True
    if args.no_per:
        flat_config['use_prioritized_replay'] = False
    if args.per_alpha is not None:
        flat_config['per_alpha'] = args.per_alpha
    if args.per_beta is not None:
        flat_config['per_beta'] = args.per_beta

    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Set random seeds
    np.random.seed(args.seed)                 # 设置NumPy的随机种子
    torch.manual_seed(args.seed)              # 设置PyTorch的随机种子（CPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)     # 设置PyTorch的随机种子（GPU）

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as f:
        # 使用 flat_config 确保保存了所有命令行覆盖后的参数
        # default_flow_style=False 确保 YAML 文件以清晰的多行格式保存
        yaml.dump(flat_config, f, default_flow_style=False)

    # Initialize trainer
    print("=" * 50)
    print("Starting eVTOL TD3 Training with Curriculum Learning & PER")
    print(f"Output Directory: {args.output_dir}")
    print(f"Device: {device}")
    print(
        f"Training Cycles: {flat_config['train_steps_per_cycle']} steps + {flat_config['validation_episodes_per_cycle']} validation episodes")

    # PER信息
    if flat_config.get('use_prioritized_replay', True):
        print(
            f"Prioritized Experience Replay: α={flat_config.get('per_alpha', 0.6)}, β={flat_config.get('per_beta', 0.4)}")
    else:
        print("Using Standard Experience Replay")

    print("=" * 50)

    trainer = CurriculumTrainer(flat_config)

    # Start training
    trainer.train()

    print("\nTraining Complete!")


if __name__ == "__main__":
    main()