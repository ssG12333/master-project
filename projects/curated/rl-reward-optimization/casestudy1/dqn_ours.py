# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple
from utils.reward_machine import RewardFunction
from tqdm import tqdm


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""

    reward_frequency: int = 1000
    """the frequency of reward updates"""
    hidden_dim: int = 256
    """the number of hidden units in the network"""
    encode_dim: int = 64
    """the dimension of the encoded representation"""
    activate_function: str = "tanh"
    """the activation function used in the network"""
    last_activate_function: str = "None"
    """the activation function used in the last layer"""
    reward_lr: float = 1e-4
    """the learning rate for the reward model"""
    reward_buffer_size: int = 100
    """the size of the reward-specific buffer"""

    reward_type: str = "sparse"
    """reward type: sparse (env native), rmbo, ours"""
    use_advantage: bool = False
    """whether to use advantage function instead of Q value"""
    use_adamw: bool = False
    """whether to use AdamW optimizer"""
    n_samples: int = 10
    """number of sampled actions for advantage estimation"""


def make_env(env_id, seed, idx, capture_video, run_name):
    """
    创建环境函数
    
    Args:
        env_id: 环境ID
        seed: 随机种子
        idx: 环境索引
        capture_video: 是否捕获视频
        run_name: 运行名称
    """
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk


class QNetwork(nn.Module):
    """
    Q网络，用于DQN算法
    """
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        """前向传播"""
        return self.network(x)

    def get_action_prob_from_mu(self, mu, n):
        """
        从策略分布中采样动作并计算对数概率
        
        Args:
            mu: 策略分布参数
            n: 采样数量
        """
        dist = torch.distributions.Categorical(probs=F.softmax(mu, dim=-1))
        actions = dist.sample((n,))
        log_probs = dist.log_prob(actions)
        actions_bs = actions.transpose(0, 1)
        log_probs_bs = log_probs.transpose(0, 1)
        return actions_bs, log_probs_bs

    def get_q_values(self, x):
        """获取Q值"""
        return self.forward(x)


class ValueNetwork(nn.Module):
    """
    值网络，用于Advantage计算
    """
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 1),
        )

    def forward(self, x):
        """前向传播"""
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """
    线性调度函数，用于epsilon衰减
    
    Args:
        start_e: 起始epsilon
        end_e: 结束epsilon
        duration: 持续时间
        t: 当前时间步
    """
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    
    # 解析参数
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    
    # 设置运行名称
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    # 初始化wandb（如果需要）
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    # 初始化TensorBoard
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # 创建环境
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # 初始化网络
    q_network = QNetwork(envs).to(device)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    # 初始化值网络（用于Advantage计算）
    value_network = ValueNetwork(envs).to(device)
    value_optimizer = optim.Adam(value_network.parameters(), lr=args.learning_rate) if args.use_advantage else None

    # 选择优化器（Adam或AdamW）
    if args.use_adamw:
        optimizer = optim.AdamW(q_network.parameters(), lr=args.learning_rate, weight_decay=0.01)
        print("Using AdamW optimizer with weight decay 0.01")
    else:
        optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
        print("Using Adam optimizer")

    # 初始化经验回放缓冲区
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    # 初始化奖励函数
    reward_function = RewardFunction(env=envs, args=args, device=device)
    Transition = namedtuple('Transition', ['state', 'action', 'reward', 'log_probs', 'mu', 'overline_V', 'advantage'])
    epidata = []
    start_time = time.time()

    # 重置环境
    obs, _ = envs.reset(seed=args.seed)
    
    # 主训练循环
    print(f"Starting training for {args.total_timesteps} timesteps...")
    for global_step in tqdm(range(args.total_timesteps), desc="Training", unit="step"):
        # 计算epsilon
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        
        # 选择动作
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
            log_probs = F.log_softmax(q_values, dim=1)
            mu = q_values.detach().cpu().numpy()

        # 执行动作
        next_obs, _, terminations, truncations, infos = envs.step(actions)

        # 计算奖励
        with torch.no_grad():
            if args.reward_type == "sparse":
                # 使用环境原生奖励
                env = gym.make(args.env_id)
                native_rewards = np.array([env.step(a)[1] for a in actions])
                rewards = native_rewards
            else:
                # 使用自定义奖励函数
                rewards = reward_function.observe_reward(obs, actions, next_obs)

            # 计算Advantage（如果使用）
            if args.use_advantage:
                q_values_current = q_network(torch.Tensor(obs).to(device))
                v_values_current = value_network(torch.Tensor(obs).to(device))
                advantage = q_values_current.gather(1, torch.LongTensor(actions).to(device).unsqueeze(1)).squeeze() - v_values_current.squeeze()
                if global_step % 100 == 0:
                    print(f"Step {global_step}: Advantage = {advantage.item():.4f}")
            else:
                advantage = 0.0

            # 存储转换
            transition = Transition(state=obs, action=actions, reward=rewards, log_probs=log_probs, mu=mu, overline_V=0.0, advantage=advantage)
            epidata.append(transition)

        # 处理结束信息
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episodic_return = info["episode"]["r"]
                    print(f"Episode completed! global_step={global_step}, episodic_return={episodic_return:.2f}")
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    if args.reward_type != "sparse":
                        reward_function.D_xi.append(reward_function.store_V(epidata))
                    epidata = []

        # 处理截断观察
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # 更新观察
        obs = next_obs

        # 训练
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                # 从缓冲区采样
                data = rb.sample(args.batch_size)
                
                # 计算TD目标
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                
                # 计算Q值和损失
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                # 记录和打印
                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    sps = int(global_step / (time.time() - start_time))
                    print(f"Step {global_step}: SPS={sps}, Loss={loss.item():.4f}, Q-value={old_val.mean().item():.4f}")
                    writer.add_scalar("charts/SPS", sps, global_step)

                # 优化Q网络
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 更新目标网络
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

            # 优化值网络（如果使用Advantage）
            if args.use_advantage and global_step % args.train_frequency == 0:
                for t in reversed(range(len(epidata))):
                    if t == len(epidata) - 1:
                        next_value = 0.0
                    else:
                        next_value = value_network(torch.Tensor(epidata[t+1].state).to(device)).squeeze().item()
                    q_value = epidata[t].reward + args.gamma * next_value
                    v_value = value_network(torch.Tensor(epidata[t].state).to(device)).squeeze()
                    advantage_target = q_value - v_value.item()
                    v_loss = F.mse_loss(v_value, torch.tensor(advantage_target).to(device))
                    value_optimizer.zero_grad()
                    v_loss.backward()
                    value_optimizer.step()

            # 优化奖励函数（如果使用自定义奖励）
            if global_step % args.reward_frequency == 0 and args.reward_type != "sparse":
                print(f"Updating reward function at step {global_step}...")
                reward_function.optimize_reward(agent=q_network, use_advantage=args.use_advantage, value_network=value_network if args.use_advantage else None)

    # 保存模型（如果需要）
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        
        # 评估模型
        from cleanrl_utils.evals.dqn_eval import evaluate
        print("Evaluating model...")
        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)
            print(f"Evaluation episode {idx+1}: return={episodic_return:.2f}")

        # 上传模型（如果需要）
        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub
            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    # 关闭环境和writer
    envs.close()
    writer.close()
    print("Training completed!")