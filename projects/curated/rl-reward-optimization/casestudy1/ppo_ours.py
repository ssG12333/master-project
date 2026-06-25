# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple
from utils.reward_machine import RewardFunction
from tqdm import tqdm


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False

    env_id: str = "CartPole-v1"
    total_timesteps: int = 500000
    learning_rate: float = 2.5e-4
    num_envs: int = 4
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

    reward_frequency: int = 1024
    hidden_dim: int = 256
    encode_dim: int = 64
    activate_function: str = "tanh"
    last_activate_function: str = "None"
    reward_lr: float = 1e-4
    buffer_size: int = 100
    reward_buffer_size: int = 100

    reward_type: str = "sparse"
    use_advantage: bool = False
    use_adamw: bool = False
    n_samples: int = 10


def make_env(env_id, idx, capture_video, run_name):
    """
    创建环境函数
    
    Args:
        env_id: 环境ID
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
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    正交初始化网络层
    
    Args:
        layer: 网络层
        std: 标准差
        bias_const: 偏置常量
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """
    PPO智能体网络
    """
    def __init__(self, envs):
        super().__init__()
        # 评论家网络（值函数）
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # 演员网络（策略）
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        """获取状态值"""
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """
        获取动作和值
        
        Args:
            x: 状态
            action: 动作（可选）
        """
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x), probs

    def get_action_prob_from_mu(self, mu, n):
        """
        从分布中采样动作
        
        Args:
            mu: 分布
            n: 采样数量
        """
        actions = mu.sample((n,))
        log_probs = mu.log_prob(actions)
        return actions.transpose(0, 1), log_probs.transpose(0, 1)


if __name__ == "__main__":
    # 解析参数
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    
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
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # 初始化智能体
    agent = Agent(envs).to(device)

    # 选择优化器（Adam或AdamW）
    if args.use_adamw:
        optimizer = optim.AdamW(agent.parameters(), lr=args.learning_rate, eps=1e-5, weight_decay=0.01)
        print("Using AdamW optimizer with weight decay 0.01")
    else:
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
        print("Using Adam optimizer")

    # 初始化存储
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # 初始化奖励函数和其他变量
    global_step = 0
    reward_function = RewardFunction(env=envs, args=args, device=device)
    Transition = namedtuple('Transition', ['state', 'action', 'reward', 'log_probs', 'mu', 'overline_V', 'advantage'])
    epidata = []
    start_time = time.time()
    
    # 重置环境
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # 主训练循环
    print(f"Starting training for {args.total_timesteps} timesteps...")
    for iteration in tqdm(range(1, args.num_iterations + 1), desc="Iterations", unit="iter"):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # 收集轨迹
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            now_ob = next_obs
            obs[step] = next_obs
            dones[step] = next_done

            # 获取动作和值
            with torch.no_grad():
                action, logprob, _, value, mu = agent.get_action_and_value(now_ob)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # 执行动作
            next_obs, _, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            
            # 计算奖励
            with torch.no_grad():
                if args.reward_type == "sparse":
                    # 使用环境原生奖励
                    env = gym.make(args.env_id)
                    native_rewards = np.array([env.step(a.cpu().numpy())[1] for a in action.cpu()])
                    reward = native_rewards
                else:
                    # 使用自定义奖励函数
                    reward = reward_function.observe_reward(now_ob.cpu().numpy(), action.cpu().numpy(), next_obs)

                # 计算Advantage（如果使用）
                if args.use_advantage:
                    q_value = value
                    v_value = value
                    advantage = q_value - v_value.detach()
                    if step % 100 == 0:
                        print(f"Step {step}: Advantage = {advantage.mean().item():.4f}")
                else:
                    advantage = 0.0

            # 存储奖励和观察
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            transition = Transition(state=now_ob.cpu().numpy(), action=action.cpu().numpy(), reward=reward, log_probs=logprob.cpu().numpy(), mu=mu, overline_V=0.0, advantage=advantage.cpu().item())
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

        # 计算GAE优势函数
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # 展平批次
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # 优化策略和值函数
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # 计算新的策略
                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # 计算KL散度
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                # 处理优势函数
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # 选择策略梯度权重
                if args.use_advantage:
                    policy_weight = mb_advantages
                else:
                    policy_weight = b_returns[mb_inds] - b_values[mb_inds].detach()
                    if args.norm_adv:
                        policy_weight = (policy_weight - policy_weight.mean()) / (policy_weight.std() + 1e-8)

                # 计算策略损失
                pg_loss1 = -policy_weight * ratio
                pg_loss2 = -policy_weight * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # 计算值函数损失
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # 计算总损失
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # 优化
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            # 提前停止（如果KL散度超过阈值）
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # 计算解释方差
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # 优化奖励函数
        if args.reward_type != "sparse" and global_step % args.reward_frequency == 0:
            print(f"Updating reward function at iteration {iteration}...")
            reward_function.optimize_reward(agent, use_advantage=args.use_advantage, value_network=None)

        # 记录和打印
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        sps = int(global_step / (time.time() - start_time))
        print(f"Iteration {iteration}: SPS={sps}, Loss={loss.item():.4f}, Policy Loss={pg_loss.item():.4f}, Value Loss={v_loss.item():.4f}")
        writer.add_scalar("charts/SPS", sps, global_step)

    # 关闭环境和writer
    envs.close()
    writer.close()
    print("Training completed!")