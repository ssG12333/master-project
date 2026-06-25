import os
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from net import ActorCritic
from worker import Worker


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_checkpoint(path, model, optimizer, episode, best_final_time, args):
    ckpt = {
        "episode": episode,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_final_time": best_final_time,
        "args": vars(args),
    }
    torch.save(ckpt, path)


def build_batch_from_episode_buffer(episode_buffer, device):
    agents_info = torch.cat(episode_buffer[0], dim=0).to(device)
    task_info = torch.cat(episode_buffer[1], dim=0).to(device)
    mask_info = torch.cat(episode_buffer[2], dim=0).to(device)

    actions = torch.cat(episode_buffer[3], dim=0).to(device)
    Gt = torch.cat(episode_buffer[4], dim=0).to(device)
    adv = torch.cat(episode_buffer[5], dim=0).to(device)

    actions = actions.squeeze(-1).long()
    Gt = Gt.squeeze(-1)
    adv = adv.squeeze(-1)

    return agents_info, task_info, mask_info, actions, Gt, adv


def train_one_episode(worker, model, optimizer, device, args):
    episode_buffer = worker.run_episode()

    if len(episode_buffer[0]) == 0:
        return {
            "loss": 0.0,
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "final_time": float("inf"),
            "episode_reward": float("-inf"),
            "steps": 0,
        }

    agents_info, task_info, mask_info, actions, Gt, adv = build_batch_from_episode_buffer(
        episode_buffer, device
    )

    if args.norm_adv and adv.numel() > 1:
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

    model.train()

    logits, values = model(agents_info, task_info, mask_info)
    values = values.squeeze(-1)

    dist = torch.distributions.Categorical(logits=logits)
    log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    actor_loss = -(log_probs * adv.detach()).mean()
    critic_loss = F.mse_loss(values, Gt.detach())

    loss = actor_loss + args.value_coef * critic_loss - args.entropy_coef * entropy

    optimizer.zero_grad()
    loss.backward()

    if args.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

    optimizer.step()

    final_time = float(worker.env.current_time)
    episode_reward = -final_time

    return {
        "loss": float(loss.item()),
        "actor_loss": float(actor_loss.item()),
        "critic_loss": float(critic_loss.item()),
        "entropy": float(entropy.item()),
        "final_time": final_time,
        "episode_reward": episode_reward,
        "steps": len(episode_buffer[0]),
    }


def parse_args():
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--episodes", type=int, default=200000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)

    # model
    parser.add_argument("--hidden-dim", type=int, default=128)

    # loss
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--norm-adv", action="store_true")

    # logging / save
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--gif-every", type=int, default=10000)
    parser.add_argument("--gif-fps", type=int, default=5)

    # dirs
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--gif-dir", type=str, default="gifs")

    # resume
    parser.add_argument("--resume", type=str, default="")

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    ensure_dir(args.save_dir)
    ensure_dir(args.gif_dir)

    set_seed(args.seed)

    model = ActorCritic(hidden_dim=args.hidden_dim).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    start_episode = 1
    best_final_time = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_episode = ckpt["episode"] + 1
        best_final_time = ckpt.get("best_final_time", float("inf"))
        print(f"Resumed from {args.resume}, start_episode={start_episode}, best_final_time={best_final_time:.4f}")

    worker = Worker(model, device=args.device, plot_figure=False, seed=None)

    for episode in range(start_episode, args.episodes + 1):
        worker.seed = args.seed + episode
        stats = train_one_episode(worker, model, optimizer, device, args)

        final_time = stats["final_time"]
        episode_reward = stats["episode_reward"]

        # best model: final_time 越小越好
        if final_time < best_final_time:
            best_final_time = final_time
            best_path = os.path.join(args.save_dir, "best_model.pt")
            save_checkpoint(best_path, model, optimizer, episode, best_final_time, args)
            print(f"[Episode {episode}] saved BEST model -> {best_path} | final_time={final_time:.4f}")

        # regular checkpoint
        if args.save_every > 0 and episode % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"ckpt_ep{episode}.pt")
            save_checkpoint(ckpt_path, model, optimizer, episode, best_final_time, args)
            print(f"[Episode {episode}] saved checkpoint -> {ckpt_path}")

        # gif
        if args.gif_every > 0 and episode % args.gif_every == 0:
            gif_path = os.path.join(args.gif_dir, f"episode_{episode}.gif")
            worker.env.plot_figure(save_path=gif_path, fps=args.gif_fps)
            print(f"[Episode {episode}] saved gif -> {gif_path}")

        # print
        if episode % args.print_every == 0 or episode == 1:
            print(
                f"[Episode {episode:5d}] "
                f"reward={episode_reward:9.4f} | "
                f"final_time={final_time:9.4f} | "
                f"steps={stats['steps']:4d} | "
                f"loss={stats['loss']:9.4f} | "
                f"actor={stats['actor_loss']:9.4f} | "
                f"critic={stats['critic_loss']:9.4f} | "
                f"entropy={stats['entropy']:8.4f} | "
                f"best_final_time={best_final_time:9.4f}"
            )

    last_path = os.path.join(args.save_dir, "last_model.pt")
    save_checkpoint(last_path, model, optimizer, args.episodes, best_final_time, args)
    print(f"Training finished. Last model saved to {last_path}")


if __name__ == "__main__":
    main()