import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import copy
import scipy.signal as signal
import torch
import numpy as np
from task_env import TaskEnv
from parameters import *
from net import ActorCritic
from cbba_solver import CBBASolver
import matplotlib.pyplot as plt

class Worker:
    def __init__(self, network, device='cpu', plot_figure = False, seed = None):
        self.device = torch.device(device)
        self.plot_figure = plot_figure
        self.seed = seed
        self.env = TaskEnv(seed = self.seed)
        self.net = network
        self.experience = None

    # y[t]=x[t]+gamma*x[t+1]+gamma^2*x[t+2]+…,用来计算折扣奖励
    @staticmethod
    def discount(x, gamma):
        return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    def reset_env(self):
        self.env = TaskEnv(seed = self.seed)

    def run_episode(self):
        self.reset_env()
        self.net.eval()
        episode_buffer = [[] for _ in range(6)]
        values = []
        while not self.env.finished and self.env.current_time < MAX_TIME:
            with torch.no_grad():
                decision_agents, current_time = self.env.next_decision()
                groups = self.env.get_unique_group(decision_agents)
                self.env.current_time = current_time
                self.env.task_update()
                self.env.agent_update()
                for group in groups:
                    while len(group) > 0:
                        leader_id = np.random.choice(group)
                        agent = self.env.agent_dic[leader_id]
                        if not agent['returned']:
                            mask = self.env.get_finished_task_mask()
                            if np.sum(mask) == self.env.task_num:
                                mask = np.insert(mask, 0, False)
                            else:
                                mask = np.insert(mask, 0, True)

                            agents_info = torch.FloatTensor(self.env.get_current_agent_status(agent)).unsqueeze(0).to(self.device)
                            task_info = torch.FloatTensor(self.env.get_current_task_status(agent)).unsqueeze(0).to(self.device)
                            mask_info = torch.tensor(mask).unsqueeze(0).to(self.device)

                            # action = np.random.choice(np.where(np.logical_not(mask))[0])

                            action, log_prob, entropy, value = self.net.act(agents_info, task_info, mask_info)
                            action_info = action.view(1, 1).to(self.device)

                            target =  action.item() - 1

                            group = self.env.step(group, leader_id, target)
                            self.env.task_update()
                            self.env.agent_update()
                            episode_buffer[0].append(agents_info)
                            episode_buffer[1].append(task_info)
                            episode_buffer[2].append(mask_info)
                            episode_buffer[3].append(action_info)
                            values.append(value.item())
                self.env.finished = self.env.check_finished()
        T = len(values)

        if T > 0:
            final_reward = -float(self.env.current_time)
            rewards = np.zeros(T, dtype=np.float32)
            rewards[-1] = final_reward
            Gt = self.discount(rewards, GAMMA)
            adv = Gt - np.array(values, dtype=np.float32)
            for t in range(T):
                episode_buffer[4].append(
                    torch.tensor([[Gt[t]]], dtype=torch.float32).to(self.device)
                )
                episode_buffer[5].append(
                    torch.tensor([[adv[t]]], dtype=torch.float32).to(self.device)
                )
        self.experience = episode_buffer

        if self.plot_figure:
            self.env.plot_figure()

        return episode_buffer

    def _select_nearest_action(self, task_info, mask_info):
        """
        task_info: (1, 17, 5)
        mask_info: (1, 17), True 表示不可选
        返回:
            action: int
        """
        # 取出单个样本
        task_feat = task_info[0]  # (17, 5)
        invalid = mask_info[0].bool()  # (17,)

        valid_actions = torch.where(~invalid)[0]

        # task_info 的最后两维是相对当前位置的 dx, dy
        dx_dy = task_feat[valid_actions, 3:5]  # (K, 2)
        dist = torch.norm(dx_dy, dim=-1)  # (K,)

        # 选距离最小的动作
        best_idx = torch.argmin(dist).item()
        action = int(valid_actions[best_idx].item())

        return action

    @staticmethod
    def calculate_euclidean_distance(loc1, loc2):
        return np.linalg.norm(np.array(loc1) - np.array(loc2))

    def _select_greedy_action(self, env, agent, mask):
        valid_actions = np.where(~mask)[0]
        if len(valid_actions) == 0:
            return 0

        best_action = valid_actions[0]
        best_score = -np.inf

        for action in valid_actions:
            target = action - 1
            if target == -1:
                task_loc = env.depot['location']
                score = 1.0 / (self.calculate_euclidean_distance(agent['location'], task_loc) + 1e-8)
            else:
                task = env.task_dic[target]
                dist = self.calculate_euclidean_distance(agent['location'], task['location'])
                score = task['requirements'] / (dist + 1e-8)

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _run_cbba_once(self, env, agent, mask):
        return self._select_greedy_action(env, agent, mask)

    def _run_episode_with_policy(self, env, policy='model', deterministic=True):
        if policy == 'model':
            self.net.eval()

        with torch.no_grad():
            while not env.finished and env.current_time < MAX_TIME:
                decision_agents, current_time = env.next_decision()
                groups = env.get_unique_group(decision_agents)

                env.current_time = current_time
                env.task_update()
                env.agent_update()

                for group in groups:
                    while len(group) > 0:
                        leader_id = np.random.choice(group)
                        agent = env.agent_dic[leader_id]

                        if agent['returned']:
                            group.remove(leader_id)
                            continue

                        mask = env.get_finished_task_mask()
                        if np.sum(mask) == env.task_num:
                            mask = np.insert(mask, 0, False)
                        else:
                            mask = np.insert(mask, 0, True)

                        agents_info = torch.tensor(
                            env.get_current_agent_status(agent),
                            dtype=torch.float32,
                            device=self.device
                        ).unsqueeze(0)

                        task_info = torch.tensor(
                            env.get_current_task_status(agent),
                            dtype=torch.float32,
                            device=self.device
                        ).unsqueeze(0)

                        mask_info = torch.tensor(
                            mask,
                            dtype=torch.bool,
                            device=self.device
                        ).unsqueeze(0)

                        if policy == 'model':
                            action, _, _, _ = self.net.act(
                                agents_info, task_info, mask_info,
                                deterministic=deterministic
                            )
                            action = int(action.item())

                        elif policy == 'random':
                            valid_actions = np.where(~mask)[0]
                            action = int(np.random.choice(valid_actions))

                        elif policy == 'nearest':
                            action = self._select_nearest_action(task_info, mask_info)

                        elif policy == 'cbba':
                            action = self._run_cbba_once(env, agent, mask)

                        elif policy == 'greedy':
                            action = self._select_greedy_action(env, agent, mask)

                        else:
                            raise ValueError(f"Unknown policy: {policy}")

                        target = action - 1
                        group = env.step(group, leader_id, target)

                        env.task_update()
                        env.agent_update()

                env.finished = env.check_finished()

        return float(env.current_time)

    def compare_policies(
            self,
            model_path,
            num_trials=200,
            deterministic=True,
            plot_path='model_vs_baselines.png',
            show_plot=True,
            save_gifs=True,
            gif_dir='gifs'
    ):
        checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.net.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.net.load_state_dict(checkpoint)

        self.net.to(self.device)
        self.net.eval()

        model_times = []
        random_times = []
        nearest_times = []
        cbba_times = []
        greedy_times = []

        if save_gifs:
            os.makedirs(gif_dir, exist_ok=True)

        base_seed = 0 if self.seed is None else self.seed

        for i in range(num_trials):
            trial_seed = base_seed + i

            np.random.seed(trial_seed)
            env_model = TaskEnv(seed=trial_seed)
            model_time = self._run_episode_with_policy(
                env_model, policy='model', deterministic=deterministic
            )

            np.random.seed(trial_seed)
            env_random = TaskEnv(seed=trial_seed)
            random_time = self._run_episode_with_policy(
                env_random, policy='random', deterministic=True
            )

            np.random.seed(trial_seed)
            env_nearest = TaskEnv(seed=trial_seed)
            nearest_time = self._run_episode_with_policy(
                env_nearest, policy='nearest', deterministic=True
            )

            np.random.seed(trial_seed)
            env_cbba = TaskEnv(seed=trial_seed)
            cbba_time = self._run_episode_with_policy(
                env_cbba, policy='cbba', deterministic=True
            )

            np.random.seed(trial_seed)
            env_greedy = TaskEnv(seed=trial_seed)
            greedy_time = self._run_episode_with_policy(
                env_greedy, policy='greedy', deterministic=True
            )

            model_times.append(model_time)
            random_times.append(random_time)
            nearest_times.append(nearest_time)
            cbba_times.append(cbba_time)
            greedy_times.append(greedy_time)

            if save_gifs and i == 0:
                env_model.plot_figure(save_path=os.path.join(gif_dir, 'model_trial_0.gif'), fps=5)
                env_random.plot_figure(save_path=os.path.join(gif_dir, 'random_trial_0.gif'), fps=5)
                env_nearest.plot_figure(save_path=os.path.join(gif_dir, 'nearest_trial_0.gif'), fps=5)
                env_cbba.plot_figure(save_path=os.path.join(gif_dir, 'cbba_trial_0.gif'), fps=5)
                env_greedy.plot_figure(save_path=os.path.join(gif_dir, 'greedy_trial_0.gif'), fps=5)
                print(f"[Trial 0] GIFs saved to {gif_dir}/")

            if (i + 1) % 10 == 0:
                print(
                    f"[{i + 1}/{num_trials}] "
                    f"model={model_time:.4f}, "
                    f"random={random_time:.4f}, "
                    f"nearest={nearest_time:.4f}, "
                    f"cbba={cbba_time:.4f}, "
                    f"greedy={greedy_time:.4f}"
                )

        model_times = np.array(model_times, dtype=np.float32)
        random_times = np.array(random_times, dtype=np.float32)
        nearest_times = np.array(nearest_times, dtype=np.float32)
        cbba_times = np.array(cbba_times, dtype=np.float32)
        greedy_times = np.array(greedy_times, dtype=np.float32)

        trials = np.arange(1, num_trials + 1)

        model_mean = np.cumsum(model_times) / trials
        random_mean = np.cumsum(random_times) / trials
        nearest_mean = np.cumsum(nearest_times) / trials
        cbba_mean = np.cumsum(cbba_times) / trials
        greedy_mean = np.cumsum(greedy_times) / trials

        plt.figure(figsize=(12, 8))

        plt.plot(trials, model_times, alpha=0.2, linewidth=1.0, color='red', label='Model')
        plt.plot(trials, random_times, alpha=0.2, linewidth=1.0, color='gray', label='Random')
        plt.plot(trials, nearest_times, alpha=0.2, linewidth=1.0, color='orange', label='Nearest')
        plt.plot(trials, cbba_times, alpha=0.2, linewidth=1.0, color='green', label='CBBA')
        plt.plot(trials, greedy_times, alpha=0.2, linewidth=1.0, color='purple', label='Greedy')

        plt.plot(trials, model_mean, linewidth=3.0, color='red', label=f'Model mean ({model_mean[-1]:.3f})')
        plt.plot(trials, random_mean, linewidth=2.0, color='gray', label=f'Random mean ({random_mean[-1]:.3f})')
        plt.plot(trials, nearest_mean, linewidth=2.0, color='orange', label=f'Nearest mean ({nearest_mean[-1]:.3f})')
        plt.plot(trials, cbba_mean, linewidth=2.0, color='green', label=f'CBBA mean ({cbba_mean[-1]:.3f})')
        plt.plot(trials, greedy_mean, linewidth=2.0, color='purple', label=f'Greedy mean ({greedy_mean[-1]:.3f})')

        plt.xlabel('Trial', fontsize=12)
        plt.ylabel('Final Time', fontsize=12)
        plt.title(f'Model vs Baselines ({num_trials} Trials)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)

        if show_plot:
            plt.show()
        else:
            plt.close()

        print("\n===== Compare Result =====")
        print(f"Model   mean final time: {model_times.mean():.4f}")
        print(f"Random  mean final time: {random_times.mean():.4f}")
        print(f"Nearest mean final time: {nearest_times.mean():.4f}")
        print(f"CBBA    mean final time: {cbba_times.mean():.4f}")
        print(f"Greedy  mean final time: {greedy_times.mean():.4f}")
        print(f"Plot saved to         : {plot_path}")

        return {
            "model_times": model_times,
            "random_times": random_times,
            "nearest_times": nearest_times,
            "cbba_times": cbba_times,
            "greedy_times": greedy_times,
            "model_mean": float(model_times.mean()),
            "random_mean": float(random_times.mean()),
            "nearest_mean": float(nearest_times.mean()),
            "cbba_mean": float(cbba_times.mean()),
            "greedy_mean": float(greedy_times.mean()),
            "plot_path": plot_path,
        }

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    net = ActorCritic().to(torch.device('cpu'))
    worker = Worker(net, device='cpu', plot_figure=False, seed=0)

    model_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'ckpt_ep80000.pt')
    plot_path = os.path.join(os.path.dirname(__file__), 'results', 'model_vs_baselines.png')
    gif_dir = os.path.join(os.path.dirname(__file__), 'gifs_comparison')
    
    result = worker.compare_policies(
        model_path=model_path,
        num_trials=30,
        deterministic=True,
        plot_path=plot_path,
        show_plot=True,
        save_gifs=True,
        gif_dir=gif_dir
    )

    print(f"\nModel: {result['model_mean']:.4f}")
    print(f"Random: {result['random_mean']:.4f}")
    print(f"Nearest: {result['nearest_mean']:.4f}")
    print(f"CBBA: {result['cbba_mean']:.4f}")
    print(f"Greedy: {result['greedy_mean']:.4f}")

