import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob


def smooth_data(data, weight=0.6):
    smoothed = []
    last = data[0] if len(data) > 0 else 0
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def parse_tensorboard_log(log_file):
    returns = []
    steps = []
    if not os.path.exists(log_file):
        return np.array(steps), np.array(returns)

    with open(log_file, "r") as f:
        for line in f:
            if "episodic_return" in line and "global_step=" in line:
                try:
                    parts = line.split("global_step=")
                    if len(parts) > 1:
                        step_part = parts[1].split(",")[0].strip()
                        step = int(step_part)
                        ret_part = line.split("episodic_return=")
                        if len(ret_part) > 1:
                            ret = float(ret_part[1].split(",")[0].strip())
                            steps.append(step)
                            returns.append(ret)
                except:
                    pass
    return np.array(steps), np.array(returns)


def load_all_results(exp_dir):
    all_data = {}

    log_files = glob(os.path.join(exp_dir, "*.log"))

    for log_file in log_files:
        filename = os.path.basename(log_file)
        parts = filename.replace(".log", "").split("_")

        if len(parts) >= 4:
            env_id = parts[0]
            algo = parts[1]
            reward_type = parts[2]
            seed_str = "_".join(parts[3:]) if len(parts) > 3 else "seed1"
            if seed_str.startswith("seed"):
                seed = int(seed_str.replace("seed", ""))
            else:
                seed = 1

            steps, returns = parse_tensorboard_log(log_file)

            if env_id not in all_data:
                all_data[env_id] = {}
            if algo not in all_data[env_id]:
                all_data[env_id][algo] = {}
            if reward_type not in all_data[env_id][algo]:
                all_data[env_id][algo][reward_type] = []

            if len(returns) > 0:
                all_data[env_id][algo][reward_type].append({
                    "seed": seed,
                    "steps": steps,
                    "returns": returns
                })

    return all_data


def downsample_to_common_x(steps_list, returns_list, num_points=500):
    if len(steps_list) == 0:
        return np.linspace(0, 1, num_points), np.zeros(num_points)

    min_step = min(s[0] for s in steps_list if len(s) > 0)
    max_step = max(s[-1] for s in steps_list if len(s) > 0)

    common_x = np.linspace(min_step, max_step, num_points)

    interpolated_returns = []
    for steps, returns in zip(steps_list, returns_list):
        if len(steps) > 0:
            interp_returns = np.interp(common_x, steps, returns)
            interpolated_returns.append(interp_returns)
        else:
            interpolated_returns.append(np.zeros(num_points))

    return common_x, np.array(interpolated_returns)


def plot_learning_curves(all_data, output_dir):
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {
        "sparse": "#2ecc71",
        "rmbo": "#3498db",
        "ours": "#e74c3c"
    }
    labels = {
        "sparse": "Sparse (DQN/PPO)",
        "rmbo": "RMBO",
        "ours": "Ours (Adv+AdamW)"
    }

    for env_id in all_data:
        for algo in all_data[env_id]:
            fig, ax = plt.subplots(figsize=(10, 6))

            for reward_type in ["sparse", "rmbo", "ours"]:
                if reward_type not in all_data[env_id][algo]:
                    continue

                data_list = all_data[env_id][algo][reward_type]
                if len(data_list) == 0:
                    continue

                steps_list = [d["steps"] for d in data_list]
                returns_list = [d["returns"] for d in data_list]

                common_x, interpolated = downsample_to_common_x(steps_list, returns_list)

                mean_returns = np.mean(interpolated, axis=0)
                std_returns = np.std(interpolated, axis=0)

                smoothed_mean = smooth_data(mean_returns, weight=0.7)
                smoothed_std = smooth_data(std_returns, weight=0.7)

                x_normalized = common_x / common_x[-1] if common_x[-1] > 0 else common_x

                ax.plot(x_normalized, smoothed_mean, label=labels[reward_type],
                       color=colors[reward_type], linewidth=2)
                ax.fill_between(x_normalized,
                               np.array(smoothed_mean) - np.array(smoothed_std),
                               np.array(smoothed_mean) + np.array(smoothed_std),
                               color=colors[reward_type], alpha=0.2)

            ax.set_xlabel("Training Progress (Time Steps)", fontsize=12)
            ax.set_ylabel("Episodic Return", fontsize=12)
            ax.set_title(f"{env_id} - {algo.upper()} Learning Curve", fontsize=14)
            ax.legend(loc="lower right", fontsize=10)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            output_file = os.path.join(output_dir, f"{env_id}_{algo}_learning_curve.png")
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {output_file}")


def plot_final_performance_bar(all_data, output_dir):
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {
        "sparse": "#2ecc71",
        "rmbo": "#3498db",
        "ours": "#e74c3c"
    }
    labels = {
        "sparse": "Sparse",
        "rmbo": "RMBO",
        "ours": "Ours"
    }

    for idx, env_id in enumerate(["CartPole-v1", "LunarLander-v2"]):
        if env_id not in all_data:
            continue

        ax = axes[idx]

        for algo in ["dqn", "ppo"]:
            if algo not in all_data[env_id]:
                continue

            x_positions = []
            means = []
            stds = []
            bar_colors = []

            for i, reward_type in enumerate(["sparse", "rmbo", "ours"]):
                if reward_type not in all_data[env_id][algo]:
                    continue

                data_list = all_data[env_id][algo][reward_type]
                if len(data_list) == 0:
                    continue

                final_returns = []
                for d in data_list:
                    if len(d["returns"]) > 0:
                        final_returns.append(d["returns"][-1])

                if final_returns:
                    x_positions.append(i * 3 + (0 if algo == "dqn" else 1))
                    means.append(np.mean(final_returns))
                    stds.append(np.std(final_returns))
                    bar_colors.append(colors[reward_type])

            x = np.arange(3)
            width = 0.35

            dqn_means = []
            dqn_stds = []
            ppo_means = []
            ppo_stds = []

            for reward_type in ["sparse", "rmbo", "ours"]:
                if reward_type in all_data[env_id].get("dqn", {}):
                    dqn_data = all_data[env_id]["dqn"][reward_type]
                    if dqn_data:
                        final_returns = [d["returns"][-1] if d["returns"] else 0 for d in dqn_data]
                        dqn_means.append(np.mean(final_returns))
                        dqn_stds.append(np.std(final_returns))
                    else:
                        dqn_means.append(0)
                        dqn_stds.append(0)
                else:
                    dqn_means.append(0)
                    dqn_stds.append(0)

                if reward_type in all_data[env_id].get("ppo", {}):
                    ppo_data = all_data[env_id]["ppo"][reward_type]
                    if ppo_data:
                        final_returns = [d["returns"][-1] if d["returns"] else 0 for d in ppo_data]
                        ppo_means.append(np.mean(final_returns))
                        ppo_stds.append(np.std(final_returns))
                    else:
                        ppo_means.append(0)
                        ppo_stds.append(0)
                else:
                    ppo_means.append(0)
                    ppo_stds.append(0)

            x = np.arange(3)
            width = 0.35

            bars1 = ax.bar(x - width/2, dqn_means, width, label='DQN', color=[colors["sparse"], colors["rmbo"], colors["ours"]],
                          yerr=dqn_stds, capsize=3, alpha=0.8)
            bars2 = ax.bar(x + width/2, ppo_means, width, label='PPO', color=[colors["sparse"], colors["rmbo"], colors["ours"]],
                          yerr=ppo_stds, capsize=3, alpha=0.5, hatch='//')

            ax.set_ylabel('Final Episodic Return', fontsize=12)
            ax.set_title(f'{env_id}', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(['Sparse', 'RMBO', 'Ours'])
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_file = os.path.join(output_dir, "final_performance_bar.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_ablation_learning_curves(ablation_data, output_dir):
    if not ablation_data:
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {
        "Value+Adam": "#3498db",
        "Adv+Adam": "#e74c3c",
        "Value+AdamW": "#9b59b6",
        "Adv+AdamW": "#2ecc71"
    }

    env_id = "LunarLander-v2"
    algo = "ppo"

    if env_id not in ablation_data or algo not in ablation_data[env_id]:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for method_name in ["Value+Adam", "Adv+Adam", "Value+AdamW", "Adv+AdamW"]:
        if method_name not in ablation_data[env_id][algo]:
            continue

        curves = ablation_data[env_id][algo][method_name]["learning_curves"]
        if not curves:
            continue

        steps_list = [c["steps"] for c in curves]
        returns_list = [c["returns"] for c in curves]

        common_x, interpolated = downsample_to_common_x(steps_list, returns_list)

        mean_returns = np.mean(interpolated, axis=0)
        std_returns = np.std(interpolated, axis=0)

        smoothed_mean = smooth_data(mean_returns, weight=0.7)
        smoothed_std = smooth_data(std_returns, weight=0.7)

        x_normalized = common_x / common_x[-1] if common_x[-1] > 0 else common_x

        ax.plot(x_normalized, smoothed_mean, label=method_name,
               color=colors[method_name], linewidth=2)
        ax.fill_between(x_normalized,
                       np.array(smoothed_mean) - np.array(smoothed_std),
                       np.array(smoothed_mean) + np.array(smoothed_std),
                       color=colors[method_name], alpha=0.2)

    ax.set_xlabel("Training Progress (Time Steps)", fontsize=12)
    ax.set_ylabel("Episodic Return", fontsize=12)
    ax.set_title(f"Ablation Study: LunarLander-v2 - PPO", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, "ablation_learning_curves.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_ablation_results(ablation_data, output_dir):
    if not ablation_data:
        print("No ablation results to plot")
        return

    plot_ablation_learning_curves(ablation_data, output_dir)

    plt.style.use('seaborn-v0_8-whitegrid')

    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ["Value+Adam", "Adv+Adam", "Value+AdamW", "Adv+AdamW"]
    colors = ["#3498db", "#e74c3c", "#9b59b6", "#2ecc71"]

    env_id = "LunarLander-v2"
    algo = "ppo"

    if env_id not in ablation_data or algo not in ablation_data[env_id]:
        return

    means = []
    stds = []

    for method in methods:
        if method in ablation_data[env_id][algo]:
            final_returns = ablation_data[env_id][algo][method]["final_returns"]
            if final_returns:
                means.append(np.mean(final_returns))
                stds.append(np.std(final_returns))
            else:
                means.append(0)
                stds.append(0)
        else:
            means.append(0)
            stds.append(0)

    x = np.arange(len(methods))
    bars = ax.bar(x, means, color=colors, yerr=stds, capsize=5, alpha=0.8)

    ax.set_ylabel('Final Episodic Return', fontsize=12)
    ax.set_title(f'Ablation Study: {env_id} - {algo.upper()}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, mean in zip(bars, means):
        if mean > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   f'{mean:.1f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    output_file = os.path.join(output_dir, "ablation_results.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def load_ablation_results(ablation_dir):
    results_file = os.path.join(ablation_dir, "ablation_results.json")
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            return json.load(f)
    return {}


def generate_plots(exp_dir, ablation_dir=None):
    results_dir = os.path.join(exp_dir, "plots")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    all_data = load_all_results(exp_dir)

    print("Generating learning curves...")
    plot_learning_curves(all_data, results_dir)

    print("Generating bar charts...")
    plot_final_performance_bar(all_data, results_dir)

    if ablation_dir and os.path.exists(ablation_dir):
        print("Generating ablation results...")
        ablation_data = load_ablation_results(ablation_dir)
        plot_ablation_results(ablation_data, results_dir)

    print(f"\nAll plots saved to: {results_dir}")
    return all_data


if __name__ == "__main__":
    import sys
    exp_dir = sys.argv[1] if len(sys.argv) > 1 else "experiment_results"
    ablation_dir = sys.argv[2] if len(sys.argv) > 2 else "ablation_results"
    generate_plots(exp_dir, ablation_dir)