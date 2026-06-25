import os
import subprocess
import json
import time
import numpy as np
from datetime import datetime
from tqdm import tqdm


"""
实验配置
- CartPole-v1: 简单的平衡任务
- LunarLander-v2: 登月任务
- 每个任务运行DQN和PPO算法
- 每个算法运行3个随机种子
"""
EXPERIMENTS = {
    "CartPole-v1": {
        "dqn": {
            "total_timesteps": 500000,
            "seeds": [1, 2, 3],
        },
        "ppo": {
            "total_timesteps": 500000,
            "seeds": [1, 2, 3],
        }
    },
    "LunarLander-v2": {
        "dqn": {
            "total_timesteps": 1000000,
            "seeds": [1, 2, 3],
        },
        "ppo": {
            "total_timesteps": 1000000,
            "seeds": [1, 2, 3],
        }
    }
}

# 奖励类型：sparse（环境原生）、rmbo（原始方法）、ours（改进方法）
REWARD_TYPES = ["sparse", "rmbo", "ours"]
# 算法类型
ALGORITHMS = ["dqn", "ppo"]

# 结果保存目录
RESULTS_DIR = "experiment_results"


def setup_results_dir():
    """
    创建结果保存目录
    """
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(RESULTS_DIR, f"exp_{timestamp}")
    os.makedirs(exp_dir)
    return exp_dir


def run_single_experiment(env_id, algo, reward_type, seed, total_timesteps, exp_dir):
    """
    运行单个实验
    
    Args:
        env_id: 环境ID
        algo: 算法（dqn或ppo）
        reward_type: 奖励类型（sparse、rmbo、ours）
        seed: 随机种子
        total_timesteps: 总时间步
        exp_dir: 结果保存目录
    """
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if algo == "dqn":
        if reward_type == "sparse":
            script = os.path.join(script_dir, "casestudy1", "dqn.py")
            cmd = [
                "python", script,
                "--env_id", env_id,
                "--total_timesteps", str(total_timesteps),
                "--seed", str(seed),
            ]
        elif reward_type == "rmbo":
            script = os.path.join(script_dir, "casestudy1", "dqn.py")
            cmd = [
                "python", script,
                "--env_id", env_id,
                "--total_timesteps", str(total_timesteps),
                "--seed", str(seed),
            ]
        else:
            script = os.path.join(script_dir, "casestudy1", "dqn_ours.py")
            cmd = [
                "python", script,
                "--env_id", env_id,
                "--total_timesteps", str(total_timesteps),
                "--seed", str(seed),
                "--use_advantage", "True",
                "--use_adamw", "True",
            ]
    else:
        if reward_type == "sparse":
            script = os.path.join(script_dir, "casestudy1", "ppo.py")
            cmd = [
                "python", script,
                "--env_id", env_id,
                "--total_timesteps", str(total_timesteps),
                "--seed", str(seed),
            ]
        elif reward_type == "rmbo":
            script = os.path.join(script_dir, "casestudy1", "ppo.py")
            cmd = [
                "python", script,
                "--env_id", env_id,
                "--total_timesteps", str(total_timesteps),
                "--seed", str(seed),
            ]
        else:
            script = os.path.join(script_dir, "casestudy1", "ppo_ours.py")
            cmd = [
                "python", script,
                "--env_id", env_id,
                "--total_timesteps", str(total_timesteps),
                "--seed", str(seed),
                "--use_advantage", "True",
                "--use_adamw", "True",
            ]

    run_name = f"{env_id}_{algo}_{reward_type}_seed{seed}"
    log_file = os.path.join(exp_dir, f"{run_name}.log")

    print(f"Running: {' '.join(cmd)}")
    print(f"Log file: {log_file}")

    # 设置PYTHONPATH环境变量，确保子脚本可以正确导入模块
    env = os.environ.copy()
    project_root = os.path.dirname(os.path.abspath(__file__))
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    print(f"Set PYTHONPATH to: {env['PYTHONPATH']}")

    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env
        )
        for line in iter(process.stdout.readline, ''):
            if line:
                f.write(line)
                f.flush()
                print(line.rstrip())

    return process.wait()


def parse_log_for_returns(log_file):
    """
    从日志文件中解析返回值
    
    Args:
        log_file: 日志文件路径
    """
    returns = []
    if not os.path.exists(log_file):
        return returns

    with open(log_file, "r") as f:
        for line in f:
            if "episodic_return" in line and "global_step" in line:
                try:
                    parts = line.split("episodic_return=")
                    if len(parts) > 1:
                        ret = float(parts[1].split(",")[0].split()[0])
                        returns.append(ret)
                except:
                    pass
    return returns


def run_experiments():
    """
    运行所有实验
    """
    exp_dir = setup_results_dir()
    all_results = {}

    # 遍历所有环境
    for env_id in EXPERIMENTS:
        all_results[env_id] = {}
        # 遍历所有算法
        for algo in ALGORITHMS:
            all_results[env_id][algo] = {}

            config = EXPERIMENTS[env_id][algo]
            total_timesteps = config["total_timesteps"]
            seeds = config["seeds"]

            # 遍历所有奖励类型
            for reward_type in REWARD_TYPES:
                all_results[env_id][algo][reward_type] = {"seeds": [], "mean": 0, "std": 0}

                # 遍历所有种子
                for seed in tqdm(seeds, desc=f"{env_id} {algo} {reward_type}"):
                    print(f"\n{'='*60}")
                    print(f"Running: {env_id} | {algo.upper()} | {reward_type} | Seed {seed}")
                    print(f"{'='*60}\n")

                    return_code = run_single_experiment(
                        env_id, algo, reward_type, seed, total_timesteps, exp_dir
                    )

                    if return_code == 0:
                        log_file = os.path.join(exp_dir, f"{env_id}_{algo}_{reward_type}_seed{seed}.log")
                        returns = parse_log_for_returns(log_file)
                        if returns:
                            all_results[env_id][algo][reward_type]["seeds"].append(returns)
                            print(f"  Got {len(returns)} episodes, final return: {returns[-1]:.2f}")
                    else:
                        print(f"  Experiment failed with return code {return_code}")

                    time.sleep(2)

                # 计算平均值和标准差
                if all_results[env_id][algo][reward_type]["seeds"]:
                    final_returns = [s[-1] if s else 0 for s in all_results[env_id][algo][reward_type]["seeds"]]
                    all_results[env_id][algo][reward_type]["mean"] = np.mean(final_returns)
                    all_results[env_id][algo][reward_type]["std"] = np.std(final_returns)

    # 保存结果
    results_file = os.path.join(exp_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # 打印实验总结
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")

    for env_id in all_results:
        print(f"\n{env_id}:")
        for algo in all_results[env_id]:
            print(f"  {algo.upper()}:")
            for reward_type in all_results[env_id][algo]:
                data = all_results[env_id][algo][reward_type]
                if data["seeds"]:
                    print(f"    {reward_type}: {data['mean']:.2f} ± {data['std']:.2f}")
                else:
                    print(f"    {reward_type}: No data")

    print(f"\nResults saved to: {results_file}")
    return exp_dir, all_results


if __name__ == "__main__":
    """
    主函数：运行所有实验
    """
    exp_dir, all_results = run_experiments()