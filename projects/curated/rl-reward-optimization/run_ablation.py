import os
import subprocess
import json
import time
import numpy as np
from datetime import datetime
from tqdm import tqdm


"""
消融实验配置
- 仅在LunarLander-v2环境上运行
- 使用PPO算法
- 测试4种配置：
  1. Value+Adam: 原始基线（Q值 + Adam优化器）
  2. Adv+Adam: 仅优势函数（Advantage + Adam优化器）
  3. Value+AdamW: 仅AdamW（Q值 + AdamW优化器）
  4. Adv+AdamW: 完整方法（Advantage + AdamW优化器）
"""
ABLATION_EXPERIMENTS = {
    "LunarLander-v2": {
        "ppo": {
            "total_timesteps": 1000000,
            "seeds": [1, 2, 3],
            "methods": {
                "Value+Adam": {
                    "reward_type": "rmbo",
                    "use_advantage": False,
                    "use_adamw": False,
                },
                "Adv+Adam": {
                    "reward_type": "ours",
                    "use_advantage": True,
                    "use_adamw": False,
                },
                "Value+AdamW": {
                    "reward_type": "rmbo",
                    "use_advantage": False,
                    "use_adamw": True,
                },
                "Adv+AdamW": {
                    "reward_type": "ours",
                    "use_advantage": True,
                    "use_adamw": True,
                }
            }
        }
    }
}

# 结果保存目录
RESULTS_DIR = "ablation_results"


def setup_results_dir():
    """
    创建结果保存目录
    """
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(RESULTS_DIR, f"ablation_{timestamp}")
    os.makedirs(exp_dir)
    return exp_dir


def run_ablation_experiment(env_id, algo, method_name, config, seed, total_timesteps, exp_dir):
    """
    运行单个消融实验
    
    Args:
        env_id: 环境ID
        algo: 算法（dqn或ppo）
        method_name: 方法名称
        config: 方法配置
        seed: 随机种子
        total_timesteps: 总时间步
        exp_dir: 结果保存目录
    """
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if algo == "ppo":
        if config["reward_type"] == "sparse":
            script = os.path.join(script_dir, "casestudy1", "ppo.py")
            cmd = [
                "python", script,
                "--env_id", env_id,
                "--total_timesteps", str(total_timesteps),
                "--seed", str(seed),
                "--reward_type", "sparse",
            ]
        else:
            script = os.path.join(script_dir, "casestudy1", "ppo_ours.py")
            cmd = [
                "python", script,
                "--env_id", env_id,
                "--total_timesteps", str(total_timesteps),
                "--seed", str(seed),
                "--reward_type", config["reward_type"],
                "--use_advantage", str(config["use_advantage"]),
                "--use_adamw", str(config["use_adamw"]),
            ]
    else:
        if config["reward_type"] == "sparse":
            script = os.path.join(script_dir, "casestudy1", "dqn.py")
            cmd = [
                "python", script,
                "--env_id", env_id,
                "--total_timesteps", str(total_timesteps),
                "--seed", str(seed),
                "--reward_type", "sparse",
            ]
        else:
            script = os.path.join(script_dir, "casestudy1", "dqn_ours.py")
            cmd = [
                "python", script,
                "--env_id", env_id,
                "--total_timesteps", str(total_timesteps),
                "--seed", str(seed),
                "--reward_type", config["reward_type"],
                "--use_advantage", str(config["use_advantage"]),
                "--use_adamw", str(config["use_adamw"]),
            ]

    run_name = f"{env_id}_{algo}_{method_name}_seed{seed}"
    log_file = os.path.join(exp_dir, f"{run_name}.log")

    print(f"Running: {' '.join(cmd)}")
    print(f"Log file: {log_file}")

    # 确保在项目根目录运行，这样可以正确导入utils模块
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=project_root  # 设置工作目录为项目根目录
        )
        for line in iter(process.stdout.readline, ''):
            if line:
                f.write(line)
                f.flush()
                print(line.rstrip())

    return process.wait()


def parse_log_for_returns(log_file):
    """
    从日志文件中解析返回值和步骤
    
    Args:
        log_file: 日志文件路径
    """
    returns = []
    steps = []
    if not os.path.exists(log_file):
        return steps, returns

    with open(log_file, "r") as f:
        for line in f:
            if "episodic_return" in line and "global_step=" in line:
                try:
                    parts = line.split("global_step=")
                    if len(parts) > 1:
                        step = int(parts[1].split(",")[0].strip())
                        ret = float(line.split("episodic_return=")[1].split(",")[0].strip())
                        steps.append(step)
                        returns.append(ret)
                except:
                    pass
    return steps, returns


def run_ablation_experiments():
    """
    运行所有消融实验
    """
    exp_dir = setup_results_dir()
    all_results = {}

    # 遍历所有环境
    for env_id in ABLATION_EXPERIMENTS:
        all_results[env_id] = {}
        # 遍历所有算法
        for algo in ABLATION_EXPERIMENTS[env_id]:
            all_results[env_id][algo] = {}
            config = ABLATION_EXPERIMENTS[env_id][algo]
            total_timesteps = config["total_timesteps"]
            seeds = config["seeds"]

            # 遍历所有方法
            for method_name, method_config in config["methods"].items():
                all_results[env_id][algo][method_name] = {
                    "seeds": [],
                    "final_returns": [],
                    "learning_curves": []
                }

                # 遍历所有种子
                for seed in tqdm(seeds, desc=f"{env_id} {algo} {method_name}"):
                    print(f"\n{'='*60}")
                    print(f"Running Ablation: {env_id} | {algo.upper()} | {method_name} | Seed {seed}")
                    print(f"{'='*60}\n")

                    return_code = run_ablation_experiment(
                        env_id, algo, method_name, method_config, seed, total_timesteps, exp_dir
                    )

                    if return_code == 0:
                        log_file = os.path.join(exp_dir, f"{env_id}_{algo}_{method_name}_seed{seed}.log")
                        steps, returns = parse_log_for_returns(log_file)
                        if returns:
                            all_results[env_id][algo][method_name]["seeds"].append(seed)
                            all_results[env_id][algo][method_name]["learning_curves"].append({
                                "seed": seed,
                                "steps": steps,
                                "returns": returns
                            })
                            all_results[env_id][algo][method_name]["final_returns"].append(returns[-1])
                            print(f"  Got {len(returns)} episodes, final return: {returns[-1]:.2f}")
                    else:
                        print(f"  Experiment failed with return code {return_code}")

                    time.sleep(2)

    # 保存结果
    results_file = os.path.join(exp_dir, "ablation_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # 打印实验总结
    print(f"\n{'='*60}")
    print("ABLATION EXPERIMENT SUMMARY")
    print(f"{'='*60}")

    for env_id in all_results:
        for algo in all_results[env_id]:
            print(f"\n{env_id} - {algo.upper()}:")
            for method_name in all_results[env_id][algo]:
                data = all_results[env_id][algo][method_name]
                if data["final_returns"]:
                    mean_return = np.mean(data["final_returns"])
                    std_return = np.std(data["final_returns"])
                    print(f"  {method_name}: {mean_return:.2f} ± {std_return:.2f}")
                else:
                    print(f"  {method_name}: No data")

    print(f"\nResults saved to: {results_file}")
    return exp_dir, all_results


if __name__ == "__main__":
    """
    主函数：运行所有消融实验
    """
    exp_dir, all_results = run_ablation_experiments()