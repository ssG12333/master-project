import subprocess
import os
from tqdm import tqdm
import torch  # 导入PyTorch

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_game(agent1, agent2, num_games=10):
    command = [
        "python", "general_game_runner.py",
        "-g", "Azul",
        "-a", f"{agent1},{agent2}",
        "-m", str(num_games),
        "-s",
        "-Q"
    ]

    subprocess.run(command, check=True)


def main():
    my_agent = "agents.t_070.train"
    opponents = [
        "agents.generic.random",  # ranAgent
         my_agent # mmAgent
    ]
    num_iterations = 5000
    games_per_iteration = 10

    for iteration in tqdm(range(num_iterations), desc="Training Iterations"):
        for opponent in opponents:
            tqdm.write(f"Training against {opponent}")
            run_game(my_agent, opponent, games_per_iteration)

    print("Training completed")


if __name__ == "__main__":
    main()
