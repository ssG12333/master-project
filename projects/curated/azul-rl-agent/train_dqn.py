import subprocess
import os


def run_game(agent1, agent2, num_games=10):
    command = [
        "python", "general_game_runner.py",
        "-g", "Azul",
        "-a", f"{agent1},{agent2}",
        "-m", str(num_games),
        "-s",  # 保存游戏记录
        "-Q"  # 安静模式，不输出详细信息
    ]

    subprocess.run(command, check=True)
def main():
    my_agent = "agents.t_070.train"
    opponents = [
        my_agent,  # 自己
        "agents.generic.random",  # ranAgent
        "agents.mmAgent"  # mmAgent
    ]
    num_iterations = 100  # 总训练迭代次数
    games_per_iteration = 10  # 每次迭代的游戏次数
    for iteration in range(num_iterations):
        print(f"Starting iteration {iteration + 1}/{num_iterations}")
        for opponent in opponents:
            print(f"Training against {opponent}")
            run_game(my_agent, opponent, games_per_iteration)
    print("Training completed")


if __name__ == "__main__":
    main()
    #python  general_game_runner.py -g Azul -a agents.agent1_model.h5,agents.mmAgent
