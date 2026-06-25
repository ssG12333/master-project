import sys
import time
import json
import os
import pygame
import torch
from pygame.locals import *
from ControllerList import (
    PlayerController,
    AStarController,
    QLearningController,
    DDPGController,
    DeepQNetworkController
)
import random
from maze import Maze
from player import Player
from state import GameState
import numpy as np
import matplotlib.pyplot as plt
import os
import threading


class Button:
    def __init__(self, x, y, width, height, text, callback):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = (30, 144, 255)
        self.hover_color = (0, 191, 255)
        self.text = text
        self.callback = callback
        self.font = pygame.font.SysFont("MicrosoftYaHei", 28)

    def draw(self, screen):
        mouse_pos = pygame.mouse.get_pos()
        color = self.hover_color if self.rect.collidepoint(mouse_pos) else self.color
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        text_surf = self.font.render(self.text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.callback()


class GameManager:
    def __init__(self):
        pygame.init()
        self.maze_sizes = [10, 15, 20]
        self.current_size_index = 0
        self.current_size = self.maze_sizes[self.current_size_index]
        self.maze = Maze(800, 600, self.current_size, self.current_size)
        self.screen = pygame.display.set_mode((1000, 600))
        pygame.display.set_caption("路径探索类迷宫控制逻辑研究系统")
        self.clock = pygame.time.Clock()
        self.current_state = GameState.MENU
        self.player = Player(self.maze)
        self.controller = None
        self.success_time = 0
        self.current_score = 0
        self.path_records = []
        self.start_time = 0
        self.performance_data = {}
        self.training_thread = None
        self.training_in_progress = False
        self.training_progress = 0
        self.total_training_steps = 0
        self.total_training_steps = 6 * 3  # 总训练步数（6次训练 * 3种迷宫尺寸）
        self.training_completed = False  # 训练是否完成
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有可用的GPU
        self.deadends = set()  # 记录死路位置
        self._init_ui()

    def _init_ui(self):
        self.menu_buttons = [
            Button(300, 150, 400, 60, "玩家控制模式", self._start_player_mode),
            Button(300, 230, 400, 60, "A*算法路径规划", self._start_astar_mode),
            Button(300, 310, 400, 60, "Q学习模式", self._start_qlearning_mode),
            Button(300, 390, 400, 60, "深度强化学习", self._start_dqn_mode),
            Button(300, 470, 400, 60, "DDPG算法", self._start_ddpg_mode),
            Button(300, 550, 400, 60, "性能对比", self._start_performance_comparison),
            Button(300, 630, 400, 60, "训练模型", self._start_training),
            Button(300, 710, 400, 60, "退出游戏", sys.exit)
        ]

        self.game_buttons = [
            Button(810, 100, 180, 50, "返回主菜单", self._return_to_menu),
            Button(810, 180, 180, 50, "重新生成迷宫", self._regenerate_maze),
            Button(810, 260, 180, 50, "重新挑战", self._rechallenge_maze),
            Button(810, 340, 180, 50, f"尺寸: {self.current_size}x{self.current_size}", self._cycle_maze_size)
        ]

    def _rechallenge_maze(self):
        self.player.reset()
        if self.controller and hasattr(self.controller, 'reset_path'):
            self.controller.reset_path()
        else:
            if self.controller:
                self.controller.path = []
                self.controller.is_success = False

    def _start_player_mode(self):
        self.controller = PlayerController(self.player)
        self._start_game()

    def _start_astar_mode(self):
        self.controller = AStarController(self.maze, self.player, self)
        self._start_game()

    def _start_qlearning_mode(self):
        self.controller = QLearningController(self.maze, self.player, self)
        self._start_game()

    def _start_dqn_mode(self):
        self.controller = DeepQNetworkController(self.maze, self.player, self)
        self._start_game()

    def _start_ddpg_mode(self):
        self.controller = DDPGController(self.maze, self.player, self)
        self._start_game()

    def _start_performance_comparison(self):
        self._run_performance_tests()
        self._generate_performance_charts()

    def _save_training_progress(self):
        """保存训练进度"""
        progress_data = {
            "progress": self.training_progress,
            "completed": self.training_completed
        }
        with open("training_progress.json", "w") as f:
            json.dump(progress_data, f)
        print(f"训练进度已保存: {self.training_progress}/{self.total_training_steps}")

    def _load_training_progress(self):
        """加载训练进度"""
        if os.path.exists("training_progress.json"):
            with open("training_progress.json", "r") as f:
                progress_data = json.load(f)
            self.training_progress = progress_data["progress"]
            self.training_completed = progress_data["completed"]
            print(f"加载训练进度: {self.training_progress}/{self.total_training_steps}")
            return True
        return False

    def _run_performance_tests(self):
        """运行性能测试"""
        data_file = "performance_data.json"
        if os.path.exists(data_file):
            with open(data_file, "r") as f:
                self.performance_data = json.load(f)
            print("加载已有性能数据")
            return

        try:
            if not self._load_training_progress():
                self.training_progress = 0
                self.training_completed = False

            algorithms = ["A*", "QLearning", "DQN", "DDPG"]
            maze_sizes = [10, 15, 20]

            for size in maze_sizes:
                self.maze = Maze(800, 600, size, size)
                self.player = Player(self.maze)

                if str(size) not in self.performance_data:
                    self.performance_data[str(size)] = {"times": [], "path_lengths": []}

                for _ in range(6):
                    if not self.maze._validate_maze():
                        self.maze._ensure_path()

                    times = []
                    path_lengths = []

                    # A*算法
                    controller = AStarController(self.maze, self.player, self)
                    start_time = time.time()
                    while not controller.is_success:
                        controller.update()
                        self._handle_deadend(controller)  # 处理死路
                    times.append(time.time() - start_time)
                    path_lengths.append(len(controller.path))

                    # QLearning算法
                    self.player.reset()
                    controller = QLearningController(self.maze, self.player, self)
                    start_time = time.time()
                    while not controller.is_success:
                        controller.update()
                        self._handle_deadend(controller)  # 处理死路
                    times.append(time.time() - start_time)
                    path_lengths.append(len(controller.path))

                    # DQN算法
                    self.player.reset()
                    controller = DeepQNetworkController(self.maze, self.player, self)
                    start_time = time.time()
                    while not controller.is_success:
                        controller.update()
                        self._handle_deadend(controller)  # 处理死路
                    times.append(time.time() - start_time)
                    path_lengths.append(len(controller.path))

                    # DDPG算法
                    self.player.reset()
                    controller = DDPGController(self.maze, self.player, self)
                    start_time = time.time()
                    while not controller.is_success:
                        controller.update()
                        self._handle_deadend(controller)  # 处理死路
                    times.append(time.time() - start_time)
                    path_lengths.append(len(controller.path))

                    self.performance_data[str(size)]["times"].append(times)
                    self.performance_data[str(size)]["path_lengths"].append(path_lengths)

                    # 更新训练进度
                    self.training_progress += 1
                    print(f"训练进度: {self.training_progress}/{self.total_training_steps}")

                    # 保存训练进度
                    self._save_training_progress()

                    # 检查是否完成训练
                    if self.training_progress >= self.total_training_steps:
                        self.training_completed = True
                        break

                if self.training_completed:
                    break

            # 保存性能数据
            with open(data_file, "w") as f:
                json.dump(self.performance_data, f)
            print("性能数据已保存")

            # 保存训练进度
            self.training_completed = True
            self._save_training_progress()

        except KeyboardInterrupt:
            print("\n训练被用户中断")
            self._save_training_progress()
            print("训练进度已保存，可以下次继续训练")

    def _handle_deadend(self, controller):
        """综合处理死路"""
        if not self.maze._validate_maze():
            # 尝试动态调整迷宫
            self._fix_maze_deadends(controller)
            if not self.maze._validate_maze():
                # 动态调整失败，重新生成迷宫
                self._regenerates_maze(controller)
                print("动态调整失败，已重新生成迷宫")

    def _fix_maze_deadends(self, controller):
        """动态调整迷宫，打通死路"""
        from collections import deque
        visited = [[False for _ in range(self.maze.grid_cols)] for _ in range(self.maze.grid_rows)]
        parent = [[None for _ in range(self.maze.grid_cols)] for _ in range(self.maze.grid_rows)]
        queue = deque([self.maze.start_pos])
        visited[self.maze.start_pos[1]][self.maze.start_pos[0]] = True
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            x, y = queue.popleft()
            if (x, y) == self.maze.end_pos:
                break
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.maze.grid_cols and 0 <= ny < self.maze.grid_rows:
                    if not visited[ny][nx] and self.maze.maze_layout[ny][nx] == 0:
                        visited[ny][nx] = True
                        parent[ny][nx] = (x, y)
                        queue.append((nx, ny))

        # 如果终点不可达，打通路径
        if parent[self.maze.end_pos[1]][self.maze.end_pos[0]] is None:
            current = self.maze.end_pos
            while current != self.maze.start_pos:
                prev = parent[current[1]][current[0]]
                if prev is None:
                    # 随机选择一个方向打通
                    dx, dy = random.choice(directions)
                    nx, ny = current[0] + dx, current[1] + dy
                    if 0 <= nx < self.maze.grid_cols and 0 <= ny < self.maze.grid_rows:
                        self.maze.maze_layout[ny][nx] = 0
                        current = (nx, ny)
                    else:
                        break
                else:
                    current = prev

    def _regenerates_maze(self, controller):
        """在遇到死路时重新生成迷宫"""
        self.maze.reset_maze()
        self.player.reset()
        controller.reset_path()  # 重置控制器的路径
        print("已重新生成迷宫")

    def _generate_performance_charts(self):
        """生成性能对比图表"""
        data_file = "performance_data.json"
        if not os.path.exists(data_file):
            print("未找到性能数据文件，开始运行性能测试...")
            self._run_performance_tests()

        with open(data_file, "r") as f:
            self.performance_data = json.load(f)

        maze_sizes = [10, 15, 20]
        algorithms = ["A*", "QLearning", "DQN", "DDPG"]

        # 时间对比图
        for size in maze_sizes:
            times = np.array(self.performance_data[str(size)]["times"])
            plt.figure(figsize=(10, 6))

            for i, algo in enumerate(algorithms):
                plt.plot(range(1, 7), times[:, i], marker='o', label=algo)

            plt.xlabel('寻路次数')
            plt.ylabel('寻路时间（秒）')
            plt.title(f'{size}x{size}迷宫寻路时间对比')
            plt.legend()
            plt.grid(True, axis='y')
            plt.savefig(f'time_comparison_{size}.png')
            plt.close()

        # 平均路径长度柱状图
        for size in maze_sizes:
            path_lengths = np.array(self.performance_data[str(size)]["path_lengths"])
            avg_lengths = np.mean(path_lengths, axis=0)

            plt.figure(figsize=(10, 6))
            plt.bar(algorithms, avg_lengths)
            plt.xlabel('算法')
            plt.ylabel('平均路径长度')
            plt.title(f'{size}x{size}迷宫平均路径长度对比')
            plt.grid(True, axis='y')
            plt.savefig(f'path_length_comparison_{size}.png')
            plt.close()

        print("性能对比图表已生成")

    def _start_game(self):
        """开始游戏时记录起始时间"""
        self.maze.reset_maze()
        self.player.reset()
        self.current_state = GameState.PLAYING
        self.start_time = time.time()  # 记录游戏开始时间
        self.success_time = 0

    def _return_to_menu(self):
        self.current_state = GameState.MENU
        self.controller = None

        # 修改 _regenerate_maze 方法，确保控制器更新

    def _regenerate_maze(self):
        self.maze = Maze(800, 600, self.current_size, self.current_size)
        self.player = Player(self.maze)  # 重置玩家
        # 更新控制器的迷宫和玩家引用
        if self.controller:
            if hasattr(self.controller, 'maze'):
                self.controller.maze = self.maze
            if hasattr(self.controller, 'player'):
                self.controller.player = self.player
        # 重置玩家位置
        self.player.x = self.maze.start_pos[0] * self.maze.cell_size + self.maze.cell_size // 2
        self.player.y = self.maze.start_pos[1] * self.maze.cell_size + self.maze.cell_size // 2
        # 强制重新寻路（如果控制器支持）
        if self.controller and hasattr(self.controller, '_find_path'):
            self.controller._find_path()
        self.player.reset()
        if self.controller:
            self.controller.path = []
            self.controller.is_success = False

    # 修改 _cycle_maze_size 方法，添加控制器检查
    def _cycle_maze_size(self):
        self.current_size_index = (self.current_size_index + 1) % 3
        self.current_size = self.maze_sizes[self.current_size_index]
        if isinstance(self.controller, QLearningController):
            self.controller._init_q_table()  # 重置Q表
        # 检查控制器是否存在且是否有reset_path方法
        if self.controller and hasattr(self.controller, 'reset_path'):
            self.controller.reset_path()  # 仅对支持该方法的控制器生效
        # 更新按钮文本
        for btn in self.game_buttons:
            if btn.text.startswith("尺寸"):
                btn.text = f"尺寸:{self.current_size}x{self.current_size}"
        # 重新生成迷宫
        self._regenerate_maze()

    def run(self):
        while True:
            events = pygame.event.get()
            for event in events:
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                self._handle_event(event)

            self.screen.fill((40, 40, 40))

            if self.current_state == GameState.MENU:
                self._draw_menu()
            else:
                if self.controller:
                    self.controller.update(None)
                self._draw_game()
                self._draw_control_panel()

            if self.training_in_progress:
                self._draw_training_progress()

            pygame.display.flip()
            self.clock.tick(30)

    def _handle_event(self, event):
        if event.type == MOUSEBUTTONDOWN:
            if self.current_state == GameState.MENU:
                for btn in self.menu_buttons:
                    btn.handle_event(event)
            else:
                for btn in self.game_buttons:
                    btn.handle_event(event)

    def _draw_menu(self):
        title_font = pygame.font.SysFont("SimHei", 48)
        title_surf = title_font.render("迷宫路径探索逻辑研究系统", True, (255, 255, 255))
        title_rect = title_surf.get_rect(center=(500, 80))
        self.screen.blit(title_surf, title_rect)
        for btn in self.menu_buttons:
            btn.draw(self.screen)

    def _draw_game(self):
        self.maze.draw(self.screen)
        if self.controller and hasattr(self.controller, 'path'):
            if self.controller.path:
                for i in range(len(self.controller.path) - 1):
                    start = (
                        self.controller.path[i][0] * self.maze.cell_size + self.maze.cell_size // 2,
                        self.controller.path[i][1] * self.maze.cell_size + self.maze.cell_size // 2
                    )
                    end = (
                        self.controller.path[i + 1][0] * self.maze.cell_size + self.maze.cell_size // 2,
                        self.controller.path[i + 1][1] * self.maze.cell_size + self.maze.cell_size // 2
                    )
                    pygame.draw.line(self.screen, (255, 0, 0), start, end, 4)
        self.player.draw(self.screen)

    def _draw_control_panel(self):
        pygame.draw.rect(self.screen, (60, 60, 60), (800, 0, 200, 600))

        for btn in self.game_buttons:
            btn.draw(self.screen)

        font = pygame.font.SysFont("MicrosoftYaHei", 20)
        y_offset = 340

        mode_name = self.controller.__class__.__name__.replace("Controller", "") if self.controller else ""
        text = f"当前模式: {mode_name}"
        text_surf = font.render(text, True, (255, 255, 255))
        self.screen.blit(text_surf, (810, y_offset))
        y_offset += 30

        step_count = self.controller.step_count if self.controller and hasattr(self.controller, "step_count") else 0
        step_text = font.render(f"步数: {step_count}", True, (255, 255, 255))
        self.screen.blit(step_text, (810, y_offset))
        y_offset += 30

    def _start_training(self):
        if not self.training_in_progress:
            self.training_thread = threading.Thread(target=self._run_training)
            self.training_thread.daemon = True
            self.training_thread.start()

    def _run_training(self):
        self.training_in_progress = True
        self.total_training_steps = 3 * 3 * 100
        current_step = 0

        for size in [10, 15, 20]:
            self.maze = Maze(800, 600, size, size)
            self.player = Player(self.maze)
            # QLearning训练
            controller = QLearningController(self.maze, self.player, self)
            for episode in range(100):
                while not controller.is_success:
                    controller.update(None)
                self.player.reset()
                controller.reset_path()
                current_step += 1
                self.training_progress = current_step / self.total_training_steps * 100
                print(f"训练进度: {self.training_progress:.2f}%")
            # DQN训练
            self.player.reset()
            controller = DeepQNetworkController(self.maze, self.player, self)
            for episode in range(100):
                while not controller.is_success:
                    controller.update(None)
                self.player.reset()
                controller.reset_path()
                current_step += 1
                self.training_progress = current_step / self.total_training_steps * 100
                print(f"训练进度: {self.training_progress:.2f}%")
            # DDPG训练
            self.player.reset()
            controller = DDPGController(self.maze, self.player, self)
            for episode in range(100):
                while not controller.is_success:
                    controller.update(None)
                self.player.reset()
                controller.reset_path()
                current_step += 1
                self.training_progress = current_step / self.total_training_steps * 100
                print(f"训练进度: {self.training_progress:.2f}%")

        self.training_in_progress = False
        print("训练完成！")

    def _draw_training_progress(self):
        if self.training_in_progress:
            font = pygame.font.SysFont("MicrosoftYaHei", 20)
            progress_text = font.render(f"训练进度: {self.training_progress:.2f}%", True, (255, 255, 255))
            self.screen.blit(progress_text, (810, 400))
            progress_bar_width = 180
            progress_bar_height = 20
            progress_x = 810
            progress_y = 430
            pygame.draw.rect(self.screen, (60, 60, 60),
                             (progress_x, progress_y, progress_bar_width, progress_bar_height))
            pygame.draw.rect(self.screen, (0, 255, 0), (
            progress_x, progress_y, progress_bar_width * self.training_progress / 100, progress_bar_height),
                             border_radius=5)
def main():
    game = GameManager()
    try:
        game.run()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        if hasattr(game, '_save_training_progress'):
            game._save_training_progress()
            print("训练进度已保存，可以下次继续训练")
if __name__ == "__main__":
    main()


