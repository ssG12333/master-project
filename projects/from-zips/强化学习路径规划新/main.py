import sys
import time
import pygame
from collections import deque
from pygame.locals import *
from ControllerList import (
    PlayerController,
    AStarController,
    QLearningController,
    DDPGController,
    DeepQNetworkController
)
from maze import Maze
from player import Player
from state import GameState
import numpy as np
import os
import torch

# 设置环境变量以避免 OpenMP 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.callback()

class GameManager:
    def __init__(self):
        pygame.init()
        self.maze_sizes = [10, 15, 20]
        self.current_size_index = 0
        self.current_size = self.maze_sizes[self.current_size_index]
        self.maze = Maze(800, 600, self.current_size, self.current_size)
        self.screen = pygame.display.set_mode((1200, 800))
        pygame.display.set_caption("路径探索类迷宫控制逻辑研究系统")
        self.clock = pygame.time.Clock()
        self.current_state = GameState.MENU
        self.player = Player(self.maze)
        self.controller = None
        self.controllers = []
        self.selected_algorithms = []
        self.success_time = 0
        self.start_time = 0
        self.performance_data = {}
        self.training_in_progress = False
        self.training_progress = 0
        self.total_training_steps = 0
        self.current_episode = 0
        self.training_completed = False
        self.is_running = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.path_colors = {
            "AStar": (255, 0, 0),
            "QLearning": (0, 255, 0),
            "DQN": (0, 0, 255),
            "DDPG": (255, 255, 0)
        }
        self.test_completed = False
        self.training_controller = None
        self.training_episode = 0
        self.training_step_count = 0
        self.training_episode_reward = 0
        self._init_ui()

    def _init_ui(self):
        self.menu_buttons = [
            Button(400, 100, 400, 60, "玩家控制模式", self._start_player_mode),
            Button(400, 180, 400, 60, "A*算法路径规划", self._start_astar_mode),
            Button(400, 260, 400, 60, "Q学习模式", self._start_qlearning_mode),
            Button(400, 340, 400, 60, "深度强化学习", self._start_dqn_mode),
            Button(400, 420, 400, 60, "DDPG算法", self._start_ddpg_mode),
            Button(400, 500, 400, 60, "性能对比", self._start_performance_comparison),
            Button(400, 580, 400, 60, "退出游戏", sys.exit)
        ]
        self.game_buttons = [
            Button(810, 50, 180, 50, "返回主菜单", self._return_to_menu),
            Button(810, 120, 180, 50, "重新生成迷宫", self._regenerate_maze),
            Button(810, 190, 180, 50, f"尺寸: {self.current_size}x{self.current_size}", self._cycle_maze_size),
            Button(810, 260, 180, 50, "开始训练", self._start_training),
            Button(810, 330, 180, 50, "停止训练", self._stop_training),
            Button(810, 400, 180, 50, "开始测试", self._start_testing),
            Button(810, 470, 180, 50, "停止测试", self._stop_testing),
            Button(810, 540, 180, 50, "重置测试", self._reset_testing)
        ]
        self.comparison_buttons = [
            Button(400, 100, 400, 60, "A*算法", lambda: self._toggle_algorithm("AStar")),
            Button(400, 180, 400, 60, "Q学习模式", lambda: self._toggle_algorithm("QLearning")),
            Button(400, 260, 400, 60, "深度强化学习", lambda: self._toggle_algorithm("DQN")),
            Button(400, 340, 400, 60, "DDPG算法", lambda: self._toggle_algorithm("DDPG")),
            Button(400, 420, 400, 60, f"地图尺寸: {self.current_size}x{self.current_size}", self._cycle_maze_size),
            Button(400, 500, 400, 60, "开始训练", self._start_comparison_training),
            Button(400, 580, 400, 60, "停止训练", self._stop_training),
            Button(400, 660, 400, 60, "开始测试", self._start_comparison_testing),
            Button(400, 740, 400, 60, "停止测试", self._stop_testing),
            Button(400, 820, 400, 60, "重置测试", self._reset_testing),
            Button(400, 900, 400, 60, "返回主菜单", self._return_to_menu)
        ]

    def _toggle_algorithm(self, algo_name):
        if algo_name in self.selected_algorithms:
            self.selected_algorithms.remove(algo_name)
        else:
            if len(self.selected_algorithms) < 3:
                self.selected_algorithms.append(algo_name)
            else:
                print("最多选择三种算法！")

    def _cycle_maze_size(self):
        self.current_size_index = (self.current_size_index + 1) % len(self.maze_sizes)
        self.current_size = self.maze_sizes[self.current_size_index]
        self._update_maze()
        for btn in self.game_buttons + self.comparison_buttons:
            if btn.text.startswith("尺寸") or btn.text.startswith("地图尺寸"):
                btn.text = f"尺寸: {self.current_size}x{self.current_size}"

    def _update_maze(self):
        self.maze = Maze(800, 600, self.current_size, self.current_size)
        self.player = Player(self.maze)
        if self.controller:
            self.controller.maze = self.maze
            self.controller.player = self.player
            if hasattr(self.controller, 'reset_path'):
                self.controller.reset_path()
        self.controllers = []
        for algo in self.selected_algorithms:
            self._create_controller(algo)

    def _create_controller(self, algo_name):
        player = Player(self.maze)
        if algo_name == "Player":
            controller = PlayerController(player)
        elif algo_name == "AStar":
            controller = AStarController(self.maze, player, self)
        elif algo_name == "QLearning":
            controller = QLearningController(self.maze, player, self)
        elif algo_name == "DQN":
            controller = DeepQNetworkController(self.maze, player, self)
        elif algo_name == "DDPG":
            controller = DDPGController(self.maze, player, self)
        else:
            return
        self.controllers.append((algo_name, controller))

    def _start_player_mode(self):
        self.controller = PlayerController(self.player)
        self._start_game()

    def _start_astar_mode(self):
        self.controller = AStarController(self.maze, self.player, self)
        self.current_state = GameState.PLAYING
        self.training_completed = True

    def _start_qlearning_mode(self):
        self.controller = QLearningController(self.maze, self.player, self)
        self.current_state = GameState.PLAYING

    def _start_dqn_mode(self):
        self.controller = DeepQNetworkController(self.maze, self.player, self)
        self.current_state = GameState.PLAYING

    def _start_ddpg_mode(self):
        self.controller = DDPGController(self.maze, self.player, self)
        self.current_state = GameState.PLAYING

    def _start_training(self):
        if not self.controller or isinstance(self.controller, PlayerController):
            print("请选择一个算法进行训练！")
            return
        if isinstance(self.controller, AStarController):
            print("A*算法无需训练！")
            self.training_completed = True
            return
        if not self.training_in_progress:
            self.is_running = True
            self.test_completed = False
            self.current_episode = 0
            self.training_in_progress = True
            self.total_training_steps = 200
            self.training_controller = self.controller
            self.training_episode = 0
            self.training_step_count = 0
            self.training_episode_reward = 0
            self.training_controller.is_training = True
            self.training_controller.reset_path()
        self.current_state = GameState.PLAYING

    def _stop_training(self):
        if self.training_in_progress:
            self.is_running = False
            self.training_in_progress = False
            if self.current_state == GameState.PLAYING and self.training_controller:
                self.training_controller.is_training = False
                self.training_controller._generate_training_charts()
                print("训练已停止！")
                self.training_controller = None
            elif self.current_state == GameState.PERFORMANCE_ANALYSIS:
                for algo_name, controller in self.controllers:
                    if algo_name != "AStar":
                        controller.is_training = False
                        controller._generate_training_charts()
                print("性能对比训练已停止！")
                self.controllers = []
                self.selected_algorithms = []

    def _run_training_step(self):
        if not self.is_running or self.training_episode >= 200:
            self._stop_training()
            self.training_completed = True
            return

        algo_name = self.training_controller.__class__.__name__.replace("Controller", "")
        patience = 20
        best_avg_reward = -float('inf')
        no_improve_count = 0
        recent_rewards = deque(maxlen=patience)

        if self.training_step_count == 0:
            self.training_episode += 1
            self.current_episode = self.training_episode
            self.training_episode_reward = 0
            self.training_controller.reset_path()

        self.training_controller.update(None)
        self.training_step_count += 1
        if hasattr(self.training_controller, 'reward_history') and self.training_controller.reward_history:
            self.training_episode_reward += self.training_controller.reward_history[-1]

        if self.training_controller.is_success or self.training_step_count >= 200:
            self.training_progress = self.training_episode / self.total_training_steps * 100
            recent_rewards.append(self.training_episode_reward)
            if len(recent_rewards) == patience:
                avg_reward = sum(recent_rewards) / patience
                if avg_reward > best_avg_reward + 0.01:
                    best_avg_reward = avg_reward
                    no_improve_count = 0
                    if hasattr(self.training_controller, '_save_model'):
                        self.training_controller._save_model()
                else:
                    no_improve_count += 1
                if no_improve_count >= patience:
                    self._stop_training()
                    self.training_completed = True
                    return

            self.training_controller._save_training_data(self.training_episode, self.training_episode_reward)
            self.training_controller.is_training = False
            self.training_step_count = 0
            self.training_controller.is_training = True

    def _start_testing(self):
        if not self.training_completed and not isinstance(self.controller, AStarController):
            print("请先完成训练！")
            return
        self.is_running = True
        self.test_completed = False
        if self.controller:
            self.controller.is_success = False
            if isinstance(self.controller, QLearningController):
                self.controller.epsilon = 0.01
            self.controller.player.reset()
            if hasattr(self.controller, 'reset_path'):
                self.controller.reset_path()
        self.controllers = []
        algo_name = self.controller.__class__.__name__.replace("Controller", "")
        if algo_name != "Player":
            self._create_controller(algo_name)
        if algo_name != "AStar":
            self._create_controller("AStar")
        self.current_state = GameState.PLAYING

    def _stop_testing(self):
        self.is_running = False
        self.test_completed = False

    def _reset_testing(self):
        if self.controller:
            self.controller.is_success = False
            self.controller.player.reset()
            if hasattr(self.controller, 'reset_path'):
                self.controller.reset_path()
        for _, controller in self.controllers:
            controller.is_success = False
            controller.player.reset()
            if hasattr(controller, 'reset_path'):
                controller.reset_path()
        self.is_running = False
        self.test_completed = False

    def _start_performance_comparison(self):
        self.current_state = GameState.PERFORMANCE_ANALYSIS
        self.selected_algorithms = []
        self.controllers = []

    def _start_comparison_training(self):
        if not self.selected_algorithms:
            print("请至少选择一种算法！")
            return
        if not self.training_in_progress:
            self.is_running = True
            self.test_completed = False
            self.current_episode = 0
            self.training_in_progress = True
            self.total_training_steps = len([algo for algo in self.selected_algorithms if algo != "AStar"]) * 200
            self.training_controller = None
            self.training_episode = 0
            self.training_step_count = 0
            self.training_episode_reward = 0
            self.controllers = []
            for algo_name in self.selected_algorithms:
                self._create_controller(algo_name)
                if algo_name != "AStar":
                    controller = self.controllers[-1][1]
                    controller.is_training = True
                    controller.reset_path()
        self.current_state = GameState.PERFORMANCE_ANALYSIS

    def _run_comparison_training_step(self):
        if not self.is_running or not self.controllers:
            self._stop_training()
            self.training_completed = True
            return

        for algo_name, controller in self.controllers:
            if algo_name == "AStar":
                continue
            if not controller.is_training:
                continue

            patience = 20
            best_avg_reward = -float('inf')
            no_improve_count = 0
            recent_rewards = deque(maxlen=patience)

            if self.training_step_count == 0:
                self.training_episode += 1
                self.current_episode = self.training_episode
                self.training_episode_reward = 0
                controller.reset_path()

            controller.update(None)
            self.training_step_count += 1
            if hasattr(controller, 'reward_history') and controller.reward_history:
                self.training_episode_reward += controller.reward_history[-1]

            if controller.is_success or self.training_step_count >= 200:
                self.training_progress = (self.training_episode / self.total_training_steps) * 100
                recent_rewards.append(self.training_episode_reward)
                controller._save_training_data(self.training_episode, self.training_episode_reward)
                if len(recent_rewards) == patience:
                    avg_reward = sum(recent_rewards) / patience
                    if avg_reward > best_avg_reward + 0.01:
                        best_avg_reward = avg_reward
                        no_improve_count = 0
                        if hasattr(controller, '_save_model'):
                            controller._save_model()
                    else:
                        no_improve_count += 1
                    if no_improve_count >= patience:
                        controller.is_training = False
                        if all(not c[1].is_training for c in self.controllers if c[0] != "AStar"):
                            for algo_name, controller in self.controllers:
                                if algo_name != "AStar":
                                    controller._generate_training_charts()
                            self._stop_training()
                            self.training_completed = True
                            return

                self.training_step_count = 0
                controller.reset_path()

    def _start_comparison_testing(self):
        if not self.training_completed:
            print("请先完成训练！")
            return
        self.is_running = True
        self.test_completed = False
        self.controllers = []
        for algo_name in self.selected_algorithms:
            self._create_controller(algo_name)
            if algo_name == "QLearning":
                for _, controller in self.controllers:
                    if isinstance(controller, QLearningController):
                        controller.epsilon = 0.01
        if "AStar" not in self.selected_algorithms:
            self._create_controller("AStar")
        for _, controller in self.controllers:
            controller.is_success = False
            controller.player.reset()
            if hasattr(controller, 'reset_path'):
                controller.reset_path()
        self.current_state = GameState.PLAYING

    def _regenerate_maze(self):
        self._update_maze()
        self.training_completed = False
        self.is_running = False
        self.test_completed = False

    def _return_to_menu(self):
        self.current_state = GameState.MENU
        self.controller = None
        self.controllers = []
        self.selected_algorithms = []
        self.training_completed = False
        self.is_running = False
        self.test_completed = False
        self.current_episode = 0
        self.training_in_progress = False
        self.training_controller = None

    def _start_game(self):
        self.maze.reset_maze()
        self.player.reset()
        self.current_state = GameState.PLAYING
        self.start_time = time.time()

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
            elif self.current_state == GameState.PERFORMANCE_ANALYSIS:
                self._draw_comparison_selection()
            else:
                if self.is_running and not self.test_completed:
                    if self.controller and not self.training_in_progress:
                        self.controller.update(None)
                        if self.controller.is_success:
                            self.is_running = False
                            self.test_completed = True
                    for _, controller in self.controllers:
                        if not self.training_in_progress or controller.is_training:
                            controller.update(None)
                        if controller.is_success:
                            self.is_running = False
                            self.test_completed = True
                if self.training_in_progress:
                    if self.current_state == GameState.PLAYING and self.training_controller:
                        self._run_training_step()
                    elif self.current_state == GameState.PERFORMANCE_ANALYSIS:
                        self._run_comparison_training_step()
                self._draw_game()
                self._draw_control_panel()
            if self.training_in_progress:
                self._draw_training_progress()
            pygame.display.flip()
            self.clock.tick(30)

    def _handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.current_state == GameState.MENU:
                for btn in self.menu_buttons:
                    btn.handle_event(event)
            elif self.current_state == GameState.PERFORMANCE_ANALYSIS:
                for btn in self.comparison_buttons:
                    btn.handle_event(event)
            else:
                for btn in self.game_buttons:
                    btn.handle_event(event)

    def _draw_menu(self):
        title_font = pygame.font.SysFont("SimHei", 48)
        title_surf = title_font.render("迷宫路径探索逻辑研究系统", True, (255, 255, 255))
        title_rect = title_surf.get_rect(center=(600, 50))
        self.screen.blit(title_surf, title_rect)
        for btn in self.menu_buttons:
            btn.draw(self.screen)

    def _draw_comparison_selection(self):
        title_font = pygame.font.SysFont("SimHei", 36)
        title_surf = title_font.render("选择算法进行对比", True, (255, 255, 255))
        title_rect = title_surf.get_rect(center=(600, 50))
        self.screen.blit(title_surf, title_rect)
        for btn in self.comparison_buttons:
            btn.draw(self.screen)
        font = pygame.font.SysFont("MicrosoftYaHei", 20)
        selected_text = f"已选算法: {', '.join(self.selected_algorithms) if self.selected_algorithms else '无'}"
        text_surf = font.render(selected_text, True, (255, 255, 255))
        self.screen.blit(text_surf, (400, 70))

    def _draw_game(self):
        self.maze.draw(self.screen)
        if self.controller and hasattr(self.controller, 'path') and self.controller.path:
            algo_name = self.controller.__class__.__name__.replace("Controller", "")
            if not getattr(self.controller, 'is_training', False):
                for i in range(len(self.controller.path) - 1):
                    start = (
                        self.controller.path[i][0] * self.maze.cell_size + self.maze.cell_size // 2,
                        self.controller.path[i][1] * self.maze.cell_size + self.maze.cell_size // 2
                    )
                    end = (
                        self.controller.path[i + 1][0] * self.maze.cell_size + self.maze.cell_size // 2,
                        self.controller.path[i + 1][1] * self.maze.cell_size + self.maze.cell_size // 2
                    )
                    pygame.draw.line(self.screen, self.path_colors.get(algo_name, (255, 0, 0)), start, end, 4)
        for algo_name, controller in self.controllers:
            if hasattr(controller, 'path') and controller.path:
                if not getattr(controller, 'is_training', False):
                    for i in range(len(controller.path) - 1):
                        start = (
                            controller.path[i][0] * self.maze.cell_size + self.maze.cell_size // 2,
                            controller.path[i][1] * self.maze.cell_size + self.maze.cell_size // 2
                        )
                        end = (
                            controller.path[i + 1][0] * self.maze.cell_size + self.maze.cell_size // 2,
                            controller.path[i + 1][1] * self.maze.cell_size + self.maze.cell_size // 2
                        )
                        pygame.draw.line(self.screen, self.path_colors.get(algo_name, (255, 0, 0)), start, end, 4)
                    controller.player.draw(self.screen, self.path_colors.get(algo_name, (255, 255, 0)))

    def _draw_control_panel(self):
        pygame.draw.rect(self.screen, (60, 60, 60), (800, 0, 400, 800))
        for btn in self.game_buttons:
            btn.draw(self.screen)
        font = pygame.font.SysFont("MicrosoftYaHei", 20)
        y_offset = 600
        if self.controller:
            mode_name = self.controller.__class__.__name__.replace("Controller", "")
            step_count = self.controller.step_count if hasattr(self.controller, "step_count") else 0
            status = "已完成" if self.controller.is_success else "运行中"
            text = f"{mode_name} 步数: {step_count} ({status})"
            text_surf = font.render(text, True, self.path_colors.get(mode_name, (255, 255, 255)))
            self.screen.blit(text_surf, (810, y_offset))
            y_offset += 30
        for algo_name, controller in self.controllers:
            step_count = controller.step_count if hasattr(controller, "step_count") else 0
            status = "已完成" if controller.is_success else "运行中"
            text = f"{algo_name} 步数: {step_count} ({status})"
            text_surf = font.render(text, True, self.path_colors.get(algo_name, (255, 255, 255)))
            self.screen.blit(text_surf, (810, y_offset))
            y_offset += 30
        if self.test_completed:
            text_surf = font.render("测试完成！", True, (255, 255, 0))
            self.screen.blit(text_surf, (810, y_offset))

    def _draw_training_progress(self):
        font = pygame.font.SysFont("MicrosoftYaHei", 20)
        episode_text = font.render(f"轮次: {self.current_episode}/200", True, (255, 255, 255))
        self.screen.blit(episode_text, (810, 540))
        progress_text = font.render(f"训练进度: {self.training_progress:.2f}%", True, (255, 255, 255))
        self.screen.blit(progress_text, (810, 570))
        progress_bar_width = 180
        progress_bar_height = 20
        progress_x = 810
        progress_y = 600
        pygame.draw.rect(self.screen, (60, 60, 60),
                         (progress_x, progress_y, progress_bar_width, progress_bar_height))
        pygame.draw.rect(self.screen, (0, 255, 0), (
            progress_x, progress_y, progress_bar_width * self.training_progress / 100, progress_bar_height),
                         border_radius=5)

if __name__ == "__main__":
    game = GameManager()
    try:
        game.run()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        pygame.quit()
        sys.exit()