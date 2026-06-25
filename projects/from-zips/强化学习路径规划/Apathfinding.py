# ai/Apathfinding.py

import heapq
import time
import pygame

from state import GameState


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

class AStar:
    def __init__(self, maze, player, game_manager):
        self.maze = maze
        self.player = player
        self.game_manager = game_manager
        self.path = []
        self.path_time = 0.0
        self.is_success = False
        self.start_time = time.time()
        self._find_path()  # 初始化时立即生成路径
        self.step_interval = 0.15  # 移动间隔
        self.last_move_time = time.time()

    def _find_path(self):
        """A*寻路核心逻辑，生成合法路径"""
        # 动态获取起点（玩家当前位置）和终点
        start = (
            int(self.player.x // self.maze.cell_size),
            int(self.player.y // self.maze.cell_size)
        )
        end = self.maze.end_pos

        open_heap = []
        closed_set = set()
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, end)}
        heapq.heappush(open_heap, (f_score[start], start))

        cols = self.maze.grid_cols
        rows = self.maze.grid_rows

        while open_heap:
            current = heapq.heappop(open_heap)[1]
            if current == end:
                self.path = self._reconstruct_path(came_from, end)
                self._validate_path()  # 增加路径验证
                return

            closed_set.add(current)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)

                # 严格边界和墙壁检查
                if not (0 <= neighbor[0] < cols and 0 <= neighbor[1] < rows):
                    continue
                if self.maze.maze_layout[neighbor[1]][neighbor[0]] == 1:
                    continue

                tentative_g = g_score[current] + 1
                if neighbor in closed_set and tentative_g >= g_score.get(neighbor, float('inf')):
                    continue

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, end)
                    heapq.heappush(open_heap, (f_score[neighbor], neighbor))

        self.path = []  # 未找到路径时置空
        self._validate_path()  # 增加路径验证

    def _validate_path(self):
        """确保路径不穿墙"""
        valid_path = []
        for cell in self.path:
            x, y = cell
            if self.maze.maze_layout[y][x] == 1:
                break  # 发现墙壁则截断路径
            valid_path.append(cell)
        self.path = valid_path

    def _reconstruct_path(self, came_from, end):
        """重建路径并确保顺序正确"""
        path = [end]
        current = end
        max_steps = self.maze.grid_cols * self.maze.grid_rows  # 防止无限循环

        while current != self.maze.start_pos and max_steps > 0:
            if current not in came_from:
                # 路径断裂时自动补全直线路径
                dx = current[0] - self.maze.start_pos[0]
                dy = current[1] - self.maze.start_pos[1]
                next_x = current[0] - (1 if dx > 0 else -1 if dx < 0 else 0)
                next_y = current[1] - (1 if dy > 0 else -1 if dy < 0 else 0)
                current = (next_x, next_y)
            else:
                current = came_from[current]

            path.append(current)
            max_steps -= 1

        return path[::-1]  # 反转路径为起点到终点

    def reset_path(self):
        """重置并重新寻路"""
        self.path = []
        self.is_success = False
        self._find_path()  # 强制重新生成路径
        # 确保玩家回到起点
        self.player.x = self.maze.start_pos[0] * self.maze.cell_size + self.maze.cell_size // 2
        self.player.y = self.maze.start_pos[1] * self.maze.cell_size + self.maze.cell_size // 2

    def update(self, events=None):
        """移动逻辑（包含路径空时自动重新寻路）"""
        if self.is_success or not self.path:
            if not self.path:  # 路径为空时重新寻路
                self._find_path()
            return

        current_time = time.time()
        if current_time - self.last_move_time < self.step_interval:
            return

        # 移动逻辑（严格碰撞检测）
        target_cell = self.path.pop(0)
        target_x = target_cell[0] * self.maze.cell_size + self.maze.cell_size // 2
        target_y = target_cell[1] * self.maze.cell_size + self.maze.cell_size // 2

        # 最终网格碰撞校验
        grid_x = int(target_x // self.maze.cell_size)
        grid_y = int(target_y // self.maze.cell_size)
        if self.maze.maze_layout[grid_y][grid_x] == 1:
            self.path = []  # 立即终止非法移动
            return

        self.player.x = target_x
        self.player.y = target_y
        self.last_move_time = current_time

        # 成功检测
        if (self.player.x, self.player.y) == self.maze.end_pixel_pos:
            self.is_success = True
            self._record_success()

    def _record_success(self):
        """记录成功状态"""
        self.path_time = time.time() - self.start_time
        self.is_success = True
        self.game_manager.current_state = GameState.SUCCESS
        self.game_manager.success_time = self.path_time




