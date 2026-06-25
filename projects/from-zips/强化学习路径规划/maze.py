# game/maze.py
import numpy as np
import pygame
import random
from collections import deque
from enum import Enum


class Maze:
    def __init__(self, screen_width, screen_height, grid_cols, grid_rows):
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        # 动态计算单元格大小（保持比例）
        self.cell_size = min(
            screen_width // grid_cols,
            screen_height // grid_rows
        )
        self.maze_width = self.grid_cols * self.cell_size
        self.maze_height = self.grid_rows * self.cell_size
        # 起点和终点（网格坐标）
        self.start_pos = (1, 1)
        self.end_pos = (self.grid_cols - 2, self.grid_rows - 2)

        # 迷宫数据结构（二维数组）
        self.maze_layout = []
        self._generate_valid_maze()  # 生成有效迷宫
        self.score_map = np.zeros((self.grid_rows, self.grid_cols), dtype=int)
        self._initialize_score_map()

    def _generate_valid_maze(self):
        """生成迷宫并确保起点到终点有通路"""
        self._prim_generate()
        # 确保起点和终点畅通
        self.maze_layout[self.start_pos[1]][self.start_pos[0]] = 0
        self.maze_layout[self.end_pos[1]][self.end_pos[0]] = 0
        # 验证迷宫是否可达
        if not self._validate_maze():
            self._ensure_path()

    def _prim_generate(self):
        """Prim算法生成迷宫核心逻辑"""
        self.maze_layout = [[1 for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]
        walls = []
        start_x, start_y = self.start_pos
        self.maze_layout[start_y][start_x] = 0

        # 初始化边缘墙
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = start_x + dx, start_y + dy
            if 0 <= nx < self.grid_cols and 0 <= ny < self.grid_rows:
                walls.append((nx, ny, start_x, start_y))

        while walls:
            # 随机选择一堵墙
            wall_idx = random.randint(0, len(walls) - 1)
            wx, wy, src_x, src_y = walls.pop(wall_idx)

            # 计算对面单元格
            dx = wx - src_x
            dy = wy - src_y
            opp_x = wx + dx
            opp_y = wy + dy

            if 0 <= opp_x < self.grid_cols and 0 <= opp_y < self.grid_rows:
                if self.maze_layout[opp_y][opp_x] == 1:
                    # 打通当前墙和对面单元格
                    self.maze_layout[wy][wx] = 0
                    self.maze_layout[opp_y][opp_x] = 0

                    # 添加新的边缘墙
                    for ddx, ddy in directions:
                        nx = opp_x + ddx
                        ny = opp_y + ddy
                        if 0 <= nx < self.grid_cols and 0 <= ny < self.grid_rows:
                            if self.maze_layout[ny][nx] == 1:
                                walls.append((nx, ny, opp_x, opp_y))

        # 确保起点终点畅通
        self.maze_layout[self.start_pos[1]][self.start_pos[0]] = 0
        self.maze_layout[self.end_pos[1]][self.end_pos[0]] = 0

    def _validate_maze(self):
        """BFS验证迷宫是否可达"""
        visited = [[False for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]
        queue = deque([self.start_pos])
        visited[self.start_pos[1]][self.start_pos[0]] = True
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            x, y = queue.popleft()
            if (x, y) == self.end_pos:
                return True
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_cols and 0 <= ny < self.grid_rows:
                    if not visited[ny][nx] and self.maze_layout[ny][nx] == 0:
                        visited[ny][nx] = True
                        queue.append((nx, ny))
        return False

    def _ensure_path(self):
        """确保起点到终点有路径"""
        from collections import deque

        # 使用BFS寻找路径
        queue = deque([self.start_pos])
        visited = {self.start_pos}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        came_from = {}
        found = False

        while queue and not found:
            current = queue.popleft()

            for dx, dy in directions:
                nx = current[0] + dx
                ny = current[1] + dy

                if 0 <= nx < self.grid_cols and 0 <= ny < self.grid_rows:
                    if (nx, ny) == self.end_pos:
                        came_from[self.end_pos] = current
                        found = True
                        break
                    if self.maze_layout[ny][nx] == 1 or (nx, ny) in visited:
                        continue

                    visited.add((nx, ny))
                    queue.append((nx, ny))
                    came_from[(nx, ny)] = current

            if found:
                # 回溯并打通路径
                current = self.end_pos
                while current != self.start_pos:
                    x, y = current
                    self.maze_layout[y][x] = 0
                    current = came_from[current]
                break

        # 如果BFS失败，强制打通直线路径
        if not found:
            current = list(self.start_pos)
            end = list(self.end_pos)

            # 横向移动优先
            while current[0] != end[0]:
                step = 1 if current[0] < end[0] else -1
                current[0] += step
                self.maze_layout[current[1]][current[0]] = 0

            # 纵向移动
            while current[1] != end[1]:
                step = 1 if current[1] < end[1] else -1
                current[1] += step
                self.maze_layout[current[1]][current[0]] = 0

    def reset_maze(self):
        """重置迷宫布局"""
        self._generate_valid_maze()
        self._validate_maze()
        self._ensure_path()

    def is_wall(self, x, y, radius=0):
        # 确保cell_size动态更新
        grid_x = int(x // self.cell_size)
        grid_y = int(y // self.cell_size)

        # 精确到浮点数坐标检测
        for dy_offset in [-1, 0, 1]:
            for dx_offset in [-1, 0, 1]:
                check_x = grid_x + dx_offset
                check_y = grid_y + dy_offset
                if 0 <= check_x < self.grid_cols and 0 <= check_y < self.grid_rows:
                    if self.maze_layout[check_y][check_x] == 1:
                        # 计算到墙壁边界的精确距离
                        wall_left = check_x * self.cell_size
                        wall_right = (check_x + 1) * self.cell_size
                        wall_top = check_y * self.cell_size
                        wall_bottom = (check_y + 1) * self.cell_size

                        closest_x = max(wall_left, min(x, wall_right))
                        closest_y = max(wall_top, min(y, wall_bottom))
                        distance_sq = (x - closest_x) ** 2 + (y - closest_y) ** 2

                        if distance_sq < (radius + 2) ** 2:
                            return True
        return False

    def draw(self, surface):
        """绘制迷宫"""
        pygame.draw.rect(surface, (40, 40, 40), (0, 0, self.maze_width, self.maze_height))
        # 绘制背景
        for y in range(self.grid_rows):
            for x in range(self.grid_cols):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                if rect.bottom > 600:  # 屏幕高度限制
                    continue

                if self.maze_layout[y][x] == 1:
                    pygame.draw.rect(surface, (100, 100, 120), rect)
                else:
                    pygame.draw.rect(surface, (40, 40, 40), rect)

                # 绘制起点
                if (x, y) == self.start_pos:
                    center = (
                        x * self.cell_size + self.cell_size // 2,
                        y * self.cell_size + self.cell_size // 2
                    )
                    pygame.draw.circle(surface, (0, 255, 0), center, self.cell_size // 3)

                    # 动态绘制终点（确保在可见区域）
                end_rect = pygame.Rect(
                    self.end_pos[0] * self.cell_size,
                    self.end_pos[1] * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                if end_rect.bottom <= 600:  # 屏幕底部边界检查
                    center = (
                        self.end_pos[0] * self.cell_size + self.cell_size // 2,
                        self.end_pos[1] * self.cell_size + self.cell_size // 2
                    )
                    pygame.draw.circle(surface, (255, 0, 0), center, self.cell_size // 3)

    def _initialize_score_map(self):
        # 为每个单元格设置分数，可以根据位置或难度设置不同的分数
        for y in range(self.grid_rows):
            for x in range(self.grid_cols):
                if self.maze_layout[y][x] == 0:  # 只为通道设置分数
                    self.score_map[y][x] = 10  # 基础分数

    def draw_path(self, surface, path):
        """绘制路径"""
        if not path:
            return
        for i in range(len(path) - 1):
            start = (
                path[i][0] * self.cell_size + self.cell_size // 2,
                path[i][1] * self.cell_size + self.cell_size // 2
            )
            end = (
                path[i + 1][0] * self.cell_size + self.cell_size // 2,
                path[i + 1][1] * self.cell_size + self.cell_size // 2
            )
            pygame.draw.line(surface, (0, 255, 255), start, end, 4)

    @property
    def end_pixel_pos(self):
        """获取终点的像素坐标"""
        return (
            self.end_pos[0] * self.cell_size + self.cell_size // 2,
            self.end_pos[1] * self.cell_size + self.cell_size // 2
        )




















