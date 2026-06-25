import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
from scipy.spatial import distance
import pandas as pd
import matplotlib.pyplot as plt
import os
import heapq
import traceback
import math
import shutil

try:
    import ezdxf
    from ezdxf.math import Vec3
except ImportError:
    print("错误: 未找到 ezdxf 库。")
    print("请使用 'pip install ezdxf' 命令安装。")
    exit()

try:
    import imageio.v2 as imageio
except ImportError:
    print("错误: 未找到 imageio 库。")
    print("请使用 'pip install imageio' 命令安装 (用于生成视频)。")
    exit()

try:
    from tqdm import tqdm
except ImportError:
    print("警告: 未找到 tqdm 库。进度条将不会显示。")
    print("请使用 'pip install tqdm' 命令安装。")


    class tqdm:
        def __init__(self, total=None, desc=None):
            self.total = total
            self.desc = desc
            self.n = 0
            if self.desc:
                print(self.desc)

        def update(self, n=1):
            self.n += n
            if self.total:
                print(f"  ... 进度: {self.n}/{self.total}", end='\r')
            else:
                print(f"  ... 进度: {self.n}", end='\r')

        def close(self):
            print("\n进度完成。")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.close()

# [NEW-SPEED] 导入 numba JIT 编译器
try:
    import numba
except ImportError:
    print("警告: 未找到 numba 库。射线投射将会很慢。")
    print("请使用 'pip install numba' 命令安装。")


    # 创建一个虚拟的 @numba.jit 装饰器，使代码在没有numba时也能运行
    def jit_decorator(func=None, nopython=False):
        if func:
            return func
        return lambda f: f


    class numba_jit_obj:
        def __call__(self, *args, **kwargs):
            return jit_decorator


    numba = numba_jit_obj()
    numba.jit = jit_decorator

# --- 设置 ---
if not torch.cuda.is_available():
    print("警告: CUDA 不可用。训练将在 CPU 上运行。")
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda")
    print(f"CUDA 可用: {torch.cuda.get_device_name(0)}")

PPO_DIR = "ppo_v4_dxf_output"
if not os.path.exists(PPO_DIR):
    os.makedirs(PPO_DIR)

VIDEO_FRAMES_DIR = os.path.join(PPO_DIR, "video_frames")

# [NEW-SPEED] 视频生成开关。设为 False 可以跳过第一轮的视频渲染，加快启动速度。
CREATE_VIDEO_EPISODE_0 = True


# --- [MODIFIED] 蓝图步骤 1 & 2: 重构 MapConfig ---
class MapConfig:
    SCALE = 1.0
    BOUNDS = [0, 100, 0, 100]

    # [MODIFIED] A*网格大小 (50mm)
    ASTAR_GRID_SIZE = 50.0

    # --- [需求: 环境空间-2 (动态环境)] ---
    # 描述：能见度和速度随时间动态变化
    # 实现：定义以下两个列表，供 PathPlanningEnv 调用
    SPEED_LEVELS = [
        (40, 2.8), (90, 2.5), (180, 1.9), (280, 1.0), (360, 0.6)
    ]
    VISION_LEVELS = [
        (40, 30.0), (90, 10.0), (180, 5.0), (280, 2.0), (360, 0.5)
    ]
    # --- [需求: 环境空间-3 (最大疏散时间)] ---
    MAX_SIMULATION_TIME = 360

    def __init__(self):
        # --- [需求: 环境空间-1 (数据分层)] ---
        # 描述：将环境物体分为不同性质 (危险/引导/阻碍) 和不同维度 (宏观/微观)
        # 实现：创建不同的列表来存储从DXF加载的实体

        self.START_POINTS = []
        self.ZONES_EXITS = []  # [REQ-ENV-1: 引导性 (出口)]
        self.ZONES_HAZARD = []  # [REQ-ENV-1: 危险性]

        # [MODIFIED] v4.12: 细分引导标识为四种
        self.GUIDANCE_UP = []  # 红色 (ACI 1)
        self.GUIDANCE_DOWN = []  # 绿色 (ACI 3, 但在文件中为 4)
        self.GUIDANCE_RIGHT = []  # 蓝色 (ACI 5)
        self.GUIDANCE_LEFT = []  # 粉色 (ACI 6)

        # [REQ-ENV-1: 物理维度 - 宏观] (A*规划器使用)
        self.MACRO_OBSTACLES = []

        # [REQ-ENV-1: 感知维度 - 视觉阻挡] (射线在此停止)
        self.MICRO_OBSTACLES_VISION = []
        # [REQ-ENV-1: 感知维度 - 移动阻挡] (射线可穿透)
        self.MICRO_OBSTACLES_MOVEMENT = []

        # DXF图层到内部列表的映射
        # [REQ-ENV-1: 实现方式]
        # [MODIFIED] v5: 基于用户反馈，不再使用 "引导-X" 图层。
        # 而是使用单一图层 "引导标识"，并通过颜色来区分。
        self.layer_mapping = {
            "遮挡视线": self.MICRO_OBSTACLES_VISION,  # (墙壁/书架)
            "不遮挡": self.MICRO_OBSTACLES_MOVEMENT,  # (桌椅)
            "危险源": self.ZONES_HAZARD,
            "地图边界": self.MACRO_OBSTACLES,

            # [NEW] v5: 使用一个特殊的键，在加载循环中单独处理
            "引导标识": "GUIDANCE_LAYER_BY_COLOR"
        }

    def load_from_dxf(self, filepath):
        print(f"正在从 {filepath} 加载地图...")
        try:
            doc = ezdxf.readfile(filepath)
        except IOError:
            print(f"错误: 无法读取DXF文件: {filepath}")
            return
        except ezdxf.DXFStructureError:
            print("错误: DXF文件结构无效。")
            return

        msp = doc.modelspace()

        # [NEW-DIAGNOSTIC] 打印所有图层名称
        all_layers = [layer.dxf.name for layer in doc.layers]
        print(f"  检测到的图层: {all_layers}")

        all_points = []
        try:
            for entity in msp:
                if entity.dxftype() == 'LINE':
                    all_points.append(entity.dxf.start)
                    all_points.append(entity.dxf.end)
                elif entity.dxftype() == 'LWPOLYLINE':
                    for p in entity.vertices:
                        all_points.append(p)
                elif entity.dxftype() == 'CIRCLE':
                    center = entity.dxf.center
                    radius = entity.dxf.radius
                    all_points.append((center.x - radius, center.y - radius))
                    all_points.append((center.x + radius, center.y + radius))

            if not all_points:
                print("  警告: DXF 模型空间为空，使用默认边界。")
                return

            min_x = min(p[0] for p in all_points)
            max_x = max(p[0] for p in all_points)
            min_y = min(p[1] for p in all_points)
            max_y = max(p[1] for p in all_points)

            padding_x = (max_x - min_x) * 0.05
            padding_y = (max_y - min_y) * 0.05

            self.BOUNDS = [min_x - padding_x, max_x + padding_x, min_y - padding_y, max_y + padding_y]
            print(
                f"  地图边界检测: X({self.BOUNDS[0]:.2f} - {self.BOUNDS[1]:.2f}), Y({self.BOUNDS[2]:.2f} - {self.BOUNDS[3]:.2f})")

        except Exception as e:
            print(f"  警告: 无法自动检测DXF边界 ({e})，使用默认值。")

        entity_count = 0
        for entity in msp:
            layer = entity.dxf.layer

            if layer in self.layer_mapping:
                target_list_or_key = self.layer_mapping[layer]
                entity_type = entity.dxftype()
                color = entity.dxf.color  # 获取 AutoCAD 颜色索引 (ACI)

                # --- [NEW] v5: 特殊处理 "引导标识" 图层 ---
                if target_list_or_key == "GUIDANCE_LAYER_BY_COLOR":
                    target_list = None

                    # [MODIFIED] v5.1: 根据 check_dxf_layers.py 的输出更新颜色映射
                    # 1 = Red (上)
                    # 6 = Magenta/Pink (左)
                    # 4 = Cyan (假设这是用户所说的 "绿色") -> 下
                    # 5 = Blue (用户所说的 "蓝色", 尽管文件中没有) -> 右
                    # 256 = ByLayer (忽略)

                    if color == 1:
                        target_list = self.GUIDANCE_UP
                    elif color == 256:  # [MODIFIED] 从 3 改为 4 (青色)
                        target_list = self.GUIDANCE_DOWN
                    elif color == 4:
                        target_list = self.GUIDANCE_RIGHT
                    elif color == 6:
                        target_list = self.GUIDANCE_LEFT
                    else:
                        # 忽略此图层上任何其他颜色（如 ACI=256 ByLayer）
                        pass

                    if target_list is not None:
                        # 确保只处理线段，忽略文字和圆形
                        if entity_type == 'LINE':
                            p1 = [entity.dxf.start.x, entity.dxf.start.y]
                            p2 = [entity.dxf.end.x, entity.dxf.end.y]
                            target_list.append([p1, p2])
                            entity_count += 1
                        elif entity_type == 'LWPOLYLINE':
                            points = list(entity.vertices)
                            if not points:
                                continue

                            is_closed = entity.is_closed
                            for i in range(len(points) - 1):
                                p1 = [points[i][0], points[i][1]]
                                p2 = [points[i + 1][0], points[i + 1][1]]
                                target_list.append([p1, p2])
                                entity_count += 1

                            if is_closed and len(points) > 1:
                                p1 = [points[-1][0], points[-1][1]]
                                p2 = [points[0][0], points[0][1]]
                                target_list.append([p1, p2])
                                entity_count += 1

                        # 忽略此图层上的 CIRCLE, TEXT, MTEXT 等

                # --- 其他图层的原始逻辑 ---
                else:
                    target_list = target_list_or_key  # 它是一个列表

                    if entity_type == 'LINE':
                        p1 = [entity.dxf.start.x, entity.dxf.start.y]
                        p2 = [entity.dxf.end.x, entity.dxf.end.y]
                        target_list.append([p1, p2])
                        entity_count += 1

                        # [REQ-ENV-1: 宏观障碍实现]
                        if layer == "遮挡视线" or layer == "地图边界":
                            if [p1, p2] not in self.MACRO_OBSTACLES:
                                self.MACRO_OBSTACLES.append([p1, p2])

                    elif entity_type == 'LWPOLYLINE':
                        points = list(entity.vertices)
                        if not points:
                            continue

                        is_closed = entity.is_closed

                        for i in range(len(points) - 1):
                            p1 = [points[i][0], points[i][1]]
                            p2 = [points[i + 1][0], points[i + 1][1]]
                            target_list.append([p1, p2])
                            entity_count += 1
                            if layer == "遮挡视线" or layer == "地图边界":
                                if [p1, p2] not in self.MACRO_OBSTACLES:
                                    self.MACRO_OBSTACLES.append([p1, p2])

                        if is_closed and len(points) > 1:
                            p1 = [points[-1][0], points[-1][1]]
                            p2 = [points[0][0], points[0][1]]
                            target_list.append([p1, p2])
                            entity_count += 1
                            if layer == "遮挡视线" or layer == "地图边界":
                                if [p1, p2] not in self.MACRO_OBSTACLES:
                                    self.MACRO_OBSTACLES.append([p1, p2])

                    elif entity_type == 'CIRCLE':
                        center = np.array([entity.dxf.center.x, entity.dxf.center.y])
                        radius = entity.dxf.radius
                        if layer == "危险源":
                            target_list.append([center, radius])

                        # [FIX] v5: 确保此处的 elif 不会捕获到 "引导标识" 图层
                        # (它已经被 'GUIDANCE_LAYER_BY_COLOR' 逻辑处理)
                        elif layer.startswith("引导"):
                            # 此分支现在基本不会被触发，但作为安全保留
                            pass

                        entity_count += 1

                    # 注意: TEXT, MTEXT 等实体类型被自动忽略

        print(f"  地图加载完成。共解析 {entity_count} 个实体。")
        print(f"  宏观障碍 (A*使用): {len(self.MACRO_OBSTACLES)}")
        print(f"  视觉障碍 (射线停止): {len(self.MICRO_OBSTACLES_VISION)}")
        print(f"  移动障碍 (射线穿透): {len(self.MICRO_OBSTACLES_MOVEMENT)}")
        print(f"  危险区域 (射线穿透): {len(self.ZONES_HAZARD)}")
        # [MODIFIED] v4.12: 打印所有引导标识的计数
        print(f"  引导-上 (红 ACI 1): {len(self.GUIDANCE_UP)}")
        print(f"  引导-下 (青 ACI 4): {len(self.GUIDANCE_DOWN)}")
        print(f"  引导-右 (蓝 ACI 256): {len(self.GUIDANCE_RIGHT)}")
        print(f"  引导-左 (粉 ACI 6): {len(self.GUIDANCE_LEFT)}")

        self.precompute_obstacles_for_numba()

    def precompute_obstacles_for_numba(self):
        """将障碍物列表转换为Numba可以高效处理的Numpy数组"""

        self.nb_micro_vision_obs = np.array(self.MICRO_OBSTACLES_VISION, dtype=np.float64)
        self.nb_micro_movement_obs = np.array(self.MICRO_OBSTACLES_MOVEMENT, dtype=np.float64)
        self.nb_macro_obstacles = np.array(self.MACRO_OBSTACLES, dtype=np.float64)

        # [MODIFIED] v4.12: 预处理所有四色引导标识
        def _process_guidance(guidance_list):
            lines = []
            points = []
            for item in guidance_list:
                if isinstance(item[0], (list, np.ndarray)) and len(item[0]) == 2 and isinstance(item[1], (list,
                                                                                                          np.ndarray)):  # 线段
                    lines.append(item)
                # [FIX] v5: 移除了圆形(点)的处理逻辑，因为引导标识现在只有线段
                # elif isinstance(item[0], (list, np.ndarray)) and len(item[0]) == 2 and isinstance(item[1],
                #                                                                                   float):  # 圆形(点)
                #     points.append(item[0])

            # [FIX] v5: 确保即使 'points' 列表为空，也创建空的Numpy数组
            lines_np = np.array(lines, dtype=np.float64)
            points_np = np.array(points, dtype=np.float64)

            # 确保数组具有正确的形状，即使它们是空的
            if lines_np.ndim == 1 and lines_np.shape[0] == 0:
                lines_np = np.empty((0, 2, 2), dtype=np.float64)
            if points_np.ndim == 1 and points_np.shape[0] == 0:
                points_np = np.empty((0, 2), dtype=np.float64)

            return lines_np, points_np

        self.nb_guidance_up_lines, self.nb_guidance_up_points = _process_guidance(self.GUIDANCE_UP)
        self.nb_guidance_down_lines, self.nb_guidance_down_points = _process_guidance(self.GUIDANCE_DOWN)
        self.nb_guidance_right_lines, self.nb_guidance_right_points = _process_guidance(self.GUIDANCE_RIGHT)
        self.nb_guidance_left_lines, self.nb_guidance_left_points = _process_guidance(self.GUIDANCE_LEFT)

        if self.ZONES_HAZARD:
            centers = []
            radii = []
            for center, radius in self.ZONES_HAZARD:
                centers.append(center)
                radii.append(radius)
            self.nb_zones_hazard_centers = np.array(centers, dtype=np.float64)
            self.nb_zones_hazard_radii = np.array(radii, dtype=np.float64)
        else:
            self.nb_zones_hazard_centers = np.empty((0, 2), dtype=np.float64)
            self.nb_zones_hazard_radii = np.empty((0,), dtype=np.float64)

        print("  Numba 障碍物预处理完成。")

    # --- [MODIFIED] A* 规划器 ---


# --- [需求: 奖励空间-R2 (拓扑距离)] ---
# 描述：需要一个规划器来计算拓扑距离，作为宏观奖励。
# 实现：AStarPlanner 类。它只使用 MACRO_OBSTACLES (墙壁/边界) 来构建网格。
class AStarPlanner:
    def __init__(self, map_data, grid_size):
        self.macro_obstacles = map_data.MACRO_OBSTACLES
        self.bounds = map_data.BOUNDS
        self.grid_size = grid_size

        self.grid_width = max(1, int((self.bounds[1] - self.bounds[0]) / self.grid_size))
        self.grid_height = max(1, int((self.bounds[3] - self.bounds[2]) / self.grid_size))
        print(f"  A* 网格初始化: 尺寸=({self.grid_height}, {self.grid_width}), 网格大小={self.grid_size}")

        self.grid = self.create_grid()

    def create_grid(self):
        grid = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        for obs in self.macro_obstacles:
            p1, p2 = np.array(obs[0]), np.array(obs[1])
            x1, y1 = self.to_grid_coords(p1[0], p1[1])
            x2, y2 = self.to_grid_coords(p2[0], p2[1])
            self.mark_obstacle_line(grid, x1, y1, x2, y2)
        return grid

    def to_grid_coords(self, x, y):
        grid_x = int((x - self.bounds[0]) / self.grid_size)
        grid_y = int((y - self.bounds[2]) / self.grid_size)
        grid_x = np.clip(grid_x, 0, self.grid_width - 1)
        grid_y = np.clip(grid_y, 0, self.grid_height - 1)
        return grid_x, grid_y

    def to_world_coords(self, grid_x, grid_y):
        x = self.bounds[0] + grid_x * self.grid_size
        y = self.bounds[2] + grid_y * self.grid_size
        return x, y

    def mark_obstacle_line(self, grid, x1, y1, x2, y2):
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            if 0 <= x1 < self.grid_width and 0 <= y1 < self.grid_height:
                grid[y1, x1] = True
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

    def heuristic(self, pos, goal):
        return distance.euclidean(pos, goal)

    def get_neighbors(self, pos):
        x, y = pos
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height and not self.grid[ny, nx]:
                neighbors.append((nx, ny))
        return neighbors

    def find_path(self, start, goal):
        start_grid = self.to_grid_coords(start[0], start[1])
        goal_grid = self.to_grid_coords(goal[0], goal[1])

        if not (0 <= start_grid[0] < self.grid_width and 0 <= start_grid[1] < self.grid_height) or \
                not (0 <= goal_grid[0] < self.grid_width and 0 <= goal_grid[1] < self.grid_height):
            print(f"A* 警告: 起点 {start} 或终点 {goal} 超出地图边界 {self.bounds}")
            return None

        if self.grid[start_grid[1], start_grid[0]]:
            print(f"A* 警告: 起点 {start} 在障碍物网格 {start_grid} 中。正在搜索附近可用点...")
            q = [start_grid]
            visited = {start_grid}
            found = False
            while q:
                if not q:
                    break
                cx, cy = q.pop(0)
                if not self.grid[cy, cx]:
                    start_grid = (cx, cy)
                    print(f"  A* 信息: 找到新的有效起始网格 {start_grid}")
                    found = True
                    break

                neighbors = []
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height and (nx, ny) not in visited:
                        neighbors.append((nx, ny))
                        visited.add((nx, ny))

                for n in neighbors:
                    q.append(n)

            if not found:
                print("A* 错误: 在起点附近未找到有效的非障碍物网格。")
                return None

        if self.grid[goal_grid[1], goal_grid[0]]:
            print(f"A* 警告: 终点 {goal} 在障碍物网格 {goal_grid} 中。正在搜索附近可用点...")
            q = [goal_grid]
            visited = {goal_grid}
            found = False
            while q:
                if not q:
                    break
                cx, cy = q.pop(0)
                if not self.grid[cy, cx]:
                    goal_grid = (cx, cy)
                    print(f"  A* 信息: 找到新的有效终点网格 {goal_grid}")
                    found = True
                    break

                neighbors = []
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height and (nx, ny) not in visited:
                        neighbors.append((nx, ny))
                        visited.add((nx, ny))

                for n in neighbors:
                    q.append(n)
            if not found:
                print("A* 错误: 在终点附近未找到有效的非障碍物网格。")
                return None

        open_set = [(0, start_grid)]
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}

        while open_set:
            current_f, current = heapq.heappop(open_set)

            if current == goal_grid:
                path = []
                while current in came_from:
                    path.append(self.to_world_coords(current[0], current[1]))
                    current = came_from[current]
                path.append(self.to_world_coords(start_grid[0], start_grid[1]))
                return path[::-1]

            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + distance.euclidean(current, neighbor)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal_grid)
                    if (f_score[neighbor], neighbor) not in open_set:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        print(f"A* 警告: 未找到从 {start} 到 {goal} 的路径")
        return None

    def get_path_length(self, path):
        if not path or len(path) < 2:
            return float('inf')
        length = 0
        for i in range(len(path) - 1):
            length += distance.euclidean(path[i], path[i + 1])
        return length


# --- [MODIFIED] PPO 网络 (V2中的 BeliefStateNetwork 被移除) ---
class PPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim_continuous):
        super(PPONetwork, self).__init__()
        self.actor_continuous_mean = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim_continuous),
            nn.Tanh()
        )
        self.actor_continuous_log_std = nn.Parameter(torch.zeros(action_dim_continuous))
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        mean = self.actor_continuous_mean(state)
        std = torch.exp(self.actor_continuous_log_std.clamp(-20, 2)).expand_as(mean)
        policy_continuous = distributions.Normal(mean, std)
        value = self.critic(state)
        return policy_continuous, value


# --- [NEW-SPEED] Numba JIT 加速的辅助函数 ---

@numba.jit(nopython=True)
def _numba_ccw(A_x, A_y, B_x, B_y, C_x, C_y):
    return (C_y - A_y) * (B_x - A_x) > (B_y - A_y) * (C_x - A_x)


@numba.jit(nopython=True)
def _numba_segments_intersect(p1_x, p1_y, p2_x, p2_y, q1_x, q1_y, q2_x, q2_y):
    return _numba_ccw(p1_x, p1_y, q1_x, q1_y, q2_x, q2_y) != _numba_ccw(p2_x, p2_y, q1_x, q1_y, q2_x, q2_y) and \
        _numba_ccw(p1_x, p1_y, p2_x, p2_y, q1_x, q1_y) != _numba_ccw(p1_x, p1_y, p2_x, p2_y, q2_x, q2_y)


@numba.jit(nopython=True)
def _numba_get_ray_segment_intersection(ray_o_x, ray_o_y, ray_d_x, ray_d_y, seg_p1_x, seg_p1_y, seg_p2_x, seg_p2_y):
    p1_x, p1_y = ray_o_x, ray_o_y
    p2_x, p2_y = ray_o_x + ray_d_x * 1e6, ray_o_y + ray_d_y * 1e6
    q1_x, q1_y = seg_p1_x, seg_p1_y
    q2_x, q2_y = seg_p2_x, seg_p2_y

    if not _numba_segments_intersect(p1_x, p1_y, p2_x, p2_y, q1_x, q1_y, q2_x, q2_y):
        return -1.0

    s1_x = p2_x - p1_x
    s1_y = p2_y - p1_y
    s2_x = q2_x - q1_x
    s2_y = q2_y - q1_y

    denom = (-s2_x * s1_y + s1_x * s2_y)
    if denom == 0:
        return -1.0

    s = (-s1_y * (p1_x - q1_x) + s1_x * (p1_y - q1_y)) / denom
    t = (s2_x * (p1_y - q1_y) - s2_y * (p1_x - q1_x)) / denom

    if s >= 0 and s <= 1 and t >= 0:
        i_x = p1_x + (t * s1_x)
        i_y = p1_y + (t * s1_y)
        dist = math.sqrt((i_x - ray_o_x) ** 2 + (i_y - ray_o_y) ** 2)
        return dist

    return -1.0


@numba.jit(nopython=True)
def _numba_get_ray_circle_intersection(ray_o_x, ray_o_y, ray_d_x, ray_d_y, c_x, c_y, radius):
    oc_x = ray_o_x - c_x
    oc_y = ray_o_y - c_y

    a = ray_d_x * ray_d_x + ray_d_y * ray_d_y
    b = 2.0 * (oc_x * ray_d_x + oc_y * ray_d_y)
    c = oc_x * oc_x + oc_y * oc_y - radius ** 2

    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        return -1.0
    else:
        dist = (-b - math.sqrt(discriminant)) / (2.0 * a)
        if dist > 1e-6:
            return dist
        return -1.0


@numba.jit(nopython=True)
def _numba_get_ray_point_distance(ray_o_x, ray_o_y, ray_d_x, ray_d_y, p_x, p_y):
    """计算点到射线的投影距离（如果点在射线前方）"""
    v_x = p_x - ray_o_x
    v_y = p_y - ray_o_y

    # 投影长度 t
    t = v_x * ray_d_x + v_y * ray_d_y

    # 只有当点在射线前方时才考虑
    if t > 0:
        # 投影点
        proj_x = ray_o_x + t * ray_d_x
        proj_y = ray_o_y + t * ray_d_y

        # 点到射线的垂直距离
        perp_dist = math.sqrt((p_x - proj_x) ** 2 + (p_y - proj_y) ** 2)

        # 返回点到射线原点的距离
        return math.sqrt((p_x - ray_o_x) ** 2 + (p_y - ray_o_y) ** 2), perp_dist

    return -1.0, -1.0


# --- [MODIFIED] 蓝图步骤 3 & 4: 重构环境类 (基于V2) ---
class PathPlanningEnv:
    def __init__(self, start, goal, map_data, device):
        self.device = device
        self.map_data = map_data
        self.start = np.array(start, dtype=np.float32)
        self.agent_pos = self.start.copy()

        self.simulation_time = 0.0
        self.max_steps = 2000

        self.macro_planner = AStarPlanner(self.map_data, grid_size=self.map_data.ASTAR_GRID_SIZE)

        self.all_exits = [np.array(goal, dtype=np.float32)]
        self.current_best_exit = self.all_exits[0]

        self.macro_replan_interval = 30
        self.macro_path = []
        self.current_sub_goal = self.start.copy()
        self.prev_astar_dist = 0.0

        # [需求: 状态空间-S1/S2 (目标)]
        # 描述：智能体需要知道目标的位置和方向。
        # 实现：通过 "宏观/微观" 架构，PPO的目标被设置为 self.current_sub_goal (子目标)。
        # 子目标在A*路径上领先智能体 5000mm (5m)。
        self.sub_goal_lookahead_dist = 5000.0 * MapConfig.SCALE

        # --- [需求: 状态空间-S3 (射线配置)] ---
        # 描述：13条射线，120度视野 (每10度一条)。
        self.num_rays = 13
        self.ray_fov = np.deg2rad(120)
        # [MODIFIED] v4.12: 状态维度增加 (6 -> 9)
        # 6 = cos, sin, f_move, f_vis, f_haz
        # 9 = cos, sin, f_move, f_vis, f_haz, f_up, f_down, f_right, f_left
        self.observation_dim = (self.num_rays * 9) + 3

        self.action_dim_continuous = 2
        self.max_turn_angle = np.deg2rad(25)
        self.prev_action_continuous = np.zeros(2, dtype=np.float32)

        # --- [需求: 状态空间-S3 (加权函数)] ---
        # 描述：使用指数衰减函数计算排斥力
        # 实现：定义 A 和 B 两个参数
        self.FORCE_A = 2000.0
        self.FORCE_B = 1000.0

        self.last_ray_visuals = []

    # --- [需求: 动作空间-动态速度] ---
    # 描述：智能体的最大速度随时间(烟雾)变化。
    # 实现：此函数被 env.step() 调用，以获取当前允许的最大速度。
    def get_current_speed(self):
        """根据当前时间获取动态最大速度 (mm/s)"""
        for time_limit, speed_mps in self.map_data.SPEED_LEVELS:
            if self.simulation_time <= time_limit:
                return speed_mps * 1000
        return self.map_data.SPEED_LEVELS[-1][1] * 1000

    # --- [需求: 状态空间-S3 (动态视距)] ---
    # 描述：射线的最大长度随时间(烟雾)变化。
    # 实现：此函数被 get_observation() 调用，以获取当前的最大射线长度。
    def get_current_vision(self):
        """根据当前时间获取动态最大视距 (mm)"""
        for time_limit, vision_dist_m in self.map_data.VISION_LEVELS:
            if self.simulation_time <= time_limit:
                return vision_dist_m * 1000
        return self.map_data.VISION_LEVELS[-1][1] * 1000

    # --- [需求: 奖励空间-R2 (拓扑距离)] ---
    # 描述：基于空间拓扑路径距离的奖励。
    # 实现：此函数低频调用(每30步)，运行A*规划器，并根据两次规划的"路径长度差"返回奖励。
    def run_macro_planner(self):
        best_path_len = float('inf')
        best_path = []

        path = self.macro_planner.find_path(self.agent_pos, self.current_best_exit)
        path_len = self.macro_planner.get_path_length(path)

        # [MODIFIED] A* 失败处理
        if path_len == float('inf'):
            print(f"  [宏观规划器] A* 规划失败 (位置: {self.agent_pos})。智能体可能处于无效位置。")
            reward_R2 = -10.0  # 惩罚
            self.macro_path = []  # 清空路径
            self.current_sub_goal = self.current_best_exit  # 回退到最终目标

            fallback_path = self.macro_planner.find_path(self.start, self.current_best_exit)
            self.prev_astar_dist = self.macro_planner.get_path_length(fallback_path)
            return reward_R2

        if path_len < float('inf'):
            best_path_len = path_len
            best_path = path

        reward_R2 = 0.0
        if self.prev_astar_dist > 0 and best_path_len != float('inf'):
            # [MODIFIED] 调整奖励系数
            reward_R2 = (self.prev_astar_dist - best_path_len) * 0.005

        if best_path_len != float('inf'):
            self.prev_astar_dist = best_path_len

        self.macro_path = best_path
        self.update_sub_goal()
        return reward_R2

    # --- [需求: 状态空间-S1/S2 (子目标)] ---
    # 描述：PPO需要一个当前目标点。
    # 实现：不直接看最终出口，而是看A*路径上 5000mm 处的子目标。
    def update_sub_goal(self):
        if not self.macro_path:
            self.current_sub_goal = self.current_best_exit
            return

        agent_pos_np = np.array(self.agent_pos)
        path_points_np = np.array(self.macro_path)
        distances = np.linalg.norm(path_points_np - agent_pos_np, axis=1)
        closest_index = np.argmin(distances)

        lookahead_index = closest_index
        current_dist = 0.0
        while lookahead_index < len(self.macro_path) - 1:
            current_dist += distance.euclidean(self.macro_path[lookahead_index], self.macro_path[lookahead_index + 1])
            if current_dist >= self.sub_goal_lookahead_dist:
                break
            lookahead_index += 1

        self.current_sub_goal = self.macro_path[lookahead_index]

    # --- [需求: 状态空间-S3 (完整实现)] ---
    # 描述：整合13射线、动态视距、多层穿透、加权力和Cos/Sin向量。
    # 实现：调用下面 Numba JIT 优化的 _numba_get_observation 函数。
    def get_observation(self):
        max_length = self.get_current_vision()  # [REQ-S3: 动态视距]
        sub_goal = self.current_sub_goal

        # [NEW-SPEED] 调用 Numba JIT 编译的函数
        state, visuals = _numba_get_observation(
            self.agent_pos[0], self.agent_pos[1],
            np.linalg.norm(self.prev_action_continuous),
            self.prev_action_continuous[0], self.prev_action_continuous[1],
            sub_goal[0], sub_goal[1],
            self.ray_fov, self.num_rays,  # [REQ-S3: 13射线/120度]
            max_length,
            self.map_data.nb_micro_vision_obs,  # [REQ-S3: 多层穿透 (视觉障碍)]
            self.map_data.nb_micro_movement_obs,  # [REQ-S3: 多层穿透 (移动障碍)]
            self.map_data.nb_macro_obstacles,  # [BUG-FIX] 传入宏观障碍(边界)用于碰撞
            self.map_data.nb_zones_hazard_centers,  # [REQ-S3: 加权 Hazard]
            self.map_data.nb_zones_hazard_radii,
            # [MODIFIED] v4.12: 传入所有四色引导
            self.map_data.nb_guidance_up_lines,
            self.map_data.nb_guidance_up_points,
            self.map_data.nb_guidance_down_lines,
            self.map_data.nb_guidance_down_points,
            self.map_data.nb_guidance_right_lines,
            self.map_data.nb_guidance_right_points,
            self.map_data.nb_guidance_left_lines,
            self.map_data.nb_guidance_left_points,
            self.FORCE_A, self.FORCE_B,  # [REQ-S3: 加权函数 (A, B)]
            self.map_data.BOUNDS[1] - self.map_data.BOUNDS[0],  # [REQ-S1/S2: 距离比率]
            self.map_data.BOUNDS[3] - self.map_data.BOUNDS[2]
        )
        self.last_ray_visuals = visuals
        return state

    # --- [需求: 奖励空间 (完整实现)] ---
    def step(self, action_continuous):
        self.step_count += 1
        self.simulation_time += 1.0

        R_total = 0.0
        done = False
        reached_goal = False

        # [REQ-ACTION: 动态速度]
        current_max_speed = self.get_current_speed()

        # 转向角限制 (平滑)
        if np.linalg.norm(self.prev_action_continuous) > 1e-6:
            prev_dir = self.prev_action_continuous / np.linalg.norm(self.prev_action_continuous)
            curr_dir = action_continuous / (np.linalg.norm(action_continuous) + 1e-6)
            dot_product = np.clip(np.dot(prev_dir, curr_dir), -1.0, 1.0)
            angle_diff = np.arccos(dot_product)
            if angle_diff > self.max_turn_angle:
                cross_prod = prev_dir[0] * curr_dir[1] - prev_dir[1] * curr_dir[0]
                rot_dir = np.sign(cross_prod)
                if rot_dir == 0: rot_dir = 1.0
                cos_a = math.cos(rot_dir * self.max_turn_angle)
                sin_a = math.sin(rot_dir * self.max_turn_angle)
                new_dir_x = prev_dir[0] * cos_a - prev_dir[1] * sin_a
                new_dir_y = prev_dir[0] * sin_a + prev_dir[1] * cos_a
                new_dir = np.array([new_dir_x, new_dir_y])
                action_continuous = np.linalg.norm(action_continuous) * new_dir

        move = action_continuous * current_max_speed
        new_pos = self.agent_pos + move

        # [BUG-FIX] 穿墙修复 (安全锁)
        # 强制将新位置限制在地图边界内
        new_pos[0] = np.clip(new_pos[0], self.map_data.BOUNDS[0] + 1, self.map_data.BOUNDS[1] - 1)
        new_pos[1] = np.clip(new_pos[1], self.map_data.BOUNDS[2] + 1, self.map_data.BOUNDS[3] - 1)

        # --- [需求: 奖励空间-R1 (保持移动)] ---
        reward_R1 = -0.5  # [MODIFIED] 加大时间惩罚
        R_total += reward_R1

        # --- [需求: 奖励空间-R5 (趋向出口)] ---
        # 描述：R5被整合为"趋向子目标"
        # 实现：PPO的主要驱动力，奖励它靠近A*路径上的子目标。
        old_dist_to_subgoal = distance.euclidean(self.agent_pos, self.current_sub_goal)
        new_dist_to_subgoal = distance.euclidean(new_pos, self.current_sub_goal)
        reward_subgoal = (old_dist_to_subgoal - new_dist_to_subgoal) * 0.05  # [MODIFIED] 加大奖励
        R_total += reward_subgoal

        # --- [需求: 奖励空间-R2 (拓扑距离)] ---
        # 描述：低频(每30步)计算的宏观拓扑奖励
        reward_R2 = 0.0
        if self.step_count % self.macro_replan_interval == 0:
            reward_R2 = self.run_macro_planner()
            R_total += reward_R2

            # --- [需求: 奖励空间-R4, R6, R8, R9 (惩罚)] ---
        # 描述：R4(避开危险)和R6(绕行障碍)通过R8(进入危险区)和R9(碰撞)的惩罚来实现。
        # 实现：
        reward_penalty = 0.0
        # [REQ-R6, R9: 碰撞惩罚]
        if self.check_collision(self.agent_pos, new_pos):
            R_total -= 20.0  # [MODIFIED] 加大惩罚
            reward_penalty -= 20.0
            # 如果碰撞，不更新位置 (撞墙)
        else:
            self.agent_pos = new_pos  # 只有在不碰撞时才更新位置

        # [REQ-R4, R8: 危险区惩罚]
        if self.is_in_hazard_zone(self.agent_pos):  # 检查新位置
            R_total -= 20.0  # [MODIFIED] 加大惩罚
            reward_penalty -= 20.0

        # --- [需求: 奖励空间-R3 (到达出口)] ---
        reward_goal = 0.0
        if distance.euclidean(self.agent_pos, self.current_best_exit) < 1000 * MapConfig.SCALE:  # 1000mm (1m)
            reward_goal = 2000.0
            R_total += reward_goal
            done = True
            reached_goal = True

        # [REQ-ENV-3: 最大疏散时间]
        if self.step_count >= self.max_steps or self.simulation_time > self.map_data.MAX_SIMULATION_TIME:
            done = True
            R_total -= 100.0  # [MODIFIED] 加大超时惩罚
            reward_penalty -= 100.0

        self.prev_action_continuous = action_continuous.copy()
        self.update_sub_goal()
        observation = self.get_observation()

        # [REQ-DATA: 路径奖励/惩罚]
        # R1, R3, R_subgoal 是“引导性”奖励
        reward_positive_component = reward_R1 + reward_subgoal + reward_goal

        return observation, R_total, done, {
            'steps': self.step_count,
            'reached_goal': reached_goal,
            'curr_dist': distance.euclidean(self.agent_pos, self.current_best_exit),
            'simulation_time': self.simulation_time,  # [REQ-DATA: 疏散总时间]
            'current_speed': current_max_speed / 1000,  # [REQ-DATA: 实时移动速度]
            'vision_distance': self.get_current_vision() / 1000,  # [REQ-DATA: 实时有效视觉距离]
            'reward_R2': reward_R2,  # [REQ-DATA: 路径奖励 (拓扑)]
            'reward_positive': reward_positive_component,  # [REQ-DATA: 路径奖励 (本地)]
            'reward_penalty': reward_penalty,  # [REQ-DATA: 路径惩罚]
            'ray_visuals': self.last_ray_visuals
        }

    def reset(self):
        self.agent_pos = self.start.copy()
        self.step_count = 0
        self.simulation_time = 0.0
        self.prev_action_continuous = np.zeros(2, dtype=np.float32)
        self.last_ray_visuals = []

        self.prev_astar_dist = 0.0
        self.run_macro_planner()
        self.update_sub_goal()
        self.prev_astar_dist = self.macro_planner.get_path_length(self.macro_path)

        return self.get_observation()

    # --- [需求: 奖励空间-R6, R9 (碰撞惩罚)] ---
    # 描述：必须惩罚与“所有”障碍物的碰撞。
    # 实现：[BUG-FIX] 现在检查所有三种类型的障碍物。
    def check_collision(self, start, end):
        # 1. 检查视觉障碍 (墙壁/书架)
        for i in range(len(self.map_data.nb_micro_vision_obs)):
            obs = self.map_data.nb_micro_vision_obs[i]
            if _numba_segments_intersect(start[0], start[1], end[0], end[1], obs[0, 0], obs[0, 1], obs[1, 0],
                                         obs[1, 1]):
                return True
        # 2. 检查移动障碍 (桌椅)
        for i in range(len(self.map_data.nb_micro_movement_obs)):
            obs = self.map_data.nb_micro_movement_obs[i]
            if _numba_segments_intersect(start[0], start[1], end[0], end[1], obs[0, 0], obs[0, 1], obs[1, 0],
                                         obs[1, 1]):
                return True
        # 3. [BUG-FIX] 检查宏观障碍 (地图边界)
        for i in range(len(self.map_data.nb_macro_obstacles)):
            obs = self.map_data.nb_macro_obstacles[i]
            if _numba_segments_intersect(start[0], start[1], end[0], end[1], obs[0, 0], obs[0, 1], obs[1, 0],
                                         obs[1, 1]):
                return True
        return False

    # --- [需求: 奖励空间-R4, R8 (危险区惩罚)] ---
    def is_in_hazard_zone(self, pos):
        return _numba_is_in_hazard_zone(pos,
                                        self.map_data.nb_zones_hazard_centers,
                                        self.map_data.nb_zones_hazard_radii)


@numba.jit(nopython=True)
def _numba_is_in_hazard_zone(pos, centers, radii):
    for i in range(len(centers)):
        dist = math.sqrt((pos[0] - centers[i, 0]) ** 2 + (pos[1] - centers[i, 1]) ** 2)
        if dist < radii[i]:
            return True
    return False


# --- [需求: 状态空间-S3 (完整实现)] ---
# 描述：整合13射线、动态视距、多层穿透、加权力和Cos/Sin向量。
# 实现：此 Numba JIT 函数被 get_observation() 调用，执行所有繁重的射线计算。
@numba.jit(nopython=True)
def _numba_get_observation(
        agent_pos_x, agent_pos_y,
        prev_action_norm, prev_action_x, prev_action_y,
        sub_goal_x, sub_goal_y,
        ray_fov, num_rays, max_length,  # [REQ-S3: 动态视距]
        micro_vision_obs, micro_movement_obs,
        macro_obstacles,  # [BUG-FIX] 传入宏观障碍
        hazard_centers, hazard_radii,

        # [MODIFIED] v4.12: 传入所有四色引导
        guidance_up_lines, guidance_up_points,
        guidance_down_lines, guidance_down_points,
        guidance_right_lines, guidance_right_points,
        guidance_left_lines, guidance_left_points,

        FORCE_A, FORCE_B,  # [REQ-S3: 加权函数 (A, B)]
        map_width, map_height
):
    # [MODIFIED] v4.12: 状态向量维度增加
    final_state_vector = np.zeros((num_rays * 9) + 3, dtype=np.float64)
    visuals_array = np.zeros((num_rays, 7), dtype=np.float64)

    if prev_action_norm > 1e-6:
        agent_dir_angle = math.atan2(prev_action_y, prev_action_x)
    else:
        agent_dir_angle = math.atan2(sub_goal_y - agent_pos_y, sub_goal_x - agent_pos_x)

    ray_angles = np.linspace(-ray_fov / 2, ray_fov / 2, num_rays) + agent_dir_angle

    idx = 0
    for i in range(num_rays):
        angle = ray_angles[i]
        ray_d_x = math.cos(angle)
        ray_d_y = math.sin(angle)

        dist_vision = max_length
        dist_move = np.inf
        dist_hazard = np.inf
        # [MODIFIED] v4.12: 四个引导距离
        dist_guide_up = np.inf
        dist_guide_down = np.inf
        dist_guide_right = np.inf
        dist_guide_left = np.inf

        # --- [REQ-S3: 多层穿透 (实现)] ---
        # 1. 先检查视觉阻挡 (墙壁)
        for j in range(len(micro_vision_obs)):
            obs = micro_vision_obs[j]
            dist = _numba_get_ray_segment_intersection(
                agent_pos_x, agent_pos_y, ray_d_x, ray_d_y,
                obs[0, 0], obs[0, 1], obs[1, 0], obs[1, 1]
            )
            if dist > 0:
                dist_vision = min(dist_vision, dist)

        # [BUG-FIX] 同样检查宏观障碍(地图边界)是否阻挡视线
        for j in range(len(macro_obstacles)):
            obs = macro_obstacles[j]
            dist = _numba_get_ray_segment_intersection(
                agent_pos_x, agent_pos_y, ray_d_x, ray_d_y,
                obs[0, 0], obs[0, 1], obs[1, 0], obs[1, 1]
            )
            if dist > 0:
                dist_vision = min(dist_vision, dist)

        current_max_length = dist_vision  # 射线在此停止

        # 2. 再检查其他可穿透物体 (桌椅、危险区、引导)，但不能超过墙壁
        for j in range(len(micro_movement_obs)):
            obs = micro_movement_obs[j]
            dist = _numba_get_ray_segment_intersection(
                agent_pos_x, agent_pos_y, ray_d_x, ray_d_y,
                obs[0, 0], obs[0, 1], obs[1, 0], obs[1, 1]
            )
            if dist > 0 and dist < current_max_length:
                dist_move = min(dist_move, dist)

        for j in range(len(hazard_centers)):
            dist = _numba_get_ray_circle_intersection(
                agent_pos_x, agent_pos_y, ray_d_x, ray_d_y,
                hazard_centers[j, 0], hazard_centers[j, 1], hazard_radii[j]
            )
            if dist > 0 and dist < current_max_length:
                dist_hazard = min(dist_hazard, dist)

        # [MODIFIED] v4.12: 检查所有四色引导标识
        # 引导-上 (红)
        for j in range(len(guidance_up_lines)):
            obs = guidance_up_lines[j]
            dist = _numba_get_ray_segment_intersection(agent_pos_x, agent_pos_y, ray_d_x, ray_d_y, obs[0, 0], obs[0, 1],
                                                       obs[1, 0], obs[1, 1])
            if dist > 0 and dist < current_max_length: dist_guide_up = min(dist_guide_up, dist)
        # [FIX] v5: 移除了对 "引导点" 的检查，因为它们不存在
        # for j in range(len(guidance_up_points)):
        #     point = guidance_up_points[j]
        #     dist, perp_dist = _numba_get_ray_point_distance(agent_pos_x, agent_pos_y, ray_d_x, ray_d_y, point[0],
        #                                                     point[1])
        #     if dist > 0 and dist < current_max_length and perp_dist < 500.0: dist_guide_up = min(dist_guide_up, dist)

        # 引导-下 (绿/青)
        for j in range(len(guidance_down_lines)):
            obs = guidance_down_lines[j]
            dist = _numba_get_ray_segment_intersection(agent_pos_x, agent_pos_y, ray_d_x, ray_d_y, obs[0, 0], obs[0, 1],
                                                       obs[1, 0], obs[1, 1])
            if dist > 0 and dist < current_max_length: dist_guide_down = min(dist_guide_down, dist)
        # [FIX] v5: 移除了对 "引导点" 的检查
        # for j in range(len(guidance_down_points)):
        #     ...

        # 引导-右 (蓝)
        for j in range(len(guidance_right_lines)):
            obs = guidance_right_lines[j]
            dist = _numba_get_ray_segment_intersection(agent_pos_x, agent_pos_y, ray_d_x, ray_d_y, obs[0, 0], obs[0, 1],
                                                       obs[1, 0], obs[1, 1])
            if dist > 0 and dist < current_max_length: dist_guide_right = min(dist_guide_right, dist)
        # [FIX] v5: 移除了对 "引导点" 的检查
        # for j in range(len(guidance_right_points)):
        #     ...

        # 引导-左 (粉)
        for j in range(len(guidance_left_lines)):
            obs = guidance_left_lines[j]
            dist = _numba_get_ray_segment_intersection(agent_pos_x, agent_pos_y, ray_d_x, ray_d_y, obs[0, 0], obs[0, 1],
                                                       obs[1, 0], obs[1, 1])
            if dist > 0 and dist < current_max_length: dist_guide_left = min(dist_guide_left, dist)
        # [FIX] v5: 移除了对 "引导点" 的检查
        # for j in range(len(guidance_left_points)):
        #     ...

        # --- [REQ-S3: 加权 Movement/Vision/Hazard/Guidance] ---
        force_move = FORCE_A * math.exp(-dist_move / FORCE_B) if dist_move != np.inf else 0
        force_vision = FORCE_A * math.exp(-dist_vision / FORCE_B) if dist_vision != max_length else 0
        force_hazard = FORCE_A * math.exp(-dist_hazard / FORCE_B) if dist_hazard != np.inf else 0
        # [MODIFIED] v4.12: 四个独立的吸引力 (负值代表吸引)
        force_guide_up = -FORCE_A * math.exp(-dist_guide_up / FORCE_B) if dist_guide_up != np.inf else 0
        force_guide_down = -FORCE_A * math.exp(-dist_guide_down / FORCE_B) if dist_guide_down != np.inf else 0
        force_guide_right = -FORCE_A * math.exp(-dist_guide_right / FORCE_B) if dist_guide_right != np.inf else 0
        force_guide_left = -FORCE_A * math.exp(-dist_guide_left / FORCE_B) if dist_guide_left != np.inf else 0

        # --- [REQ-S3: Cos/Sin] ---
        final_state_vector[idx] = ray_d_x  # cos
        final_state_vector[idx + 1] = ray_d_y  # sin
        final_state_vector[idx + 2] = force_move
        final_state_vector[idx + 3] = force_vision
        final_state_vector[idx + 4] = force_hazard
        # [MODIFIED] v4.12: 添加新的引导力
        final_state_vector[idx + 5] = force_guide_up
        final_state_vector[idx + 6] = force_guide_down
        final_state_vector[idx + 7] = force_guide_right
        final_state_vector[idx + 8] = force_guide_left
        idx += 9

        ray_end_x = agent_pos_x + ray_d_x * dist_vision
        ray_end_y = agent_pos_y + ray_d_y * dist_vision
        did_not_hit = 1.0 if dist_vision == max_length else 0.0

        visuals_array[i, 0] = agent_pos_x
        visuals_array[i, 1] = agent_pos_y
        visuals_array[i, 2] = ray_end_x
        visuals_array[i, 3] = ray_end_y
        visuals_array[i, 4] = did_not_hit

    # --- [REQ-S1/S2: 相对位置/方向/距离] ---
    sub_goal_vec_x = sub_goal_x - agent_pos_x
    sub_goal_vec_y = sub_goal_y - agent_pos_y
    sub_goal_dist = math.sqrt(sub_goal_vec_x ** 2 + sub_goal_vec_y ** 2)

    sub_goal_dir_x = sub_goal_vec_x / (sub_goal_dist + 1e-6)
    sub_goal_dir_y = sub_goal_vec_y / (sub_goal_dist + 1e-6)

    map_diag = math.sqrt(map_width ** 2 + map_height ** 2)
    normalized_dist = sub_goal_dist / (map_diag + 1e-6)  # [REQ-S2: 距离比率]

    final_state_vector[idx] = sub_goal_dir_x  # [REQ-S1: 相对方向]
    final_state_vector[idx + 1] = sub_goal_dir_y
    final_state_vector[idx + 2] = normalized_dist

    # 将 Numba 数组转换回 Python 列表
    visuals = []
    for i in range(num_rays):
        v = visuals_array[i]
        visuals.append(((v[0], v[1]), (v[2], v[3]), v[4] > 0.5))

    return final_state_vector, visuals


# --- [MODIFIED] PPO 训练器 (V2中的 BeliefStateNetwork 被移除) ---
class PPOTrainer:
    def __init__(self, env, device):
        self.env = env
        self.device = device
        self.observation_dim = env.observation_dim
        self.action_dim_continuous = env.action_dim_continuous
        self.model = PPONetwork(self.observation_dim, self.action_dim_continuous).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0003)
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_param = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5

        # [NEW-SPEED] 调整超参数以加快训练
        self.ppo_epochs = 10  # 减少更新次数
        self.batch_size = 128  # 增大批次大小

        self.best_weights = {'model': self.model.state_dict()}

    def select_action(self, observation):
        state = torch.FloatTensor(observation).to(self.device)
        policy_continuous, _ = self.model(state)
        action_continuous = policy_continuous.sample()
        log_prob_continuous = policy_continuous.log_prob(action_continuous).sum(dim=-1)
        return action_continuous.cpu().numpy(), log_prob_continuous

    def compute_gae(self, rewards, values, next_value, dones):
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            next_value = values[i]
        advantages = np.array(advantages, dtype=np.float32)
        returns = advantages + values
        return advantages, returns

    def train_step(self, trajectories):
        states, actions, log_probs_old, rewards, dones, values = zip(*trajectories)
        states = np.array(states)
        actions = np.array(actions)
        log_probs_old = np.array(log_probs_old)
        rewards = np.array(rewards)
        dones = np.array(dones)
        values = np.array(values).flatten()

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        log_probs_old = torch.FloatTensor(log_probs_old).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        values = torch.FloatTensor(values).to(self.device)

        with torch.no_grad():
            _, next_value = self.model(states[-1])
            next_value = next_value.item()

        advantages, returns = self.compute_gae(rewards.cpu().numpy(), values.cpu().numpy(), next_value,
                                               dones.cpu().numpy())
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        policy_loss_total, value_loss_total = 0, 0
        for _ in range(self.ppo_epochs):
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            for start in range(0, len(states), self.batch_size):
                batch_indices = indices[start:start + self.batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_log_probs_old = log_probs_old[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                policy_continuous, value = self.model(batch_states)

                log_probs_continuous = policy_continuous.log_prob(batch_actions).sum(dim=-1)
                log_probs = log_probs_continuous
                entropy = policy_continuous.entropy().mean()

                ratio = torch.exp(log_probs - batch_log_probs_old)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(value.squeeze(), batch_returns)
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                policy_loss_total += policy_loss.item()
                value_loss_total += value_loss.item()

        num_updates = (self.ppo_epochs * (len(states) // self.batch_size)) + 1
        return policy_loss_total / num_updates, value_loss_total / num_updates

    def save_best_weights(self):
        self.best_weights = {'model': self.model.state_dict()}

    def load_best_weights(self):
        self.model.load_state_dict(self.best_weights['model'])


# --- [需求: 数据采集 (路径图片)] ---
# 描述：每隔n秒(步)输出路径图片。
# 实现：save_path_plot 函数，在 run_training 中被调用。
def save_path_plot(map_data, start, goal, ppo_path, astar_path, filename):
    plt.figure(figsize=(12, 8), dpi=300)

    for obs in map_data.MACRO_OBSTACLES:
        plt.plot([obs[0][0], obs[1][0]], [obs[0][1], obs[1][1]], 'k-', linewidth=2.0, zorder=2)
    for obs in map_data.MICRO_OBSTACLES_MOVEMENT:
        plt.plot([obs[0][0], obs[1][0]], [obs[0][1], obs[1][1]], 'gray', linewidth=1.0, zorder=1)
    if ppo_path:
        ppo_path = np.array(ppo_path)
        plt.plot(ppo_path[:, 0], ppo_path[:, 1], '#008000', linewidth=1, label='PPO Path', zorder=3)
    if astar_path:
        astar_path = np.array(astar_path)
        plt.plot(astar_path[:, 0], astar_path[:, 1], '#800080', linewidth=1, linestyle='--', label='A* Macro Path',
                 zorder=3)
    plt.plot(start[0], start[1], 'ro', markersize=3, markeredgecolor='#ff0000', label='Start', zorder=4)
    if goal:
        plt.plot(goal[0], goal[1], 'bo', markersize=5, markeredgecolor='b', label='Goal', zorder=4)

    # [MODIFIED] v4.12: 绘制四色引导
    def _plot_guidance(guidance_list, color):
        for item in guidance_list:
            if isinstance(item[0], (list, np.ndarray)) and len(item[0]) == 2 and isinstance(item[1],
                                                                                            (list, np.ndarray)):  # 线段
                plt.plot([item[0][0], item[1][0]], [item[0][1], item[1][1]], color=color, linestyle='--', linewidth=1,
                         zorder=1)
            # [FIX] v5: 移除了圆形(点)的绘制逻辑
            # elif isinstance(item[0], (list, np.ndarray)) and len(item[0]) == 2 and isinstance(item[1], float):  # 圆形(点)
            #     plt.plot(item[0][0], item[0][1], color=color, marker='.', markersize=2, zorder=1)

    _plot_guidance(map_data.GUIDANCE_UP, 'red')
    _plot_guidance(map_data.GUIDANCE_DOWN, 'cyan')  # [MODIFIED] v5.1: 改为 cyan (青色)
    _plot_guidance(map_data.GUIDANCE_RIGHT, 'blue')
    _plot_guidance(map_data.GUIDANCE_LEFT, 'magenta')

    for smoke_center, smoke_radius in map_data.ZONES_HAZARD:
        circle = plt.Circle(smoke_center, smoke_radius, color='r', alpha=0.2, zorder=1)
        plt.gca().add_patch(circle)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(map_data.BOUNDS[0], map_data.BOUNDS[1])
    plt.ylim(map_data.BOUNDS[2], map_data.BOUNDS[3])
    # [MODIFIED] 保持图表英文
    plt.xlabel('X (mm)', fontsize=12)
    plt.ylabel('Y (mm)', fontsize=12)
    plt.title('Fire Evacuation Simulation (v5.1 Color Fix)', fontsize=14, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


# --- [NEW-VIDEO] 用于绘制视频的单帧 ---
def plot_simulation_step(map_data, start, goal, path_so_far, current_pos, ray_visuals, astar_path, filename):
    plt.figure(figsize=(12, 8), dpi=150)
    ax = plt.gca()

    # 1. 绘制地图
    for obs in map_data.MACRO_OBSTACLES:
        ax.plot([obs[0][0], obs[1][0]], [obs[0][1], obs[1][1]], 'k-', linewidth=1.5, zorder=2)
    for obs in map_data.MICRO_OBSTACLES_MOVEMENT:
        ax.plot([obs[0][0], obs[1][0]], [obs[0][1], obs[1][1]], 'gray', linewidth=0.5, zorder=1)

    # [MODIFIED] v4.12: 绘制四色引导
    def _plot_guidance(guidance_list, color):
        for item in guidance_list:
            if isinstance(item[0], (list, np.ndarray)) and len(item[0]) == 2 and isinstance(item[1],
                                                                                            (list, np.ndarray)):  # 线段
                ax.plot([item[0][0], item[1][0]], [item[0][1], item[1][1]], color=color, linestyle='--', linewidth=0.5,
                        zorder=1)
            # [FIX] v5: 移除了圆形(点)的绘制逻辑
            # elif isinstance(item[0], (list, np.ndarray)) and len(item[0]) == 2 and isinstance(item[1], float):  # 圆形(点)
            #     ax.plot(item[0][0], item[0][1], color=color, marker='.', markersize=2, zorder=1)

    _plot_guidance(map_data.GUIDANCE_UP, 'red')
    _plot_guidance(map_data.GUIDANCE_DOWN, 'cyan')  # [MODIFIED] v5.1: 改为 cyan (青色)
    _plot_guidance(map_data.GUIDANCE_RIGHT, 'blue')
    _plot_guidance(map_data.GUIDANCE_LEFT, 'magenta')

    for smoke_center, smoke_radius in map_data.ZONES_HAZARD:
        circle = plt.Circle(smoke_center, smoke_radius, color='r', alpha=0.2, zorder=1)
        ax.add_patch(circle)

    # 2. 绘制A*路径
    if astar_path:
        astar_path = np.array(astar_path)
        ax.plot(astar_path[:, 0], astar_path[:, 1], '#800080', linewidth=1, linestyle='--', label='A* Macro Path',
                zorder=3)

    # 3. 绘制PPO已走路径
    if path_so_far:
        path_np = np.array(path_so_far)
        ax.plot(path_np[:, 0], path_np[:, 1], '#008000', linewidth=1.5, label='PPO Path', zorder=4)

    # 4. 绘制AGV当前位置
    ax.plot(current_pos[0], current_pos[1], 'go', markersize=6, label='AGV', zorder=5)

    # 5. 绘制射线
    if ray_visuals:
        for ray_start, ray_end, did_not_hit in ray_visuals:
            if did_not_hit:
                ax.plot([ray_start[0], ray_end[0]], [ray_start[1], ray_end[1]], color='green', linestyle=':',
                        linewidth=0.5, alpha=0.4, zorder=4)
            else:
                ax.plot([ray_start[0], ray_end[0]], [ray_start[1], ray_end[1]], color='red', linestyle='-',
                        linewidth=0.7, alpha=0.6, zorder=4)

    # 6. 绘制起点和终点
    ax.plot(start[0], start[1], 'ro', markersize=3, markeredgecolor='#ff0000', label='Start', zorder=4)
    if goal:
        ax.plot(goal[0], goal[1], 'bo', markersize=5, markeredgecolor='b', label='Goal', zorder=4)

    # 7. 设置格式 (英文)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(map_data.BOUNDS[0], map_data.BOUNDS[1])
    ax.set_ylim(map_data.BOUNDS[2], map_data.BOUNDS[3])
    ax.set_xlabel('X (mm)', fontsize=10)
    ax.set_ylabel('Y (mm)', fontsize=10)
    ax.set_title('AGV Evacuation Simulation', fontsize=12)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


# --- [NEW-VIDEO] 用于从帧创建视频 ---
def create_video_from_frames(frames_dir, video_filename, fps=10):
    try:
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png") and f.startswith("frame_")])
        if not frame_files:
            print("  [视频错误] 未找到帧文件。")
            return

        frame_files.sort(key=lambda f: int(f.split('_')[1].split('.')[0]))

        print(f"  正在从 {len(frame_files)} 帧创建视频... {video_filename}")

        with imageio.get_writer(video_filename, fps=fps) as writer:
            for filename in tqdm(frame_files, desc="  合成视频中"):
                frame_path = os.path.join(frames_dir, filename)
                image = imageio.imread(frame_path)
                writer.append_data(image)

        print(f"  视频创建成功。正在清理临时帧...")
        shutil.rmtree(frames_dir)
        print("  清理完成。")

    except Exception as e:
        print(f"  [视频错误] 创建视频失败: {e}")
        traceback.print_exc()


# --- [MODIFIED] 主模拟器类 (V2的逻辑被替换) ---
class PathPlanningSimulator:
    def __init__(self, map_data, start_point, goal_point):
        self.device = DEVICE

        self.map_data = map_data
        self.start = start_point
        self.goal = goal_point

        self.episodes = 1000
        self.patience = 100

        self.env = self.create_env()
        self.trainer = PPOTrainer(self.env, self.device)

        self.astar_plotter = AStarPlanner(self.map_data, grid_size=self.map_data.ASTAR_GRID_SIZE)

        self.astar_path = None

        # --- [需求: 数据采集 (所有指标)] ---
        # 描述：采集疏散效率、路径、速度、视距、奖励、惩罚等信息。
        # 实现：定义 self.results 字典来存储所有数据。
        # [MODIFIED] v4.11: 拆分奖励，并添加路径长度
        self.results = {
            'episode': [], 'policy_loss': [], 'value_loss': [], 'reward': [], 'steps': [],
            'reached_goal': [],
            'simulation_time': [],  # [REQ-DATA: 疏散总时间]
            'path_length': [],  # [REQ-DATA: 疏散路径总长]
            'current_speed': [],  # [REQ-DATA: 平均/实时移动速度]
            'vision_distance': [],  # [REQ-DATA: 平均/实时有效视觉距离]
            'reward_R2': [],  # [REQ-DATA: 路径奖励 (拓扑R2)]
            'reward_positive': [],  # [REQ-DATA: 路径奖励 (本地R1,R3,Subgoal)]
            'reward_penalty': []  # [REQ-DATA: 路径惩罚 (R8,R9)]
        }
        self.best_reward = -float('inf')
        self.patience_counter = 0

    def create_env(self):
        return PathPlanningEnv(self.start, self.goal, self.map_data, self.device)

    def run_training(self):
        print(f"开始训练 (v5.1 Color Fix - Numba 加速版)...")

        self.astar_path = self.astar_plotter.find_path(self.start, self.goal)
        if self.astar_path is None:
            print("警告: 训练开始时无法生成用于绘图的A*路径。")

        save_path_plot(self.map_data, self.start, self.goal, [], self.astar_path,
                       os.path.join(PPO_DIR, "initial_astar_path.png"))

        for current_episode in range(self.episodes):

            is_first_episode = (current_episode == 0) and CREATE_VIDEO_EPISODE_0
            frame_count = 0
            pbar = None

            if is_first_episode:
                if os.path.exists(VIDEO_FRAMES_DIR):
                    shutil.rmtree(VIDEO_FRAMES_DIR)
                os.makedirs(VIDEO_FRAMES_DIR, exist_ok=True)
                print("--- [视频录制] 第 0 轮开始，正在录制帧... ---")
                pbar = tqdm(total=self.env.map_data.MAX_SIMULATION_TIME, desc="[视频录制] 渲染第 0 轮")

            trajectories = []
            total_reward = 0
            observation = self.env.reset()
            path = [self.env.agent_pos.copy()]

            # [REQ-DATA: 采集信息 (临时列表)]
            episode_speeds = []
            episode_visions = []
            episode_reward_R2 = 0
            episode_reward_positive = 0
            episode_reward_penalty = 0
            total_path_length = 0.0

            while True:
                action, log_prob = self.trainer.select_action(observation)
                observation, reward, done, info = self.env.step(action)

                with torch.no_grad():
                    state_tensor = torch.FloatTensor(observation).to(self.device)
                    _, value = self.trainer.model(state_tensor)

                trajectories.append(
                    (observation, action, log_prob.detach().cpu().numpy(), reward, done, value.detach().cpu().numpy()))
                total_reward += reward

                # [REQ-DATA: 疏散路径总长 (计算)]
                if len(path) > 0:
                    total_path_length += np.linalg.norm(self.env.agent_pos - path[-1])
                path.append(self.env.agent_pos.copy())

                # [REQ-DATA: 采集信息 (每一步)]
                episode_speeds.append(info['current_speed'])
                episode_visions.append(info['vision_distance'])
                episode_reward_R2 += info['reward_R2']
                episode_reward_positive += info['reward_positive']
                episode_reward_penalty += info['reward_penalty']

                if pbar:
                    pbar.update(1)

                if is_first_episode:
                    frame_filename = os.path.join(VIDEO_FRAMES_DIR, f"frame_{frame_count:04d}.png")
                    try:
                        plot_simulation_step(
                            self.map_data, self.start, self.goal,
                            path, self.env.agent_pos,
                            info['ray_visuals'], self.astar_path,
                            frame_filename
                        )
                        frame_count += 1
                    except Exception as e:
                        print(f" [!] 绘制帧失败: {e}")

                if done:
                    if pbar:
                        pbar.close()
                    break

            if is_first_episode:
                if pbar and pbar.n < pbar.total:
                    pbar.close()
                video_filename = os.path.join(PPO_DIR, "episode_0_simulation.mp4")
                print(f"--- [视频录制] 第 0 轮结束，正在创建视频: {video_filename} ---")
                create_video_from_frames(VIDEO_FRAMES_DIR, video_filename, fps=10)
                print("--- [视频录制] 视频创建完成。 ---")

            policy_loss, value_loss = self.trainer.train_step(trajectories)

            # [REQ-DATA: 采集信息 (聚合)]
            self.results['episode'].append(current_episode)
            self.results['policy_loss'].append(policy_loss)
            self.results['value_loss'].append(value_loss)
            self.results['reward'].append(total_reward)
            self.results['steps'].append(info['steps'])
            self.results['reached_goal'].append(info['reached_goal'])
            self.results['simulation_time'].append(info['simulation_time'])
            self.results['path_length'].append(total_path_length / 1000.0)  # 转换为米
            self.results['current_speed'].append(np.mean(episode_speeds) if episode_speeds else 0)
            self.results['vision_distance'].append(np.mean(episode_visions) if episode_visions else 0)
            self.results['reward_R2'].append(episode_reward_R2)
            self.results['reward_positive'].append(episode_reward_positive)
            self.results['reward_penalty'].append(episode_reward_penalty)

            print(f"轮数 {current_episode}: 总奖励={total_reward:.2f}, 步数={info['steps']}, "
                  f"时间={info['simulation_time']:.1f}s, 到达目标={info['reached_goal']}, "
                  f"距终点={info['curr_dist']:.2f}")

            # [REQ-DATA: 路径图片]
            if current_episode % 20 == 0 or (info['reached_goal'] and total_reward > self.best_reward):
                current_astar_path = self.astar_plotter.find_path(self.start, self.goal)
                save_path_plot(self.map_data, self.start, self.goal, path, current_astar_path,
                               os.path.join(PPO_DIR, f"path_episode_{current_episode}.png"))

            if total_reward > self.best_reward:
                self.best_reward = total_reward
                self.trainer.save_best_weights()
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                print(f"在第 {current_episode} 轮提前停止训练")
                break

        self.save_plots()
        pd.DataFrame(self.results).to_csv(os.path.join(PPO_DIR, "training_results.csv"), index=False)
        print("训练完成。")

    def run_testing(self):
        print("开始测试...")
        self.trainer.load_best_weights()

        observation = self.env.reset()
        path = [self.env.agent_pos.copy()]
        total_reward = 0
        total_path_length = 0.0

        while True:
            action, _ = self.trainer.select_action(observation)
            observation, reward, done, info = self.env.step(action)

            if len(path) > 0:
                total_path_length += np.linalg.norm(self.env.agent_pos - path[-1])
            path.append(self.env.agent_pos.copy())
            total_reward += reward

            if done:
                break

        current_astar_path = self.astar_plotter.find_path(self.start, self.goal)
        print(f"测试: 总奖励={total_reward:.2f}, 步数={info['steps']}, "
              f"时间={info['simulation_time']:.1f}s, 路径长度={total_path_length / 1000.0:.2f}m, "
              f"到达目标={info['reached_goal']}, 距终点={info['curr_dist']:.2f}")

        save_path_plot(self.map_data, self.start, self.goal, path, current_astar_path,
                       os.path.join(PPO_DIR, "test_path.png"))
        print("测试完成。")

        # --- [需求: 数据采集 (生成图表)] ---

    # 描述：生成奖励曲线、效率、速度、视距、奖励/惩罚分值等图表。
    # 实现：此函数绘制所有收集到的 self.results 数据。
    def save_plots(self):
        # [MODIFIED] v4.11: 3x2 布局，包含所有需求的图表
        fig, axes = plt.subplots(3, 2, figsize=(15, 20), dpi=300)

        # 1. 奖励曲线 (RL Curve)
        axes[0, 0].plot(self.results['episode'], self.results['reward'], label='Total Reward')
        axes[0, 0].set_title('Total Reward (RL Curve)')  # [REQ-DATA: 奖励曲线]
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()

        # 2. 疏散时间 (Efficiency)
        axes[0, 1].plot(self.results['episode'], self.results['simulation_time'], label='Evacuation Time')
        axes[0, 1].set_title('Evacuation Time (Efficiency)')  # [REQ-DATA: 疏散效率]
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Time (s)')
        axes[0, 1].legend()

        # 3. 疏散路径总长 (Path Length)
        axes[1, 0].plot(self.results['episode'], self.results['path_length'], label='Path Length')
        axes[1, 0].set_title('Total Path Length')  # [REQ-DATA: 疏散路径总长]
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Length (m)')
        axes[1, 0].legend()

        # 4. 奖励/惩罚分值 (Reward Components)
        axes[1, 1].plot(self.results['episode'], self.results['reward_R2'], label='R2 Topology Reward', alpha=0.8)
        axes[1, 1].plot(self.results['episode'], self.results['reward_positive'],
                        label='Local Positive Reward (R1+R3+Subgoal)', alpha=0.8)
        axes[1, 1].plot(self.results['episode'], self.results['reward_penalty'], label='Local Penalty (R8+R9+Timeout)',
                        alpha=0.8)
        axes[1, 1].set_title('Reward Components')  # [REQ-DATA: 路径奖励/惩罚]
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].legend()

        # 5. 环境动态 (Speed/Vision)
        ax_speed = axes[2, 0]
        ax_vision = ax_speed.twinx()
        ax_speed.plot(self.results['episode'], self.results['current_speed'], label='Avg Speed', color='tab:blue')
        ax_vision.plot(self.results['episode'], self.results['vision_distance'], label='Avg Vision', color='tab:orange')
        ax_speed.set_title('Environment Dynamics (Episode Avg)')  # [REQ-DATA: 速度/视距]
        ax_speed.set_xlabel('Episode')
        ax_speed.set_ylabel('Max Speed (m/s)', color='tab:blue')
        ax_vision.set_ylabel('Max Vision (m)', color='tab:orange')
        ax_speed.legend(loc='upper left')
        ax_vision.legend(loc='upper right')

        # 6. PPO 损失 (Loss Curves)
        axes[2, 1].plot(self.results['episode'], self.results['policy_loss'], label='Policy Loss', alpha=0.8)
        axes[2, 1].plot(self.results['episode'], self.results['value_loss'], label='Value Loss', alpha=0.8)
        axes[2, 1].set_title('PPO Loss Curves')
        axes[2, 1].set_xlabel('Episode')
        axes[2, 1].set_ylabel('Loss')
        axes[2, 1].legend()

        fig.tight_layout()
        plt.savefig(os.path.join(PPO_DIR, "training_plots.png"), bbox_inches='tight')
        plt.close(fig)


# --- [MODIFIED] 主执行流程 ---
if __name__ == "__main__":
    dxf_file = "Drawing2.dxf"

    # 1. 首先加载地图数据
    map_data = MapConfig()
    map_data.load_from_dxf(dxf_file)

    # 2. [MODIFIED] 移除 PointSelector, 使用硬编码的坐标
    start_point = [7494.3383, 25027.4642]
    goal_point = [17057.3508, 21129.0]

    # 3. 检查是否成功选择 (现在总是成功)
    if start_point and goal_point:
        print("--- 起点和终点已确认 ---")
        print(f"起点: {start_point}")
        print(f"终点: {goal_point}")
        print("---------------------")

        try:
            # 4. [MODIFIED] 将地图数据和选中的点传入模拟器
            simulator = PathPlanningSimulator(
                map_data=map_data,
                start_point=start_point,
                goal_point=goal_point
            )
            simulator.run_training()
            simulator.run_testing()
        except Exception as e:
            print("程序出错:")
            traceback.print_exc()
    else:
        # This part should not be reachable anymore
        print("未设置起点和终点。正在退出。")