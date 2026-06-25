import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import heapq
import traceback
from tqdm import tqdm
import seaborn as sns
from numba import jit  # 导入 numba 库

# 设置 matplotlib 默认字体为英文，解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 检查 CUDA 是否可用
if not torch.cuda.is_available():
    print("CUDA 不可用，训练将在 CPU 上进行。")
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda")
    print("CUDA 可用，训练将在 GPU 上进行。")

# 全局地图定义
# 0 是墙，1 是可通行区域，2 是目标
MAP_STRINGS = [
    "00000000020000000000000000200000000",
    "01111111111111111111111111111111112",
    "21111111111111111111111111111111110",
    "01111102000200111111000020020111110",
    "01111100020020111111002000200111110",
    "01111111111111111111111111111111112",
    "21111111111111111111111111111111110",
    "01111102000200111111000020020111110",
    "01111100020020111111002000200111110",
    "01111111111111111111111111111111112",
    "21111111111111111111111111111111110",
    "01111102000200111111000020020111110",
    "01111100020020111111002000200111110",
    "01111111111111111111111111111111112",
    "21111111111111111111111111111111110",
    "01111102000200111111000020020111110",
    "01111100020020111111002000200111110",
    "01111111111111111111111111111111112",
    "21111111111111111111111111111111110",
    "01111102000200111111000020020111110",
    "01111100020020111111002000200111110",
    "01111111111111111111111111111111112",
    "21111111111111111111111111111111110",
    "01111102000200111111000020020111110",
    "01111100020020111111002000200111110",
    "01111111111111111111111111111111112",
    "21111111111111111111111111111111110",
    "01111102000200111111000020020111110",
    "01111100020020111111002000200111110",
    "01111111111111111111111111111111112",
    "21111111111111111111111111111111110",
    "01111102000200111111000020020111110",
    "01111100020020111111002000200111110",
    "01111111111111111111111111111111110",
    "00000002000000000200000000020000000"
]

CELL_SIZE = 1.0  # 每个字符代表 1x1 单位
AGENT_RADIUS = 0.4 * CELL_SIZE  # 代理半径，用于碰撞检测

# 用于存储初始随机生成的起点和目标位置的全局变量
# 确保每次reset时使用相同的起始位置和目标，以保证训练的稳定性
_initial_agv_start_positions = None
_initial_agv_goals = None
_initial_agv_initial_a_star_distances = None
_initial_agv_positions_set = False


def parse_map(map_strings, cell_size=CELL_SIZE):
    """
    解析地图字符串以生成障碍物、纯可通行坐标和目标坐标。
    """
    obstacles = []
    pure_walkable_coords = []
    goal_coords = []

    rows = len(map_strings)
    cols = len(map_strings[0])

    grid = [list(row) for row in map_strings]

    for r in range(rows):
        for c in range(cols):
            x = c * cell_size
            y = r * cell_size

            if grid[r][c] == '0':  # 墙
                # 添加四条边作为障碍物段，并确保是 float32
                obstacles.append([np.array([x, y], dtype=np.float32), np.array([x + cell_size, y], dtype=np.float32)])
                obstacles.append([np.array([x, y], dtype=np.float32), np.array([x, y + cell_size], dtype=np.float32)])
                obstacles.append([np.array([x + cell_size, y], dtype=np.float32),
                                  np.array([x + cell_size, y + cell_size], dtype=np.float32)])
                obstacles.append([np.array([x, y + cell_size], dtype=np.float32),
                                  np.array([x + cell_size, y + cell_size], dtype=np.float32)])
            elif grid[r][c] == '1':  # 纯可通行区域
                pure_walkable_coords.append(np.array([x + cell_size / 2, y + cell_size / 2], dtype=np.float32))
            elif grid[r][c] == '2':  # 目标
                goal_coords.append(np.array([x + cell_size / 2, y + cell_size / 2], dtype=np.float32))

    # 移除重复的障碍物段，确保唯一性
    unique_obstacles = []
    seen_obstacles = set()
    for obs in obstacles:
        p1, p2 = obs[0], obs[1]  # obs[0] and obs[1] are already np.array
        # 将点对排序，确保 (p1, p2) 和 (p2, p1) 被视为相同
        sorted_obs = tuple(map(tuple, sorted([p1, p2], key=lambda p: (p[0], p[1]))))
        if sorted_obs not in seen_obstacles:
            unique_obstacles.append(obs)
            seen_obstacles.add(sorted_obs)

    obstacles = unique_obstacles

    map_bounds = [0, cols * cell_size, 0, rows * cell_size]
    return obstacles, pure_walkable_coords, goal_coords, map_bounds, rows, cols


# 解析地图字符串以获取全局地图信息
# 这些全局常量在环境实例化时被使用
OBSTACLES, PURE_WALKABLE_COORDS, GOAL_COORDS, MAP_BOUNDS, MAP_ROWS, MAP_COLS = parse_map(MAP_STRINGS)


# A* 寻路算法
class AStar:
    def __init__(self, grid_map, cell_size=CELL_SIZE):
        """
        A* 寻路算法的实现。
        grid_map: 代表地图的 2D 字符列表。
        cell_size: 每个网格单元的大小。
        """
        self.grid_map = grid_map
        self.rows = len(grid_map)
        self.cols = len(grid_map[0])
        self.cell_size = cell_size

    def heuristic(self, a, b):
        """
        启发式函数 (曼哈顿距离)。
        a, b: 网格坐标 (行, 列)。
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, node):
        """
        获取给定节点的邻居 (包括对角线)。
        node: 网格坐标 (行, 列)。
        """
        neighbors = []
        # 8 个方向：上下左右，以及四个对角线
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            r, c = node[0] + dr, node[1] + dc
            if 0 <= r < self.rows and 0 <= c < self.cols and self.grid_map[r][c] != '0':
                neighbors.append((r, c))
        return neighbors

    def find_path(self, start_world, goal_world):
        """
        在世界坐标中查找从起点到目标的 A* 路径。
        start_world, goal_world: 世界坐标 (x, y)。
        """
        # 将世界坐标转换为网格坐标
        start_grid = (int(start_world[1] / self.cell_size), int(start_world[0] / self.cell_size))
        goal_grid = (int(goal_world[1] / self.cell_size), int(goal_world[0] / self.cell_size))

        # 检查起点和终点是否可通行
        if not (0 <= start_grid[0] < self.rows and 0 <= start_grid[1] < self.cols and \
                self.grid_map[start_grid[0]][start_grid[1]] != '0'):
            return None  # 起点不可通行

        if not (0 <= goal_grid[0] < self.rows and 0 <= goal_grid[1] < self.cols and \
                self.grid_map[goal_grid[0]][goal_grid[1]] != '0'):
            return None  # 终点不可通行

        open_set = []
        heapq.heappush(open_set, (0, start_grid))  # (f_score, node)

        came_from = {}  # 用于重建路径

        g_score = {(r, c): float('inf') for r in range(self.rows) for c in range(self.cols)}
        g_score[start_grid] = 0

        f_score = {(r, c): float('inf') for r in range(self.rows) for c in range(self.cols)}
        f_score[start_grid] = self.heuristic(start_grid, goal_grid)

        while open_set:
            current_f_score, current_node = heapq.heappop(open_set)

            if current_node == goal_grid:
                path = []
                while current_node in came_from:
                    path.append(current_node)
                    current_node = came_from[current_node]
                path.append(start_grid)
                path.reverse()
                # 将网格路径转换为世界坐标 (单元格中心)，并确保是 float32
                return [np.array([c * self.cell_size + self.cell_size / 2, r * self.cell_size + self.cell_size / 2],
                                 dtype=np.float32) for r, c in path]

            for neighbor in self.get_neighbors(current_node):
                # 计算从 current_node 到 neighbor 的距离
                dist_cost = 1.0 if (neighbor[0] == current_node[0] or neighbor[1] == current_node[1]) else np.sqrt(2.0)
                tentative_g_score = g_score[current_node] + dist_cost

                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current_node
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None  # 未找到路径


# 使用 numba 加速线段相交辅助函数
@jit(nopython=True, cache=True)
def _ccw(A, B, C):
    """确定点 C 相对于向量 AB 是顺时针、逆时针还是共线。"""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


@jit(nopython=True, cache=True)
def segments_intersect(p1, p2, q1, q2):
    """
    检查两条线段 (p1, p2) 和 (q1, q2) 是否相交。
    使用方向检查 (叉积属性)。
    """
    return _ccw(p1, q1, q2) != _ccw(p2, q1, q2) and _ccw(p1, p2, q1) != _ccw(p1, p2, q2)


# 使用 numba 加速射线与线段交点检测
@jit(nopython=True, cache=True)
def _ray_segment_intersect_numba(ray_origin, ray_direction, segment_p1, segment_p2):
    """
    计算射线与线段的交点。
    返回从 ray_origin 到交点的距离，或 float('inf') 如果没有交点。
    """
    # 将线段表示为 Q = P1 + u * (P2 - P1)
    # 将射线表示为 R = O + t * D
    # 联立方程：O + t * D = P1 + u * (P2 - P1)
    # t*D - u*(P2 - P1) = P1 - O

    # 向量 B_vec = P2 - P1
    B_vec_x = segment_p2[0] - segment_p1[0]
    B_vec_y = segment_p2[1] - segment_p1[1]

    # 行列式分母
    det = ray_direction[0] * (-B_vec_y) - ray_direction[1] * (-B_vec_x)

    if abs(det) < 1e-6:  # 几乎平行
        return np.inf

    inv_det = 1.0 / det

    vec_AO_x = ray_origin[0] - segment_p1[0]
    vec_AO_y = ray_origin[1] - segment_p1[1]

    # 计算 t 和 u
    t = (vec_AO_x * (-B_vec_y) - vec_AO_y * (-B_vec_x)) * inv_det
    u = (ray_direction[0] * vec_AO_y - ray_direction[1] * vec_AO_x) * inv_det

    # 射线交点条件: t >= 0 (射线方向)
    # 线段交点条件: 0 <= u <= 1 (在线段内部)
    if t >= 0 and 0 <= u <= 1:
        return t  # 返回距离
    return np.inf


# 使用 numba 加速碰撞检测，移到类外部
@jit(nopython=True, cache=True)
def check_collision_with_obstacles_numba(start_pos, end_pos, relevant_obstacles, agent_radius):
    """
    检查代理是否与静态障碍物发生碰撞。
    此函数直接接收相关障碍物列表，避免在 JIT 编译函数中调用 Python 列表操作。
    它检查 (start_pos, end_pos) 移动线段与障碍物线段的交点，
    并检查 end_pos 圆形区域与障碍物线段的距离。
    """
    # 1. 检查移动路径 (线段) 与障碍物线段的交点
    for i in range(relevant_obstacles.shape[0]):  # 遍历 NumPy 数组的行
        obs = relevant_obstacles[i]
        p1 = obs[0]  # obs[0] and obs[1] are already np.array
        p2 = obs[1]
        if segments_intersect(start_pos, end_pos, p1, p2):
            return True

    # 2. 检查代理的圆形区域是否与障碍物线段相交
    for i in range(relevant_obstacles.shape[0]):
        obs = relevant_obstacles[i]
        p1 = obs[0]
        p2 = obs[1]

        line_vec = p2 - p1  # 确保这里的数据类型一致
        line_len_sq = np.dot(line_vec, line_vec)  # 确保这里的数据类型一致

        if line_len_sq < 1e-6:  # 如果障碍物是点
            if np.linalg.norm(end_pos - p1) < agent_radius:
                return True
        else:
            # 在线段上找到距离 end_pos 最近的点
            t = np.dot(end_pos - p1, line_vec) / line_len_sq  # 确保这里的数据类型一致
            t = max(0.0, min(1.0, t))  # 限制 t 在 [0, 1] 之间，找到线段上的最近点
            closest_point_on_segment = p1 + t * line_vec
            if np.linalg.norm(end_pos - closest_point_on_segment) < agent_radius:
                return True
    return False


# 多代理环境类
class MultiAgentPathPlanningEnv:
    def __init__(self, obstacles, pure_walkable_coords, goal_coords, map_bounds, map_rows, map_cols, num_agents, device,
                 max_steps_per_episode=100):
        """
        多代理路径规划环境。
        obstacles: 静态障碍物段列表。
        pure_walkable_coords: 纯可通行区域中心坐标列表 (不包括目标)。
        goal_coords: 目标区域中心坐标列表。
        map_bounds: 地图边界 [min_x, max_x, min_y, max_y]。
        map_rows, map_cols: 地图的行数和列数。
        num_agents: 代理 (AGV) 数量。
        device: PyTorch 设备 (CPU/CUDA)。
        max_steps_per_episode: 每个 episode 的最大步数。
        """
        self.obstacles = obstacles
        self.pure_walkable_coords = pure_walkable_coords
        self.goal_coords = goal_coords
        self.map_bounds = map_bounds
        self.map_rows = map_rows
        self.map_cols = map_cols
        self.num_agents = num_agents
        self.device = device
        self.max_speed = CELL_SIZE / 2.0  # 代理最大移动速度
        self.max_steps_per_episode = max_steps_per_episode
        self.agent_radius = AGENT_RADIUS  # 代理半径
        self.max_turn_angle = np.deg2rad(45)  # 最大转弯角度限制

        self.agents = []  # 存储代理状态的列表
        self.astar_finder = AStar(grid_map=[list(row) for row in MAP_STRINGS])

        # --- 优化: 障碍物空间分区 ---
        # 将地图划分为更小的网格，每个网格存储其中的障碍物，以便快速查询附近的障碍物
        self.grid_partition_size = CELL_SIZE * 5.0  # 每个分区网格的大小
        self.num_grid_rows = int(np.ceil(self.map_rows * CELL_SIZE / self.grid_partition_size))
        self.num_grid_cols = int(np.ceil(self.map_cols * CELL_SIZE / self.grid_partition_size))
        self.obstacle_spatial_grid = [[[] for _ in range(self.num_grid_cols)] for _ in range(self.num_grid_rows)]
        self._populate_obstacle_spatial_grid()  # 填充空间网格

        # --- 优化: 射线投射用于障碍物观测 ---
        self.num_rays = 8  # 射线数量
        self.ray_directions = []  # 存储射线方向向量
        for angle_deg in np.linspace(0, 360, self.num_rays, endpoint=False):
            angle_rad = np.deg2rad(angle_deg)
            self.ray_directions.append(
                np.array([np.cos(angle_rad), np.sin(angle_rad)], dtype=np.float32))  # 确保是 float32
        self.max_obs_range = self.map_bounds[1] - self.map_bounds[0]  # 最大观测范围，这里简单设为地图宽度

    def _populate_obstacle_spatial_grid(self):
        """
        使用障碍物段填充 self.obstacle_spatial_grid。
        每个网格单元将包含与其重叠的障碍物段列表。
        """
        for obs_segment in self.obstacles:
            p1, p2 = obs_segment[0], obs_segment[1]  # obs_segment 已经是 [np.array, np.array]
            # 确定障碍物段的最小/最大 x,y 坐标
            min_x, max_x = min(p1[0], p2[0]), max(p1[0], p2[0])
            min_y, max_y = min(p1[1], p2[1]), max(p1[1], p2[1])

            # 确定障碍物段覆盖的网格单元范围
            min_grid_c = max(0, int(min_x / self.grid_partition_size))
            max_grid_c = min(self.num_grid_cols - 1, int(max_x / self.grid_partition_size))
            min_grid_r = max(0, int(min_y / self.grid_partition_size))
            max_grid_r = min(self.num_grid_rows - 1, int(max_y / self.grid_partition_size))

            # 将障碍物添加到所有重叠的网格单元中
            for r in range(min_grid_r, max_grid_r + 1):
                for c in range(min_grid_c, max_grid_c + 1):
                    # 将障碍物作为元组添加，方便之后去重
                    self.obstacle_spatial_grid[r][c].append(obs_segment)

        # 对每个网格单元中的障碍物列表去重
        for r in range(self.num_grid_rows):
            for c in range(self.num_grid_cols):
                # 转换为元组以使用 set 确保唯一性，然后转回列表
                self.obstacle_spatial_grid[r][c] = list(
                    set(tuple(map(tuple, s)) for s in self.obstacle_spatial_grid[r][c]))
                # 转换回列表 of numpy arrays
                self.obstacle_spatial_grid[r][c] = [[np.array(p, dtype=np.float32) for p in s] for s in
                                                    self.obstacle_spatial_grid[r][c]]

    def _get_nearby_obstacles(self, pos, search_radius):
        """
        从空间网格中检索给定位置附近的障碍物段。
        pos: 搜索中心点的 (x, y) 坐标。
        search_radius: 围绕 'pos' 考虑的障碍物搜索半径。
        """
        center_c = int(pos[0] / self.grid_partition_size)
        center_r = int(pos[1] / self.grid_partition_size)

        # 计算需要检查的网格单元偏移量
        cells_to_check_offset = int(np.ceil(search_radius / self.grid_partition_size)) + 1

        nearby_obstacles = set()
        for r_offset in range(-cells_to_check_offset, cells_to_check_offset + 1):
            for c_offset in range(-cells_to_check_offset, cells_to_check_offset + 1):
                grid_r, grid_c = center_r + r_offset, center_c + c_offset
                if 0 <= grid_r < self.num_grid_rows and 0 <= grid_c < self.num_grid_cols:
                    for obs in self.obstacle_spatial_grid[grid_r][grid_c]:
                        nearby_obstacles.add(tuple(map(tuple, obs)))
        # 确保返回的障碍物段中的点也是 float32
        return [[np.array(p, dtype=np.float32) for p in obs] for obs in nearby_obstacles]

    def _is_walkable(self, pos):
        """
        检查一个点是否在地图边界内且不是墙单元。
        """
        x, y = pos
        if not (self.map_bounds[0] <= x <= self.map_bounds[1] and
                self.map_bounds[2] <= y <= self.map_bounds[3]):
            return False

        grid_r = int(y / CELL_SIZE)
        grid_c = int(x / CELL_SIZE)

        if not (0 <= grid_r < self.map_rows and 0 <= grid_c < self.map_cols):
            return False

        return MAP_STRINGS[grid_r][grid_c] != '0'

    def reset(self):
        """
        重置环境，初始化所有代理的起点和目标位置。
        第一轮随机生成起点和可达目标，后续轮次使用第一轮的固定位置和目标。
        """
        global _initial_agv_start_positions, _initial_agv_goals, _initial_agv_positions_set, _initial_agv_initial_a_star_distances

        self.agents = []

        if len(self.pure_walkable_coords) < self.num_agents:
            raise ValueError(
                f"地图上的纯可通行位置数量 ({len(self.pure_walkable_coords)}) 小于代理数量 ({self.num_agents})，无法分配唯一的起点。请调整地图或代理数量。")
        if len(self.goal_coords) < self.num_agents:
            raise ValueError(
                f"地图上的目标位置数量 ({len(self.goal_coords)}) 小于代理数量 ({self.num_agents})，无法分配唯一目标。请调整地图或代理数量。")

        start_positions = []
        assigned_goals = []
        initial_a_star_distances_local_copy = []

        # 如果是第一次 reset 或需要重新生成起始位置/目标
        if not _initial_agv_positions_set:
            # 随机选择唯一的起始位置
            start_positions_idx = random.sample(range(len(self.pure_walkable_coords)), self.num_agents)
            # 确保复制时保持 float32 类型
            start_positions = [self.pure_walkable_coords[i].copy().astype(np.float32) for i in start_positions_idx]

            # 存储初始生成的位置，以便后续 reset 使用
            _initial_agv_start_positions = [pos.copy() for pos in start_positions]

            temp_assigned_goals = []  # 临时存储已分配的目标，确保唯一性
            for i in range(self.num_agents):
                current_start_pos = start_positions[i]

                potential_goals_with_dist = []
                for goal_candidate in self.goal_coords:
                    # 检查目标是否已被分配
                    is_assigned = False
                    for assigned_g in temp_assigned_goals:
                        if np.array_equal(goal_candidate, assigned_g):
                            is_assigned = True
                            break
                    if not is_assigned:
                        dist = np.linalg.norm(current_start_pos - goal_candidate)
                        potential_goals_with_dist.append((dist, goal_candidate))

                # 优先选择距离最近且 A* 可达的目标
                potential_goals_with_dist.sort(key=lambda x: x[0])  # 按距离排序

                selected_goal = None
                current_agent_initial_a_star_dist = 0.0
                for dist, goal_candidate in potential_goals_with_dist:
                    path_check = self.astar_finder.find_path(current_start_pos, goal_candidate)
                    if path_check is not None:
                        selected_goal = goal_candidate
                        total_path_len = 0
                        for k in range(len(path_check) - 1):
                            total_path_len += np.linalg.norm(path_check[k] - path_check[k + 1])
                        current_agent_initial_a_star_dist = total_path_len
                        break  # 找到第一个可达目标就停止

                if selected_goal is None:
                    # 如果没有最近的可达目标，从剩余未分配的目标中随机选择一个
                    unassigned_random_goals = [g for g in self.goal_coords if not any(
                        np.array_equal(g, assigned_g) for assigned_g in temp_assigned_goals)]
                    if unassigned_random_goals:
                        selected_goal = random.choice(unassigned_random_goals)
                        print(
                            f"警告: 代理 {i} 无法从 {current_start_pos.round(2)} 到任何最近的可达目标找到 A* 路径。分配随机未分配目标 {selected_goal.round(2)}。")
                        current_agent_initial_a_star_dist = np.linalg.norm(
                            current_start_pos - selected_goal)  # 使用直线距离作为A*距离
                    else:
                        print("严重警告: 所有目标均已分配或不可达，代理将被分配一个随机目标 (可能重复)。")
                        selected_goal = random.choice(self.goal_coords)
                        current_agent_initial_a_star_dist = np.linalg.norm(
                            current_start_pos - selected_goal)  # 使用直线距离作为A*距离

                temp_assigned_goals.append(selected_goal)
                # 确保复制时保持 float32 类型
                assigned_goals.append(selected_goal.copy().astype(np.float32))
                initial_a_star_distances_local_copy.append(current_agent_initial_a_star_dist)

            _initial_agv_goals = [goal.copy() for goal in assigned_goals]
            _initial_agv_initial_a_star_distances = [d for d in initial_a_star_distances_local_copy]
            _initial_agv_positions_set = True  # 标记已设置初始位置
            print(f"首次生成 AGV 目标位置: {[p.round(2) for p in _initial_agv_goals]}")
            print("\n首次生成 AGV 起点和目标位置，A* 寻路只在此刻运行。")
        else:
            # 后续 reset 直接使用之前保存的固定位置和目标
            if (_initial_agv_start_positions is None or len(_initial_agv_start_positions) != self.num_agents or
                    _initial_agv_goals is None or len(_initial_agv_goals) != self.num_agents or
                    _initial_agv_initial_a_star_distances is None or len(
                        _initial_agv_initial_a_star_distances) != self.num_agents):
                raise ValueError(
                    "训练期间代理数量发生变化，或初始固定位置/目标/A* 距离未正确设置。请重启训练或确保 num_agents 保持不变。")
            start_positions = [pos.copy().astype(np.float32) for pos in _initial_agv_start_positions]
            assigned_goals = [goal.copy().astype(np.float32) for goal in _initial_agv_goals]
            initial_a_star_distances_local_copy = [d for d in _initial_agv_initial_a_star_distances]

        for i in range(self.num_agents):
            self.agents.append({
                'id': i,
                'pos': start_positions[i].copy(),  # 代理当前位置
                'goal': assigned_goals[i].copy(),  # 代理目标位置
                'step_count': 0,  # 当前 episode 已走的步数
                'prev_dist': np.linalg.norm(start_positions[i] - assigned_goals[i]),  # 上一步到目标的距离
                'prev_action': np.zeros(2, dtype=np.float32),  # 上一步的动作
                'reached_goal': False,  # 是否已到达目标
                'path_history': [start_positions[i].copy()],  # 路径历史记录
                'initial_a_star_dist': initial_a_star_distances_local_copy[i]  # 初始A*距离
            })
        return self._get_all_states()

    def _get_agent_state(self, agent_idx):
        """
        获取单个代理的观测状态。
        agent_idx: 代理的索引。
        状态向量包括：[当前位置 (2), 目标位置 (2), 距离目标 (1), 射线投射距离 (num_rays), 其他代理相对信息 (3 * (num_agents - 1))]
        """
        agent = self.agents[agent_idx]
        current_pos = agent['pos']
        goal = agent['goal']

        dist_to_goal = np.linalg.norm(current_pos - goal)

        # 射线投射距离到障碍物
        ray_cast_distances = []
        # 只检索代理附近区域的障碍物，以减少计算量
        relevant_obstacles_for_rays = self._get_nearby_obstacles(current_pos, self.max_obs_range)

        for ray_dir in self.ray_directions:
            min_dist_along_ray = np.inf
            # 使用 Numba 加速的射线交点检测函数
            for obs in relevant_obstacles_for_rays:
                p1, p2 = obs[0], obs[1]  # 确保 p1, p2 已经是 float32
                dist = _ray_segment_intersect_numba(current_pos, ray_dir, p1, p2)
                min_dist_along_ray = min(min_dist_along_ray, dist)

            # 归一化距离，并裁剪到观测范围
            normalized_dist = np.clip(min_dist_along_ray, 0, self.max_obs_range) / self.max_obs_range
            ray_cast_distances.append(normalized_dist)

        # 其他代理的相对位置和距离
        other_agents_relative_info = []
        for i, other_agent in enumerate(self.agents):
            if i != agent_idx:  # 排除自身
                relative_pos = other_agent['pos'] - current_pos
                dist_other_agent = np.linalg.norm(relative_pos)
                normalized_relative_pos = relative_pos / (dist_other_agent + 1e-6)  # 避免除以零
                normalized_dist_other_agent = dist_other_agent / (self.map_bounds[1] - self.map_bounds[0])  # 归一化
                other_agents_relative_info.extend(normalized_relative_pos.astype(np.float32))  # 确保是 float32
                other_agents_relative_info.append(normalized_dist_other_agent.astype(np.float32))  # 确保是 float32

        # 填充以保持一致的维度，即使代理数量变化
        # 每个其他代理提供 2 (相对位置) + 1 (距离) = 3 维信息
        max_other_agents_info_dim = (self.num_agents - 1) * 3
        flattened_other_agent_info = np.array(other_agents_relative_info).flatten()
        if len(flattened_other_agent_info) < max_other_agents_info_dim:
            flattened_other_agent_info = np.pad(flattened_other_agent_info,
                                                (0, max_other_agents_info_dim - len(flattened_other_agent_info)),
                                                'constant')  # 用常数（0）填充

        # 归一化当前位置和目标位置
        normalized_current_pos = current_pos / np.array([self.map_bounds[1], self.map_bounds[3]], dtype=np.float32)
        normalized_goal = goal / np.array([self.map_bounds[1], self.map_bounds[3]], dtype=np.float32)
        normalized_dist_to_goal = np.array([dist_to_goal / (self.map_bounds[1] - self.map_bounds[0])],
                                           dtype=np.float32)  # 确保是 float32

        state = np.concatenate([
            normalized_current_pos,
            normalized_goal,
            normalized_dist_to_goal,
            np.array(ray_cast_distances, dtype=np.float32),  # 确保是 float32
            flattened_other_agent_info
        ])
        return state.astype(np.float32)  # 确保返回 float32 类型

    def _get_all_states(self):
        """
        获取所有代理的观测状态列表 (局部状态)。
        """
        return [self._get_agent_state(i) for i in range(self.num_agents)]

    def _get_global_state(self):
        """
        获取所有代理的全局观测状态 (所有局部状态的拼接)。
        """
        all_local_states = self._get_all_states()
        return np.concatenate(all_local_states).astype(np.float32)  # 确保返回 float32 类型

    def _check_collision_with_agents(self, agent_idx, new_pos):
        """
        检查代理是否与环境中其他代理发生碰撞。
        agent_idx: 当前代理的索引。
        new_pos: 当前代理计划到达的新位置。
        """
        for i, other_agent in enumerate(self.agents):
            if i != agent_idx:  # 排除自身
                if np.linalg.norm(new_pos - other_agent['pos']) < (2 * self.agent_radius):
                    return True  # 发生碰撞
        return False

    def step(self, actions):
        """
        环境向前推进一个时间步。
        actions: 包含所有代理动作的列表，每个动作是一个 (vx, vy) 向量。
        """
        rewards = [0.0] * self.num_agents
        dones = [False] * self.num_agents
        infos = [{}] * self.num_agents

        old_positions = [agent['pos'].copy() for agent in self.agents]  # 记录所有代理的旧位置

        for i in range(self.num_agents):
            agent = self.agents[i]

            # 如果代理已经到达目标或者已经完成（例如，达到最大步数），则不进行移动和奖励计算
            if agent['reached_goal'] or agent['step_count'] >= self.max_steps_per_episode:
                rewards[i] = 0.0  # 不再获得奖励或惩罚
                dones[i] = True  # 标记为完成
                infos[i] = {
                    'steps': agent['step_count'],
                    'reached_goal': agent['reached_goal'],
                    'curr_dist': 0.0 if agent['reached_goal'] else np.linalg.norm(agent['pos'] - agent['goal']),
                    # 如果已到达，距离为0
                    'collided_static': False,
                    'collided_agents': False,
                    'in_bounds': True
                }
                continue

            agent['step_count'] += 1  # 步数增加

            current_reward = -0.1  # 每步的时间惩罚

            action = actions[i].astype(np.float32)  # 获取当前代理的动作，并确保是 float32

            # 应用最大速度限制 (这里动作向量的模长被限制为1，然后乘以max_speed)
            # 确保动作的模长不超过1，然后乘以最大速度
            action_magnitude = np.linalg.norm(action)
            if action_magnitude > 1.0:
                action = action / action_magnitude

            move = action * self.max_speed  # 计算实际移动向量，确保是 float32

            # 转弯角度限制
            # 确保 prev_action 和 move 都不是零向量，避免除以零
            if np.linalg.norm(agent['prev_action']) > 1e-6 and np.linalg.norm(move) > 1e-6:
                prev_dir = agent['prev_action'] / np.linalg.norm(agent['prev_action'])
                curr_dir = move / np.linalg.norm(move)
                angle_diff = np.arccos(np.clip(np.dot(prev_dir, curr_dir), -1.0, 1.0))  # 夹角

                if angle_diff > self.max_turn_angle:
                    # 如果转弯角度过大，将当前移动方向旋转到最大允许角度
                    # 计算叉积以确定旋转方向 (顺时针/逆时针)
                    cross_product = prev_dir[0] * curr_dir[1] - prev_dir[1] * curr_dir[0]
                    rot_sign = -1 if cross_product < 0 else 1  # 确定旋转方向
                    rot_matrix = np.array([
                        [np.cos(rot_sign * self.max_turn_angle), -np.sin(rot_sign * self.max_turn_angle)],
                        [np.sin(rot_sign * self.max_turn_angle), np.cos(rot_sign * self.max_turn_angle)]
                    ], dtype=np.float32)  # 确保旋转矩阵是 float32
                    new_move_dir = rot_matrix @ prev_dir
                    move = np.linalg.norm(move) * new_move_dir

            prospective_pos = old_positions[i] + move  # 计算预期新位置

            # 边界检查
            in_bounds = (self.map_bounds[0] + self.agent_radius <= prospective_pos[0] <= self.map_bounds[
                1] - self.agent_radius and
                         self.map_bounds[2] + self.agent_radius <= prospective_pos[1] <= self.map_bounds[
                             3] - self.agent_radius)

            # 获取相关障碍物，然后传递给 Numba 加速的碰撞检测函数
            # 搜索半径需要包含代理的移动距离和代理半径
            search_radius_for_collision = np.linalg.norm(move) + self.agent_radius
            relevant_obstacles_for_collision = self._get_nearby_obstacles(prospective_pos, search_radius_for_collision)

            # 将相关障碍物转换为 Numba 友好的 NumPy 数组形式 (列表的列表 => 2D数组)
            # Numba 的 @jit(nopython=True) 函数不支持直接传入 Python 列表的列表，需要转换为 NumPy 数组
            # 确保转换为 float32
            relevant_obstacles_np = np.array(relevant_obstacles_for_collision, dtype=np.float32)

            collision_static = False
            if len(relevant_obstacles_np) > 0:  # 只有当附近有障碍物时才进行碰撞检测
                collision_static = check_collision_with_obstacles_numba(old_positions[i].astype(np.float32),
                                                                        # 确保传入 float32
                                                                        prospective_pos.astype(np.float32),
                                                                        # 确保传入 float32
                                                                        relevant_obstacles_np,
                                                                        self.agent_radius)

            collision_agents = self._check_collision_with_agents(i, prospective_pos)  # 检查与其他代理的碰撞

            agent_done = False
            agent_reached_goal = False

            if not in_bounds or collision_static or collision_agents:
                current_reward -= 50.0  # 碰撞或出界惩罚增加
                # 发生碰撞或出界，代理位置保持不变
                pass  # 位置不更新，保持在原位
            else:
                self.agents[i]['pos'] = prospective_pos  # 更新代理位置

            # 距离目标奖励/惩罚
            curr_dist = np.linalg.norm(self.agents[i]['pos'] - agent['goal'])
            dist_diff = agent['prev_dist'] - curr_dist  # 距离目标的减少量

            # 基于 A* 初始距离的距离奖励
            if agent['initial_a_star_dist'] > 1e-6:  # 避免除以零
                current_reward += 150.0 * (dist_diff / agent['initial_a_star_dist'])  # 系数增加
            else:
                current_reward += 150.0 * dist_diff  # 如果A*距离为0 (即初始就在目标)，则直接用距离差

            self.agents[i]['prev_dist'] = curr_dist  # 更新上一步距离

            # 到达目标奖励
            if curr_dist < self.agent_radius * 1.5:  # 考虑到代理半径，放宽一点范围
                current_reward += 1000.0  # 到达目标奖励增加
                agent_done = True  # 标记为完成
                agent_reached_goal = True
                self.agents[i]['reached_goal'] = True

            # 达到最大步数惩罚
            if agent['step_count'] >= self.max_steps_per_episode:
                agent_done = True
                if not agent_reached_goal:  # 如果未到达目标就达到最大步数
                    current_reward -= 200.0  # 未在最大步数内到达目标惩罚增加

            # 代理之间接近惩罚
            proximity_penalty = 0.0
            for j, other_agent in enumerate(self.agents):
                if i != j and not other_agent['reached_goal']:  # 只对未到达目标的代理进行接近惩罚
                    dist_to_other_agent = np.linalg.norm(self.agents[i]['pos'] - other_agent['pos'])
                    # 如果两个代理距离很近且不重叠
                    if dist_to_other_agent < 2.0 * self.agent_radius and dist_to_other_agent > 1e-6:
                        # 距离越近，惩罚越大，使用平方项使得惩罚更陡峭
                        proximity_penalty -= 30.0 * (
                                    1.0 - (dist_to_other_agent / (2.0 * self.agent_radius))) ** 2  # 系数增加并使用平方项
            current_reward += proximity_penalty

            # 动作平滑性惩罚 (鼓励平滑的动作)
            action_change = np.linalg.norm(actions[i] - agent['prev_action'])
            current_reward -= 0.1 * action_change
            self.agents[i]['prev_action'] = actions[i].copy()  # 更新上一步动作

            rewards[i] = np.clip(current_reward, -1000.0, 1100.0)  # 裁剪奖励范围
            dones[i] = agent_done
            infos[i] = {
                'steps': agent['step_count'],
                'reached_goal': agent_reached_goal,
                'curr_dist': curr_dist,
                'collided_static': collision_static,
                'collided_agents': collision_agents,
                'in_bounds': in_bounds
            }
            # 只有当代理位置发生变化时才记录路径历史，或者至少每步记录一次防止重复
            self.agents[i]['path_history'].append(self.agents[i]['pos'].copy())

        # 检查所有代理是否都已完成
        all_dones = all(d['reached_goal'] or d['steps'] >= self.max_steps_per_episode for d in infos)

        # 获取下一个状态
        current_local_states = self._get_all_states()
        current_global_state = self._get_global_state()

        return current_local_states, current_global_state, rewards, all_dones, infos

