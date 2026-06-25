import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# 必须在导入matplotlib.pyplot之前设置后端，适合无显示器的服务器环境
import matplotlib

matplotlib.use('Agg')
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import heapq

# ==========================================
# 0. 全局中文字体设置 (Matplotlib)
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 环境超参数与配置
# ==========================================
X_DIM, Y_DIM, Z_DIM = 16, 16, 4
NUM_AGENTS = 4
LIFTER_TIME = 3
ACTION_DIM = 7

STATE_DIM = 41
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR_BASE = "10000论结果更小学习率"
SAVE_DIR_TRAIN = os.path.join(SAVE_DIR_BASE, "Train")
SAVE_DIR_TEST = os.path.join(SAVE_DIR_BASE, "Test")

os.makedirs(SAVE_DIR_BASE, exist_ok=True)
os.makedirs(SAVE_DIR_TRAIN, exist_ok=True)
os.makedirs(SAVE_DIR_TEST, exist_ok=True)


class WarehouseEnv:
    def __init__(self):
        self.grid = np.zeros((X_DIM, Y_DIM, Z_DIM), dtype=int)
        self.lifter_pos = [(15, 4), (15, 12)]
        self.picking_stations = [(0, 4, 0)]
        self.storage_stations = [(0, 11, 0)]

        self.build_map()
        self.find_valid_picking_points()

        self.current_paths = [[] for _ in range(NUM_AGENTS)]
        self.current_dists = np.zeros(NUM_AGENTS, dtype=int)

        self.dones = np.zeros(NUM_AGENTS, dtype=bool)
        self.task_types = np.zeros(NUM_AGENTS, dtype=int)
        self.phases = np.zeros(NUM_AGENTS, dtype=int)

        self.targets = np.zeros((NUM_AGENTS, 3), dtype=int)
        self.initial_targets = np.zeros((NUM_AGENTS, 3), dtype=int)
        self.final_targets = np.zeros((NUM_AGENTS, 3), dtype=int)
        self.min_dists = np.zeros(NUM_AGENTS, dtype=int)

        self.reset()

    def build_map(self):
        for lx, ly in self.lifter_pos:
            self.grid[lx, ly, :] = 3
        for px, py, pz in self.picking_stations:
            self.grid[px, py, pz] = 4
        for sx, sy, sz in self.storage_stations:
            self.grid[sx, sy, sz] = 5

        for z in range(Z_DIM):
            for x in [1, 4, 7, 10, 13]:
                for y in [1, 5, 9, 13]:
                    for dx in range(2):
                        for dy in range(3):
                            nx, ny = x + dx, y + dy
                            if nx < X_DIM and ny < Y_DIM:
                                if (nx, ny) not in self.lifter_pos and \
                                        (nx, ny, z) not in self.picking_stations and \
                                        (nx, ny, z) not in self.storage_stations:
                                    self.grid[nx, ny, z] = 1

    def find_valid_picking_points(self):
        """
        [极度收敛版]：将起终点严格限制在特定的直道和巷道上，形成完美的鱼骨/流水线图
        """
        self.inbound_starts = []  # 入库起点池 (左侧主干道，靠近存储站)
        self.inbound_shelves = []  # 入库货架池 (仅限 x=3 巷道)

        self.outbound_starts = []  # 出库起点池 (左侧主干道，靠近分拣站)
        self.outbound_shelves = []  # 出库货架池 (仅限 x=9 巷道)

        self.idle_starts = []  # 空闲停靠池

        for z in range(Z_DIM):
            for x in range(X_DIM):
                for y in range(Y_DIM):
                    # 【划分起点池 (Z=0 的道路)】
                    if self.grid[x, y, z] == 0 and z == 0:
                        # 入库起点区：严格限制在 x=0 道路，y=8~15 区间
                        if x == 0 and 8 <= y <= 15:
                            self.inbound_starts.append((x, y, z))
                        # 出库起点区：严格限制在 x=0 道路，y=0~7 区间
                        elif x == 0 and 0 <= y <= 7:
                            self.outbound_starts.append((x, y, z))

                        # 💡【业务逻辑修改】：空载休息区转移到 x=12 辅助通道！
                        # 彻底让出 x=15 的提升机主通道，防止空载车堵死电梯口引发连环碰撞
                        elif x == 12:
                            self.idle_starts.append((x, y, z))

                    # 【划分终点货架池 (Z>0 的道路边缘)】
                    elif self.grid[x, y, z] == 0:
                        is_adjacent = False
                        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < X_DIM and 0 <= ny < Y_DIM:
                                if self.grid[nx, ny, z] == 1:
                                    is_adjacent = True
                                    break
                        if is_adjacent and z > 0:
                            # 入库终点：严格限制在 x=3 第一货架巷道
                            if x == 3:
                                self.inbound_shelves.append((x, y, z))
                            # 出库终点：严格限制在 x=9 第三货架巷道
                            elif x == 9:
                                self.outbound_shelves.append((x, y, z))

        # 兜底合并池
        self._all_starts = self.inbound_starts + self.outbound_starts + self.idle_starts
        self._all_shelves = self.inbound_shelves + self.outbound_shelves

    def _run_astar(self, start, target):
        start = tuple(start)
        target = tuple(target)
        if start == target:
            return [start]

        def heuristic(a, b):
            x1, y1, z1 = a
            x2, y2, z2 = b
            if z1 == z2:
                # 【打破 A* 平局防抖】：增加 0.001*abs(x1-x2) 的微小偏好权重
                return abs(x1 - x2) + abs(y1 - y2) + 0.001 * abs(x1 - x2)
            else:
                min_h = float('inf')
                for lx, ly in self.lifter_pos:
                    h = abs(x1 - lx) + abs(y1 - ly) + abs(z1 - z2) + abs(lx - x2) + abs(ly - y2)
                    h += 0.001 * abs(x1 - lx)
                    if h < min_h: min_h = h
                return min_h

        open_set = []
        heapq.heappush(open_set, (0, 0, start))
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current_g, current = heapq.heappop(open_set)
            if current == target:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            if current_g > g_score.get(current, float('inf')):
                continue

            cx, cy, cz = current
            neighbors = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < X_DIM and 0 <= ny < Y_DIM:
                    if self.grid[nx, ny, cz] != 1:
                        neighbors.append((nx, ny, cz))

            if (cx, cy) in self.lifter_pos:
                for dz in [1, -1]:
                    nz = cz + dz
                    if 0 <= nz < Z_DIM:
                        neighbors.append((cx, cy, nz))

            for neighbor in neighbors:
                tentative_g = current_g + 1
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, target)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        return [start]

    def reset(self):
        """
        按任务类型从专属区域无放回抽样起终点
        """
        self.task_types = np.random.choice([0, 1, 2], NUM_AGENTS)
        self.phases = np.zeros(NUM_AGENTS, dtype=int)

        self.agents_pos = np.zeros((NUM_AGENTS, 3), dtype=int)
        chosen_shelves = np.zeros((NUM_AGENTS, 3), dtype=int)

        # 记录已被占用的点，防止重叠
        used_starts = set()
        used_shelves = set()

        def sample_point(pool, used_set, fallback_pool):
            """无放回安全抽样工具函数"""
            available = [p for p in pool if tuple(p) not in used_set]
            if not available:  # 如果专属池抽干了，启用兜底池
                available = [p for p in fallback_pool if tuple(p) not in used_set]
            chosen = random.choice(available)
            used_set.add(tuple(chosen))
            return chosen

        # 分配起点和终点
        for i in range(NUM_AGENTS):
            task = self.task_types[i]

            if task == 1:  # 入库
                self.agents_pos[i] = sample_point(self.inbound_starts, used_starts, self._all_starts)
                chosen_shelves[i] = sample_point(self.inbound_shelves, used_shelves, self._all_shelves)

            elif task == 2:  # 出库
                self.agents_pos[i] = sample_point(self.outbound_starts, used_starts, self._all_starts)
                chosen_shelves[i] = sample_point(self.outbound_shelves, used_shelves, self._all_shelves)

            else:  # 空载/待命
                self.agents_pos[i] = sample_point(self.idle_starts, used_starts, self._all_starts)
                chosen_shelves[i] = sample_point(self.idle_starts, used_shelves, self._all_starts)

        # 按原有逻辑配置目标的 Target 切换机制
        for i in range(NUM_AGENTS):
            shelf_pos = chosen_shelves[i]
            pick_pos = self.picking_stations[0]
            store_pos = self.storage_stations[0]

            if self.task_types[i] == 0:
                self.targets[i] = shelf_pos
                self.final_targets[i] = shelf_pos
                self.phases[i] = 1
            elif self.task_types[i] == 1:
                self.targets[i] = store_pos
                self.final_targets[i] = shelf_pos
                self.phases[i] = 0
            elif self.task_types[i] == 2:
                self.targets[i] = shelf_pos
                self.final_targets[i] = pick_pos
                self.phases[i] = 0

            self.current_paths[i] = self._run_astar(self.agents_pos[i], self.targets[i])
            self.current_dists[i] = len(self.current_paths[i]) - 1
            self.min_dists[i] = self.current_dists[i]

        self.initial_targets = self.targets.copy()
        self.lifter_timers = {pos: 0 for pos in self.lifter_pos}
        self.lifter_busy_by = {pos: -1 for pos in self.lifter_pos}
        self.dones = np.zeros(NUM_AGENTS, dtype=bool)

        return self._get_states()

    def get_action_masks(self):
        masks = np.ones((NUM_AGENTS, ACTION_DIM), dtype=bool)
        for i in range(NUM_AGENTS):
            if self.dones[i]:
                masks[i, 1:] = False
                continue
            x, y, z = self.agents_pos[i]
            if y + 1 >= Y_DIM or self.grid[x, y + 1, z] == 1: masks[i, 1] = False
            if y - 1 < 0 or self.grid[x, y - 1, z] == 1: masks[i, 2] = False
            if x - 1 < 0 or self.grid[x - 1, y, z] == 1: masks[i, 3] = False
            if x + 1 >= X_DIM or self.grid[x + 1, y, z] == 1: masks[i, 4] = False

            xy_pos = (x, y)
            if xy_pos not in self.lifter_pos:
                masks[i, 5] = False
                masks[i, 6] = False
            else:
                if self.lifter_busy_by[xy_pos] != -1 and self.lifter_busy_by[xy_pos] != i:
                    masks[i, 5] = False
                    masks[i, 6] = False
                else:
                    if z + 1 >= Z_DIM: masks[i, 5] = False
                    if z - 1 < 0: masks[i, 6] = False
        return masks

    def _get_states(self):
        states = np.zeros((NUM_AGENTS, STATE_DIM))
        for i in range(NUM_AGENTS):
            pos = self.agents_pos[i]
            target = self.targets[i]
            norm_pos = pos / np.array([X_DIM, Y_DIM, Z_DIM])
            rel_target = (target - pos) / np.array([X_DIM, Y_DIM, Z_DIM])

            fov_static = np.zeros(9)
            fov_dynamic = np.zeros(9)
            idx = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny, nz = pos[0] + dx, pos[1] + dy, pos[2]
                    if nx < 0 or nx >= X_DIM or ny < 0 or ny >= Y_DIM or self.grid[nx, ny, nz] == 1:
                        fov_static[idx] = 1.0
                    else:
                        for j in range(NUM_AGENTS):
                            if i != j and not self.dones[j]:
                                if self.agents_pos[j][0] == nx and self.agents_pos[j][1] == ny and self.agents_pos[j][2] == nz:
                                    fov_dynamic[idx] = 1.0
                    idx += 1

            astar_d = self.current_dists[i]
            norm_astar = np.array([astar_d / 50.0])
            is_lifter = np.array([1.0 if (pos[0], pos[1]) in self.lifter_pos else 0.0])

            one_hot_id = np.zeros(NUM_AGENTS)
            one_hot_id[i] = 1.0
            task_one_hot = np.zeros(3)
            task_one_hot[self.task_types[i]] = 1.0
            phase_val = np.array([float(self.phases[i])])

            astar_hint = np.zeros(ACTION_DIM)
            if len(self.current_paths[i]) > 1:
                cx, cy, cz = pos
                nx, ny, nz = self.current_paths[i][1]
                if ny == cy + 1:
                    astar_hint[1] = 1.0
                elif ny == cy - 1:
                    astar_hint[2] = 1.0
                elif nx == cx - 1:
                    astar_hint[3] = 1.0
                elif nx == cx + 1:
                    astar_hint[4] = 1.0
                elif nz == cz + 1:
                    astar_hint[5] = 1.0
                elif nz == cz - 1:
                    astar_hint[6] = 1.0

            states[i] = np.concatenate([
                norm_pos, rel_target, fov_static, fov_dynamic,
                norm_astar, is_lifter, one_hot_id, task_one_hot, phase_val, astar_hint
            ])
        return states

    def step(self, actions):
        """
        【基于势能函数 (PBRS) 的严格收敛奖励机制】
        """
        # 💡【漏洞修复1】：加大基础移动惩罚，保证即便存在 PBRS 微小差值，也不能靠原地挂机获益
        rewards = np.full(NUM_AGENTS, -0.05)
        collisions = 0
        avoided_collisions = 0

        old_pos = self.agents_pos.copy()
        intended_pos = self.agents_pos.copy()

        # 1. 提升机状态更新
        for pos in self.lifter_pos:
            if self.lifter_timers[pos] > 0:
                self.lifter_timers[pos] -= 1
                if self.lifter_timers[pos] == 0:
                    self.lifter_busy_by[pos] = -1

        # 2. 意图位置计算
        for i in range(NUM_AGENTS):
            if self.dones[i]:
                rewards[i] = 0.0
                continue
            a = actions[i]
            if a == 0:
                continue
            elif a == 1:
                intended_pos[i, 1] += 1
            elif a == 2:
                intended_pos[i, 1] -= 1
            elif a == 3:
                intended_pos[i, 0] -= 1
            elif a == 4:
                intended_pos[i, 0] += 1
            elif a in [5, 6]:
                xy_pos = (old_pos[i, 0], old_pos[i, 1])
                if self.lifter_busy_by[xy_pos] == -1 or self.lifter_busy_by[xy_pos] == i:
                    self.lifter_busy_by[xy_pos] = i
                    if self.lifter_timers[xy_pos] == 0:
                        self.lifter_timers[xy_pos] = LIFTER_TIME
                    if a == 5: intended_pos[i, 2] += 1
                    if a == 6: intended_pos[i, 2] -= 1

        priority_weights = {1: 3000, 2: 2000, 0: 1000}
        scores = np.zeros(NUM_AGENTS)
        for i in range(NUM_AGENTS):
            scores[i] = priority_weights[self.task_types[i]] - self.current_dists[i]
        priority_order = np.argsort(-scores)

        # 3. 换位碰撞检测
        for p_idx in range(NUM_AGENTS):
            for q_idx in range(p_idx + 1, NUM_AGENTS):
                i = priority_order[p_idx]
                j = priority_order[q_idx]
                if self.dones[i] or self.dones[j]: continue
                if np.array_equal(intended_pos[i], old_pos[j]) and np.array_equal(intended_pos[j], old_pos[i]):
                    intended_pos[i] = old_pos[i]
                    intended_pos[j] = old_pos[j]
                    rewards[i] -= 1.0
                    rewards[j] -= 1.0
                    collisions += 1

        # 4. 抢占单元格仲裁
        reserved_cells = {}
        for i in range(NUM_AGENTS):
            if self.dones[i]:
                reserved_cells[tuple(old_pos[i].tolist())] = i

        for i in priority_order:
            if self.dones[i]: continue
            pos_tuple = tuple(intended_pos[i].tolist())
            if pos_tuple in reserved_cells:
                intended_pos[i] = old_pos[i]
                if actions[i] != 0:
                    rewards[i] -= 0.1
                    avoided_collisions += 1
                reserved_cells[tuple(old_pos[i].tolist())] = i
            else:
                reserved_cells[pos_tuple] = i

        # 5. 基于势能函数 (PBRS) 的进度奖励
        GAMMA_PBRS = 0.99
        PBRS_SCALE = 0.05

        new_dists = np.zeros(NUM_AGENTS, dtype=int)
        for i in range(NUM_AGENTS):
            if self.dones[i]: continue

            is_standing_still = (tuple(intended_pos[i]) == tuple(old_pos[i]))

            if len(self.current_paths[i]) > 1 and tuple(intended_pos[i]) == self.current_paths[i][1]:
                new_dists[i] = self.current_dists[i] - 1
            elif is_standing_still:
                new_dists[i] = self.current_dists[i]
            else:
                path = self._run_astar(intended_pos[i], self.targets[i])
                new_dists[i] = len(path) - 1

            old_dist = self.current_dists[i]
            new_dist = new_dists[i]

            if is_standing_still:
                # 💡【漏洞修复2】：如果这步小车原地不动（发呆/死锁），严惩并直接剥夺 PBRS 计算
                # 彻底封杀“距离远时靠打折扣(Gamma)白嫖正向奖励”的恶性捷径
                rewards[i] -= 0.1
            else:
                # 只有真正移动了，才给予势能变动奖励
                phi_old = -old_dist * PBRS_SCALE
                phi_new = -new_dist * PBRS_SCALE
                shaping_reward = (GAMMA_PBRS * phi_new) - phi_old
                rewards[i] += shaping_reward

            self.agents_pos[i] = intended_pos[i]
            self.current_paths[i] = self._run_astar(self.agents_pos[i], self.targets[i])
            self.current_dists[i] = len(self.current_paths[i]) - 1

            # 6. 稀疏目标奖励
            if new_dist == 0:
                if self.phases[i] == 0:
                    self.phases[i] = 1
                    self.targets[i] = self.final_targets[i]
                    self.current_paths[i] = self._run_astar(self.agents_pos[i], self.targets[i])
                    self.current_dists[i] = len(self.current_paths[i]) - 1
                    rewards[i] += 1.5
                elif self.phases[i] == 1:
                    self.dones[i] = True
                    rewards[i] += 3.0

        # 7. VDN 终极团队奖励
        if np.all(self.dones) and not np.all(self.dones == True):
            team_bonus = 5.0
            rewards += team_bonus / NUM_AGENTS

        # 兜底安全检测
        final_counts = {}
        for i in range(NUM_AGENTS):
            if not self.dones[i]:
                pt = tuple(self.agents_pos[i].tolist())
                final_counts[pt] = final_counts.get(pt, 0) + 1
        for i in range(NUM_AGENTS):
            if not self.dones[i]:
                pt = tuple(self.agents_pos[i].tolist())
                if final_counts[pt] > 1:
                    rewards[i] -= 1.0
                    collisions += 1

        return self._get_states(), rewards, self.dones.copy(), collisions, avoided_collisions


class VDN_Global_DQN(nn.Module):
    def __init__(self):
        super(VDN_Global_DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_DIM)
        )

    def forward(self, x): return self.net(x)


class JointReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.s = torch.zeros((capacity, NUM_AGENTS, STATE_DIM), dtype=torch.float32, device=device)
        self.a = torch.zeros((capacity, NUM_AGENTS), dtype=torch.int64, device=device)
        self.r = torch.zeros((capacity, NUM_AGENTS), dtype=torch.float32, device=device)
        self.ns = torch.zeros((capacity, NUM_AGENTS, STATE_DIM), dtype=torch.float32, device=device)
        self.nd = torch.zeros((capacity, NUM_AGENTS), dtype=torch.float32, device=device)
        self.nm = torch.zeros((capacity, NUM_AGENTS, ACTION_DIM), dtype=torch.bool, device=device)

    def push(self, states, actions, rewards, next_states, dones, next_masks):
        self.s[self.ptr] = torch.tensor(states, dtype=torch.float32, device=device)
        self.a[self.ptr] = torch.tensor(actions, dtype=torch.int64, device=device)
        self.r[self.ptr] = torch.tensor(rewards, dtype=torch.float32, device=device)
        self.ns[self.ptr] = torch.tensor(next_states, dtype=torch.float32, device=device)
        self.nd[self.ptr] = torch.tensor(dones, dtype=torch.float32, device=device)
        self.nm[self.ptr] = torch.tensor(next_masks, dtype=torch.bool, device=device)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, size=(batch_size,), device=device)
        return self.s[idx], self.a[idx], self.r[idx], self.ns[idx], self.nd[idx], self.nm[idx]


def plot_training_metrics(csv_path, save_dir):
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)

    metrics = [
        ('Total_Reward', '回合总奖励趋势(已缩放)', 'blue'),
        ('Success_Rate', '任务成功率趋势', 'green'),
        ('Total_Steps', '回合耗时(步数)趋势', 'orange'),
        ('Collisions', '实际物理碰撞次数趋势', 'red'),
        ('Avoided_Collisions', '仲裁避免的碰撞次数趋势', 'brown'),
        ('Avg_Loss', '网络 Loss 趋势', 'purple')
    ]

    for col, title, color in metrics:
        plt.figure(figsize=(10, 5))
        plt.plot(df['Episode'], df[col], label=f'{col} (原始值)', color=color, alpha=0.3)
        ma_10 = df[col].rolling(window=10, min_periods=1).mean()
        plt.plot(df['Episode'], ma_10, label='10轮平滑均线', color=color, linewidth=2)
        if col == 'Success_Rate' and 'Avg_Success_20' in df.columns:
            plt.plot(df['Episode'], df['Avg_Success_20'], label='20轮平滑均线', color='darkgreen', linewidth=2, linestyle='--')

        plt.title(title, fontsize=15)
        plt.xlabel('训练轮次 (Episode)', fontsize=12)
        plt.ylabel('数值', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        save_file = os.path.join(save_dir, f'Chart_{title.split()[0]}.png')
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        plt.close()


def generate_plotly_3d(env, trajectories, filename, title_text, final_targets, save_dir, task_types=None):
    fig_3d = go.Figure()
    traces_info = {}
    current_idx = 0

    task_names = {0: "空载", 1: "入库", 2: "出库"}

    traces_info['floors'] = []
    for z_layer in range(Z_DIM):
        fig_3d.add_trace(go.Surface(x=[0, 15], y=[0, 15], z=[[z_layer, z_layer], [z_layer, z_layer]],
                                    showscale=False, opacity=0.1, colorscale=[[0, 'cyan'], [1, 'cyan']], hoverinfo='skip', name=f'第 {z_layer} 层'))
        traces_info['floors'].append(current_idx)
        current_idx += 1

    shelves = np.where(env.grid == 1)
    fig_3d.add_trace(go.Scatter3d(x=shelves[0], y=shelves[1], z=shelves[2], mode='markers',
                                  marker=dict(size=7, symbol='square', color=shelves[2], colorscale='Blues', opacity=0.8, line=dict(width=1, color='DarkSlateGrey')), name='货架'))
    traces_info['shelves'] = [current_idx]
    current_idx += 1

    roads = np.where(env.grid == 0)
    fig_3d.add_trace(go.Scatter3d(x=roads[0], y=roads[1], z=roads[2], mode='markers',
                                  marker=dict(size=2, color='gray', opacity=0.3), name='道路', hoverinfo='skip'))
    traces_info['roads'] = [current_idx]
    current_idx += 1

    traces_info['facilities'] = []
    ps_x, ps_y, ps_z = [ps[0] for ps in env.picking_stations], [ps[1] for ps in env.picking_stations], [ps[2] for ps in env.picking_stations]
    fig_3d.add_trace(go.Scatter3d(x=ps_x, y=ps_y, z=ps_z, mode='markers',
                                  marker=dict(size=12, symbol='circle', color='gold', line=dict(width=2, color='darkorange')), name='分拣站'))
    traces_info['facilities'].append(current_idx)
    current_idx += 1

    ss_x, ss_y, ss_z = [ss[0] for ss in env.storage_stations], [ss[1] for ss in env.storage_stations], [ss[2] for ss in env.storage_stations]
    fig_3d.add_trace(go.Scatter3d(x=ss_x, y=ss_y, z=ss_z, mode='markers',
                                  marker=dict(size=12, symbol='diamond', color='cyan', line=dict(width=2, color='teal')), name='存储站'))
    traces_info['facilities'].append(current_idx)
    current_idx += 1

    lifters = np.where(env.grid == 3)
    fig_3d.add_trace(go.Scatter3d(x=lifters[0], y=lifters[1], z=lifters[2], mode='markers',
                                  marker=dict(size=8, symbol='cross', color='orange', opacity=0.8), name='提升机'))
    traces_info['facilities'].append(current_idx)
    current_idx += 1

    max_len = max([len(traj) for traj in trajectories.values()] + [1])
    padded_trajs = {i: (np.zeros((max_len, 3)) if len(traj) == 0 else np.array(traj + [traj[-1]] * (max_len - len(traj)))) for i, traj in trajectories.items()}

    colors = ['red', 'blue', 'purple', 'green']
    targ_arr = final_targets

    task_names = {0: "空载", 1: "入库", 2: "出库"}

    current_task_types = getattr(env, 'task_types', None)

    traces_info['agv_paths'] = []
    traces_info['agv_points'] = []
    dynamic_trace_indices = []

    for i in range(NUM_AGENTS):
        traj = padded_trajs[i]
        c = colors[i]
        if len(traj) > 0:
            t_name = ""
            if current_task_types is not None and i < len(current_task_types):
                t_type = current_task_types[i]
                t_name = f"({task_names.get(t_type, '')})"

            fig_3d.add_trace(go.Scatter3d(x=[traj[0, 0]], y=[traj[0, 1]], z=[traj[0, 2]], mode='markers',
                                          marker=dict(size=6, symbol='circle-open', line=dict(color=c, width=3)), name=f'车{i}{t_name} 起点'))
            traces_info['agv_points'].append(current_idx)
            current_idx += 1

            end_pt = targ_arr[i]
            fig_3d.add_trace(go.Scatter3d(x=[end_pt[0]], y=[end_pt[1]], z=[end_pt[2]], mode='markers',
                                          marker=dict(size=10, symbol='diamond', color=c), name=f'车{i}{t_name} 目标货架'))
            traces_info['agv_points'].append(current_idx)
            current_idx += 1

            fig_3d.add_trace(go.Scatter3d(x=[traj[0, 0]], y=[traj[0, 1]], z=[traj[0, 2]], mode='lines',
                                          line=dict(width=5, color=c), name=f'车{i}{t_name} 轨迹'))
            traces_info['agv_paths'].append(current_idx)
            dynamic_trace_indices.append(current_idx)
            current_idx += 1

            fig_3d.add_trace(go.Scatter3d(x=[traj[0, 0]], y=[traj[0, 1]], z=[traj[0, 2]], mode='markers',
                                          marker=dict(size=6, color=c), showlegend=False, name=f'车头'))
            traces_info['agv_paths'].append(current_idx)
            dynamic_trace_indices.append(current_idx)
            current_idx += 1

    frames = []
    for t in range(max_len):
        frame_data = []
        for i in range(NUM_AGENTS):
            traj = padded_trajs[i]
            if len(traj) > 0:
                frame_data.append(go.Scatter3d(x=traj[:t + 1, 0], y=traj[:t + 1, 1], z=traj[:t + 1, 2]))
                frame_data.append(go.Scatter3d(x=[traj[t, 0]], y=[traj[t, 1]], z=[traj[t, 2]]))
        frames.append(go.Frame(data=frame_data, traces=dynamic_trace_indices, name=f'Step {t}'))
    fig_3d.frames = frames

    total_traces = current_idx

    def create_visible_array(hide_keys):
        arr = [True] * total_traces
        for key in hide_keys:
            for idx in traces_info[key]:
                arr[idx] = False
        return arr

    updatemenus = [
        dict(type="buttons", showactive=False, y=0, x=0.05, xanchor="right", yanchor="top", pad=dict(t=15, r=10), buttons=[
            dict(label="▶ 播放", method="animate", args=[None, dict(frame=dict(duration=150, redraw=True), transition=dict(duration=0), fromcurrent=True, mode='immediate')]),
            dict(label="⏸ 暂停", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate', transition=dict(duration=0))])
        ]),
        dict(type="buttons", direction="right", showactive=True, y=1.05, x=0.0, xanchor="left", yanchor="bottom", buttons=[
            dict(label="显示全部", method="update", args=[{"visible": create_visible_array([])}]),
            dict(label="隐藏道路", method="update", args=[{"visible": create_visible_array(['roads'])}]),
            dict(label="隐藏货架", method="update", args=[{"visible": create_visible_array(['shelves'])}]),
            dict(label="仅看起终点", method="update", args=[{"visible": create_visible_array(['floors', 'roads', 'shelves', 'agv_paths'])}]),
        ])
    ]

    fig_3d.update_layout(
        title=title_text,
        scene=dict(xaxis=dict(range=[-0.5, 15.5], title='X', dtick=3), yaxis=dict(range=[-0.5, 15.5], title='Y', dtick=3), zaxis=dict(range=[-0.5, 3.5], title='Z', dtick=1), aspectmode='manual',
                   aspectratio=dict(x=1, y=1, z=0.5)),
        margin=dict(l=0, r=0, b=0, t=40),
        updatemenus=updatemenus,
        sliders=[dict(currentvalue={"prefix": "Step: "}, pad={"t": 35}, len=0.9, x=0.1, y=0, xanchor="left", yanchor="top", steps=[
            dict(method="animate", label=str(t), args=[[f'Step {t}'], dict(mode="immediate", frame=dict(duration=150, redraw=True), transition=dict(duration=0))]) for t in range(max_len)
        ])]
    )
    html_path = os.path.join(save_dir, filename)
    fig_3d.write_html(html_path)


def save_static_3d_images(trajectories, initial_targets, final_targets, task_types, base_filename, title_prefix, save_dir, lifter_pos):
    colors = ['red', 'blue', 'purple', 'green']
    task_names = {0: "空载", 1: "入库", 2: "出库"}
    lifter_color = 'orange'

    def is_on_lifter(x, y):
        return (x, y) in lifter_pos

    def draw_plot(agent_idx=None):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=38, azim=35)

        try:
            ax.set_box_aspect((X_DIM, Y_DIM, Z_DIM * 2))
        except AttributeError:
            pass

        ax.set_xlim(0, X_DIM - 1)
        ax.set_ylim(0, Y_DIM - 1)
        ax.set_zlim(0, Z_DIM - 1)
        ax.set_xticks(np.arange(0, X_DIM, 2))
        ax.set_yticks(np.arange(0, Y_DIM, 2))
        ax.set_zticks(np.arange(0, Z_DIM, 1))

        ax.set_xlabel('X (宽度)')
        ax.set_ylabel('Y (深度)')
        ax.set_zlabel('Z (高度/层级)')

        x_mesh, y_mesh = np.meshgrid(np.arange(0, X_DIM), np.arange(0, Y_DIM))
        for z_level in range(Z_DIM):
            ax.plot_surface(x_mesh, y_mesh, np.full_like(x_mesh, z_level),
                            color='cyan', alpha=0.05, edgecolor='black', linewidth=0.1)

        title = title_prefix + (" (全局视角)" if agent_idx is None else f" (车{agent_idx} 视角)")
        ax.set_title(title, pad=20, fontsize=14, fontweight='bold')

        agents_to_plot = range(NUM_AGENTS) if agent_idx is None else [agent_idx]

        for i in agents_to_plot:
            traj = np.array(trajectories[i])
            if len(traj) < 2: continue

            c = colors[i]
            t_name = task_names[task_types[i]]

            normal_path_x, normal_path_y, normal_path_z = [], [], []
            lifter_path_x, lifter_path_y, lifter_path_z = [], [], []

            for t in range(len(traj) - 1):
                p1 = traj[t]
                p2 = traj[t + 1]
                if list(p1) == list(p2): continue

                is_lifter_move = (p1[0] == p2[0] and p1[1] == p2[1] and is_on_lifter(p1[0], p1[1]) and p1[2] != p2[2])

                if is_lifter_move:
                    lifter_path_x.extend([p1[0], p2[0], np.nan])
                    lifter_path_y.extend([p1[1], p2[1], np.nan])
                    lifter_path_z.extend([p1[2], p2[2], np.nan])
                else:
                    normal_path_x.extend([p1[0], p2[0], np.nan])
                    normal_path_y.extend([p1[1], p2[1], np.nan])
                    normal_path_z.extend([p1[2], p2[2], np.nan])

            if len(normal_path_x) > 0:
                ax.plot(normal_path_x, normal_path_y, normal_path_z, color=c, linewidth=2.5, alpha=0.8, label=f'车{i}({t_name}) 路线')
            if len(lifter_path_x) > 0:
                ax.plot(lifter_path_x, lifter_path_y, lifter_path_z, color=lifter_color, linewidth=2.5, alpha=0.9, linestyle='--', label=f'车{i}({t_name}) 提升机路径')

            ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color=c, marker='o', s=120, edgecolors='black', linewidth=1.5, zorder=5, label=f'车{i} 起点')
            end_pt = final_targets[i]
            ax.scatter(end_pt[0], end_pt[1], end_pt[2], color=c, marker='*', s=250, edgecolors='black', linewidth=1.5, zorder=6, label=f'车{i} 目标货架')

            if task_types[i] == 1 and initial_targets is not None:
                via_pt = initial_targets[i]
                ax.scatter(via_pt[0], via_pt[1], via_pt[2], color=c, marker='s', s=100, edgecolors='black', linewidth=1.5, zorder=5, label=f'车{i} 存储站')

            if task_types[i] == 2 and initial_targets is not None:
                via_pt = initial_targets[i]
                ax.scatter(via_pt[0], via_pt[1], via_pt[2], color=c, marker='s', s=100, edgecolors='black', linewidth=1.5, zorder=5, label=f'车{i} 分拣站')

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize=10)

        suffix = "_all" if agent_idx is None else f"_agent{agent_idx}"
        plt.savefig(os.path.join(save_dir, f"{base_filename}{suffix}.png"), dpi=200, bbox_inches='tight')
        plt.close()

    draw_plot(None)
    for i in range(NUM_AGENTS):
        draw_plot(i)


def train():
    print("\n" + "=" * 50)
    print("🚀 启动 [VDN 深度强化学习 - 优化震荡版] 🚀")
    print("=" * 50)

    env = WarehouseEnv()
    global_q = VDN_Global_DQN().to(device)
    global_target_q = VDN_Global_DQN().to(device)
    global_target_q.load_state_dict(global_q.state_dict())

    optimizer = optim.Adam(global_q.parameters(), lr=5e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.90)
    buffer = JointReplayBuffer(capacity=200000)

    EPOCHS = 10000
    BATCH_SIZE = 256
    GAMMA = 0.99
    TAU = 0.005

    epsilon = 1.0
    epsilon_decay = 0.9985
    epsilon_min = 0.005

    UPDATE_FREQ = 5
    MAX_STEPS = 150

    metrics_log = []
    recent_success_rates = deque(maxlen=20)
    best_avg_success = 0.0
    top_5_episodes = []

    pbar = tqdm(range(1, EPOCHS + 1), desc="训练进度")

    for epoch in pbar:
        if epoch >= 100:
            recent_success = [log['Success_Rate'] for log in metrics_log[-10:]]
            if np.mean(recent_success) <= 0.05 and epsilon < 0.05:
                epsilon = 0.15
                tqdm.write(f"⚠️ 陷入重度局部最优，触发 Epsilon 软回弹 (至 {epsilon})！")

        states = env.reset()
        episode_reward = 0
        step_count = 0
        ep_collisions = 0
        ep_avoided_collisions = 0
        ep_losses = []
        agent_steps = np.zeros(NUM_AGENTS, dtype=int)
        trajectories = {i: [list(env.agents_pos[i])] for i in range(NUM_AGENTS)}

        while not all(env.dones) and step_count < MAX_STEPS:
            actions = np.zeros(NUM_AGENTS, dtype=int)
            masks = env.get_action_masks()
            state_tensors = torch.FloatTensor(states).to(device)

            with torch.no_grad():
                q_vals_all = global_q(state_tensors)

            for i in range(NUM_AGENTS):
                if env.dones[i]:
                    actions[i] = 0
                else:
                    agent_steps[i] += 1
                    q_vals = q_vals_all[i].clone()
                    invalid_mask = torch.tensor(~masks[i], device=device)
                    q_vals.masked_fill_(invalid_mask, -1e9)

                    if random.random() < epsilon:
                        valid_actions = np.where(masks[i])[0]
                        actions[i] = random.choice(valid_actions)
                    else:
                        actions[i] = q_vals.argmax().item()

            next_states, rewards, new_dones, colls, avoided_colls = env.step(actions)

            for i in range(NUM_AGENTS):
                if not env.dones[i]:
                    trajectories[i].append(list(env.agents_pos[i]))
                elif len(trajectories[i]) > 0 and trajectories[i][-1] != list(env.final_targets[i]):
                    trajectories[i].append(list(env.final_targets[i]))

            next_masks = env.get_action_masks()
            ep_collisions += colls
            ep_avoided_collisions += avoided_colls
            episode_reward += np.sum(rewards)

            buffer.push(states, actions, rewards, next_states, new_dones, next_masks)
            states = next_states
            step_count += 1

            if step_count % UPDATE_FREQ == 0 or all(env.dones):
                if buffer.size > BATCH_SIZE:
                    b_s, b_a, b_r, b_ns, b_nd, b_nm = buffer.sample(BATCH_SIZE)
                    b_s_flat = b_s.view(BATCH_SIZE * NUM_AGENTS, STATE_DIM)
                    q_vals_flat = global_q(b_s_flat)
                    q_vals = q_vals_flat.view(BATCH_SIZE, NUM_AGENTS, ACTION_DIM)
                    b_a_unsqueezed = b_a.unsqueeze(-1)
                    chosen_q_vals = q_vals.gather(2, b_a_unsqueezed).squeeze(-1)
                    q_tot = chosen_q_vals.sum(dim=1, keepdim=True)

                    with torch.no_grad():
                        b_ns_flat = b_ns.view(BATCH_SIZE * NUM_AGENTS, STATE_DIM)
                        target_q_vals_flat = global_target_q(b_ns_flat)
                        target_q_vals = target_q_vals_flat.view(BATCH_SIZE, NUM_AGENTS, ACTION_DIM)
                        target_q_vals.masked_fill_(~b_nm, -1e9)
                        max_target_q_vals = target_q_vals.max(dim=2)[0]
                        masked_target_q = max_target_q_vals * (1.0 - b_nd)
                        target_q_tot = masked_target_q.sum(dim=1, keepdim=True)
                        team_reward = b_r.sum(dim=1, keepdim=True)
                        expected_q_tot = team_reward + GAMMA * target_q_tot

                    loss = nn.SmoothL1Loss()(q_tot, expected_q_tot)

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(global_q.parameters(), max_norm=0.5)
                    optimizer.step()
                    ep_losses.append(loss.item())

                    for target_param, local_param in zip(global_target_q.parameters(), global_q.parameters()):
                        target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        scheduler.step()

        success_rate = np.sum(env.dones) / NUM_AGENTS
        avg_loss = np.mean(ep_losses) if ep_losses else 0.0

        recent_success_rates.append(success_rate)
        current_avg_success = np.mean(recent_success_rates)

        if len(recent_success_rates) >= 20 and current_avg_success > best_avg_success:
            best_avg_success = current_avg_success
            torch.save(global_q.state_dict(), os.path.join(SAVE_DIR_BASE, 'best_global_q_network.pth'))

        current_ep_info = {
            'epoch': epoch, 'reward': episode_reward, 'success_rate': success_rate,
            'steps': step_count, 'trajectories': trajectories,
            'initial_targets': env.initial_targets.copy(),
            'final_targets': env.final_targets.copy(),
            'task_types': env.task_types.copy()
        }
        top_5_episodes.append(current_ep_info)
        top_5_episodes = sorted(top_5_episodes, key=lambda x: x['reward'], reverse=True)[:5]

        metrics_log.append({
            'Episode': epoch,
            'Total_Reward': episode_reward,
            'Success_Rate': success_rate,
            'Avg_Success_20': current_avg_success,
            'Collisions': ep_collisions,
            'Avoided_Collisions': ep_avoided_collisions,
            'Total_Steps': step_count,
            'Avg_Loss': avg_loss
        })

        arrival_status = ["✔️" if d else "❌" for d in env.dones]
        log_str = f"[轮次 {epoch:03d}] 成功率:{success_rate:.2f} | "
        if len(recent_success_rates) >= 20:
            log_str += f"20轮均线:{current_avg_success:.2f} | "
        log_str += f"LR:{scheduler.get_last_lr()[0]:.1e} | Eps:{epsilon:.2f} | 步数:{step_count} | 实际碰撞:{ep_collisions} | 仲裁避免:{ep_avoided_collisions}\n"
        log_str += f"   -> 状态: {arrival_status} | 任务(0空1入2出): {env.task_types.tolist()}"

        tqdm.write(log_str)
        pbar.set_postfix({'缩放后奖励': f'{episode_reward:.2f}'})

        if len(recent_success_rates) >= 20 and current_avg_success >= 0.95:
            tqdm.write(f"\n🎉 触发早停机制！提前结束训练。最优模型已保存。")
            break

    print("\n" + "=" * 50)
    print("📸 正在生成并保存训练过程中 Top 5 最佳轮次的 HTML 与 PNG 图像...")
    for rank, ep_info in enumerate(top_5_episodes, 1):
        base_name = f'train_top{rank}_ep{ep_info["epoch"]}_reward{ep_info["reward"]:.2f}'
        title = f'Top {rank} (轮次 {ep_info["epoch"]}) - 缩放后奖励: {ep_info["reward"]:.2f}'
        generate_plotly_3d(env, ep_info['trajectories'], base_name + '.html', title, ep_info['final_targets'], SAVE_DIR_TRAIN)
        save_static_3d_images(ep_info['trajectories'], ep_info['initial_targets'], ep_info['final_targets'], ep_info['task_types'], base_name, title, SAVE_DIR_TRAIN, env.lifter_pos)
        print(f"  -> [Rank {rank}] 已保存至 Train 文件夹: {base_name} (*.html / *.png)")

    csv_path = os.path.join(SAVE_DIR_TRAIN, 'training_log.csv')
    df_metrics = pd.DataFrame(metrics_log)
    df_metrics.to_csv(csv_path, index=False)
    plot_training_metrics(csv_path, SAVE_DIR_TRAIN)
    torch.save(global_q.state_dict(), os.path.join(SAVE_DIR_BASE, 'latest_global_q_network.pth'))
    print("=" * 50)


def evaluate(test_episodes=20):
    env = WarehouseEnv()
    global_q = VDN_Global_DQN().to(device)

    best_model_path = os.path.join(SAVE_DIR_BASE, 'best_global_q_network.pth')
    if os.path.exists(best_model_path):
        global_q.load_state_dict(torch.load(best_model_path))
        print(f"✅ 成功加载测试模型: {best_model_path}")
    else:
        print("❌ 找不到最佳模型，请先运行 train() 模式！")
        return

    global_q.eval()
    task_names = {0: "空载", 1: "入库", 2: "出库"}
    success_count = 0

    for ep in range(1, test_episodes + 1):
        states = env.reset()
        trajectories = {i: [list(env.agents_pos[i])] for i in range(NUM_AGENTS)}
        step_count = 0
        agent_steps = np.zeros(NUM_AGENTS, dtype=int)

        while not all(env.dones) and step_count < 150:
            actions = np.zeros(NUM_AGENTS, dtype=int)
            masks = env.get_action_masks()
            state_tensors = torch.FloatTensor(states).to(device)

            with torch.no_grad():
                q_vals_all = global_q(state_tensors)

            for i in range(NUM_AGENTS):
                if env.dones[i]:
                    actions[i] = 0
                else:
                    agent_steps[i] += 1
                    q_vals = q_vals_all[i].clone()
                    invalid_mask = torch.tensor(~masks[i], device=device)
                    q_vals.masked_fill_(invalid_mask, -1e9)
                    actions[i] = q_vals.argmax().item()

            states, _, _, _, _ = env.step(actions)

            for i in range(NUM_AGENTS):
                if not env.dones[i]:
                    trajectories[i].append(list(env.agents_pos[i]))
                elif len(trajectories[i]) > 0 and trajectories[i][-1] != list(env.final_targets[i]):
                    trajectories[i].append(list(env.final_targets[i]))
            step_count += 1

        success_rate = np.sum(env.dones) / NUM_AGENTS
        arrival_status = ["✔️" if d else "❌" for d in env.dones]
        print(f"\n[测试推演 {ep}/{test_episodes}] 成功率: {success_rate:.2f} | 整体耗时: {step_count}步")
        for i in range(NUM_AGENTS):
            task_str = task_names[env.task_types[i]]
            print(f"  -> 车{i} [{task_str}]: 耗时 {agent_steps[i]:03d}步 | 状态 {arrival_status[i]} | 终点: {trajectories[i][-1]}")

        if success_rate == 1.0:
            success_count += 1
            base_name = f'test_success_{success_count}_ep{ep}'
            title = f'测试成功案例 {success_count} (总测试次 {ep})'
            generate_plotly_3d(env, trajectories, base_name + '.html', title, env.final_targets, SAVE_DIR_TEST)
            save_static_3d_images(trajectories, env.initial_targets, env.final_targets, env.task_types, base_name, title, SAVE_DIR_TEST, env.lifter_pos)
            print(f"  📸 [全到达] 已将 HTML 及路线图保存至 Test 文件夹: {base_name}")


if __name__ == "__main__":
    # 因为环境逻辑已修复，如果之前有过旧模型的缓存建议重新训练一遍
    RUN_MODE = 'train'
    if RUN_MODE == 'train':
        train()
    elif RUN_MODE == 'test':
        evaluate(test_episodes=20)