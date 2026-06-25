import numpy as np
import random
import matplotlib.pyplot as plt

class ImprovedPSO_ACO:
    def __init__(self, num_drones, targets, max_iter=100, swarm_size=50):
        self.num_drones = num_drones
        self.targets = targets  # 目标点列表，包含原点(0,0)
        self.max_iter = max_iter
        self.swarm_size = swarm_size
        self.num_targets = len(targets) - 1  # 目标点总数（排除原点）
        self.w = 0.7  # 惯性权重
        self.c1 = 1.5  # 个体学习因子
        self.c2 = 1.5  # 群体学习因子
        self.pheromone_weight = 2.0  # 信息素权重
        self.pheromone_decay = 0.1  # 信息素挥发率
        self.pheromone = np.ones((len(targets), len(targets)))  # 路径信息素矩阵
        self.max_speed = 50  # 最大速度 (m/s)
        self.min_distance = 50  # 最小间距 (m)
        self.max_distance = 1000  # 最大间距 (m)
        self.initialize_particles()

    def initialize_particles(self):
        """初始化粒子群：每个粒子为3架无人机的路径分配"""
        self.particles = []
        for _ in range(self.swarm_size):
            paths = [[] for _ in range(self.num_drones)]
            unassigned = list(range(1, len(self.targets)))  # 目标点索引（排除原点）
            random.shuffle(unassigned)
            for i in range(self.num_drones):
                paths[i].append(0)  # 起点为原点
                if unassigned:
                    paths[i].extend(unassigned[:2])  # 每架无人机最多分配2个目标点
                    unassigned = unassigned[2:]
            self.particles.append(paths)
        self.pbest = self.particles.copy()
        self.pbest_fitness = [self.fitness(p) for p in self.particles]
        self.gbest = min(self.particles, key=lambda x: self.fitness(x))
        self.gbest_fitness = self.fitness(self.gbest)

    def distance(self, p1, p2):
        """计算两点间欧氏距离"""
        return np.linalg.norm(np.array(self.targets[p1]) - np.array(self.targets[p2]))

    def fitness(self, particle):
        """计算适应度：总距离 + 未覆盖惩罚 + 间距约束惩罚 + 速度约束惩罚"""
        total_distance = 0
        total_time=0
        covered = set()
        time_matrix = [[0.0 for _ in range(len(path))] for path in particle]
        spacing_penalty = 0

        # 计算每架无人机的路径时间和间距惩罚
        for i, path in enumerate(particle):
            for j in range(1, len(path)):
                dist = self.distance(path[j-1], path[j])
                time_matrix[i][j] = dist / self.max_speed  # 计算每段路径的时间
                total_distance += dist
                total_time += dist

            covered.update(path[1:])  # 记录覆盖的目标点

        # 计算无人机间距惩罚
        for i in range(self.num_drones):
            for j in range(i + 1, self.num_drones):
                if len(particle[i]) > 1 and len(particle[j]) > 1:
                    # 计算两架无人机在各段路径的时间点
                    time_points_i = np.cumsum(time_matrix[i][1:])
                    time_points_j = np.cumsum(time_matrix[j][1:])

                    # 计算两架无人机在各时间点的间距
                    for t in range(max(len(time_points_i), len(time_points_j))):
                        pos_i = self.targets[particle[i][min(t, len(particle[i])-1)]]
                        pos_j = self.targets[particle[j][min(t, len(particle[j])-1)]]
                        spacing = self.distance(particle[i][min(t, len(particle[i])-1)], particle[j][min(t, len(particle[j])-1)])

                        if spacing < self.min_distance or spacing > self.max_distance:
                            spacing_penalty += 1

        # 未覆盖目标点的惩罚
        penalty = 1e4 * (self.num_targets - len(covered))
        # 间距约束惩罚
        spacing_penalty *= 1e2

        return total_distance + penalty + spacing_penalty

    def update_pheromone(self, paths):
        """更新信息素：路径被选中时增加信息素"""
        for path in paths:
            for i in range(1, len(path)):
                self.pheromone[path[i-1]][path[i]] += 1
        self.pheromone *= (1 - self.pheromone_decay)  # 信息素挥发

    def select_path(self, current_path, unvisited):
        """ACO筛选路径：根据信息素和距离选择下一个目标点"""
        probabilities = []
        for t in unvisited:
            pheromone = self.pheromone[current_path[-1]][t]
            dist = self.distance(current_path[-1], t)
            probabilities.append((pheromone ** self.pheromone_weight) / (dist + 1e-6))
        prob_sum = sum(probabilities)
        if prob_sum == 0:
            return random.choice(unvisited)
        probabilities = [p / prob_sum for p in probabilities]
        return random.choices(unvisited, weights=probabilities, k=1)[0]

    def optimize(self):
        """主优化循环"""
        for _ in range(self.max_iter):
            for idx in range(self.swarm_size):
                new_particle = []
                for drone_path in self.particles[idx]:
                    new_path = [0]
                    unvisited = [t for t in range(1, len(self.targets)) if t not in new_path]
                    while unvisited:
                        next_t = self.select_path(new_path, unvisited)
                        new_path.append(next_t)
                        unvisited.remove(next_t)
                    new_particle.append(new_path)
                # 更新信息素
                self.update_pheromone(new_particle)
                # 更新个体和全局最优
                current_fitness = self.fitness(new_particle)
                if current_fitness < self.pbest_fitness[idx]:
                    self.pbest[idx] = new_particle
                    self.pbest_fitness[idx] = current_fitness
                if current_fitness < self.gbest_fitness:
                    self.gbest = new_particle
                    self.gbest_fitness = current_fitness
        return self.gbest, self.gbest_fitness

# 目标点坐标（原点为(0,0)，索引0）
targets = [
    (0, 0),      # 原点
    (1200, 800), # T1
    (300, 450),  # T2
    (950, 200),  # T3
    (600, 1200), # T4
    (1500, 500)  # T5
]

# 运行优化
pso_aco = ImprovedPSO_ACO(num_drones=3, targets=targets, max_iter=50)
best_paths, best_distance = pso_aco.optimize()

# 输出结果
print("无人机1路径：", [targets[i] for i in best_paths[0]], "总距离：", sum(pso_aco.distance(best_paths[0][i-1], best_paths[0][i]) for i in range(1, len(best_paths[0]))))
print("无人机2路径：", [targets[i] for i in best_paths[1]], "总距离：", sum(pso_aco.distance(best_paths[1][i-1], best_paths[1][i]) for i in range(1, len(best_paths[1]))))
print("无人机3路径：", [targets[i] for i in best_paths[2]], "总距离：", sum(pso_aco.distance(best_paths[2][i-1], best_paths[2][i]) for i in range(1, len(best_paths[2]))))
print("总飞行距离：", best_distance)

# 绘制路径和目标点覆盖情况
def plot_paths(best_paths, targets):
    plt.figure(figsize=(10, 8))
    colors = ['r', 'g', 'b']  # 分别代表三架无人机的颜色

    # 绘制每架无人机的路径
    for i, path in enumerate(best_paths):
        x = [targets[node][0] for node in path]
        y = [targets[node][1] for node in path]
        plt.plot(x, y, f'{colors[i]}-o', label=f'Drone {i+1} Path')
        plt.scatter(x, y, color=colors[i])

    # 绘制目标点
    target_x = [t[0] for t in targets[1:]]  # 排除原点
    target_y = [t[1] for t in targets[1:]]
    plt.scatter(target_x, target_y, color='blue', marker='x', s=100, label='Targets')

    # 绘制原点
    plt.scatter([0], [0], color='black', marker='s', s=100, label='Start Point')

    plt.title('Multi-drone Path Planning')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.legend()
    plt.grid()
    plt.show()

# 调用绘图函数
plot_paths(best_paths, targets)