import numpy as np
from typing import List, Dict, Tuple, Optional
from enum import Enum  ##定义枚举类型


class TrainingStage(Enum):
    BASIC_FLIGHT = 1
    SIMPLE_NAV = 2
    COMPLEX_NAV = 3
    GENERALIZATION = 4


class ObstacleGenerator:
    '''障碍物生成器 '''

    def __init__(self, area_size: Tuple[float, float], stage: TrainingStage,
                 wingspan: float = 14.5, config: dict = None):
        if area_size[0] <= 0 or area_size[1] <= 0:
            raise ValueError("Area size must be positive")
        if wingspan <= 0:
            raise ValueError("Wingspan must be positive")

        self.area_size = area_size
        self.stage = stage
        self.wingspan = wingspan  # GUAM翼展14.5m
        self.obstacles = []
        self.config = config or {}
        self.setup_stage_parameters()
        self.danger_zones = []

    def setup_stage_parameters(self):
        """设置障碍物生成参数 """
        # 从配置文件获取障碍物参数，如果没有则使用默认值
        obstacles_config = self.config.get('obstacles', {})

        # 获取密度配置
        densities = obstacles_config.get('densities', {
            'BASIC_FLIGHT': 0,
            'SIMPLE_NAV': 0.00002,
            'COMPLEX_NAV': 0.00006,
            'GENERALIZATION': 0.0001
        })

        # 获取高度参数配置
        height_params = obstacles_config.get('height_params', {
            'BASIC_FLIGHT': {'alpha': 2.0, 'min': 20, 'max': 50},
            'SIMPLE_NAV': {'alpha': 2.0, 'min': 20, 'max': 80},
            'COMPLEX_NAV': {'alpha': 2.0, 'min': 20, 'max': 150},
            'GENERALIZATION': {'alpha': 2.5, 'min': 30, 'max': 300}
        })

        # 获取半径参数配置
        radius_params = obstacles_config.get('radius_params', {
            'BASIC_FLIGHT': {'alpha': 2.0, 'min': 10, 'max': 30},
            'SIMPLE_NAV': {'alpha': 2.0, 'min': 10, 'max': 50},
            'COMPLEX_NAV': {'alpha': 2.0, 'min': 10, 'max': 100},
            'GENERALIZATION': {'alpha': 2.5, 'min': 20, 'max': 200}
        })

        # 获取安全间距配置
        spacing_multipliers = obstacles_config.get('safety_spacing_multipliers', {
            'BASIC_FLIGHT': 0,
            'SIMPLE_NAV': 2.0,
            'COMPLEX_NAV': 1.5,
            'GENERALIZATION': 1.0
        })

        #危险区域比例配置（允许多少比例的间隙小于最低标准）
        danger_zone_ratios = obstacles_config.get('danger_zone_ratios', {
            'BASIC_FLIGHT': 0.0,  # 无危险区域
            'SIMPLE_NAV': 0.0,  # 无危险区域
            'COMPLEX_NAV': 0.15,  # 15%的间隙可能是危险的
            'GENERALIZATION': 0.25  # 25%的间隙可能是危险的
        })

        # 最小物理间距（绝对不能小于此值，否则会重叠）
        min_physical_gaps = obstacles_config.get('min_physical_gaps', {
            'BASIC_FLIGHT': 0,
            'SIMPLE_NAV': 0.7,  # 0.7×翼展（10.15m）
            'COMPLEX_NAV': 0.5,  # 0.5×翼展（7.25m）
            'GENERALIZATION': 0.4  # 0.4×翼展（5.8m）
        })

        # 其他配置参数
        self.generation_margin = obstacles_config.get('generation_margin', 50.0)
        self.max_placement_attempts = obstacles_config.get('max_placement_attempts', 20)

        # 基于训练阶段动态设置最小间距
        stage_name = self.stage.name
        base_spacing = self.wingspan  # 基础间距：翼展，保证障碍物不会重叠

        multiplier = spacing_multipliers.get(stage_name, 2.0)
        min_safe_spacing = base_spacing * multiplier   # 推荐的安全间距

        danger_ratio = danger_zone_ratios.get(stage_name, 0.0)
        min_physical_multiplier = min_physical_gaps.get(stage_name, 0.5)
        min_physical_spacing = base_spacing * min_physical_multiplier  # 绝对最小间距

        # 获取当前阶段的参数
        density = densities.get(stage_name, 0)
        height_config = height_params.get(stage_name, {'alpha': 2.0, 'min': 20, 'max': 50})
        radius_config = radius_params.get(stage_name, {'alpha': 2.0, 'min': 10, 'max': 30})

        self.params = {
            'density': density,
            'height_alpha': height_config['alpha'],
            'height_min': height_config['min'],
            'height_max': height_config['max'],
            'radius_alpha': radius_config['alpha'],
            'radius_min': radius_config['min'],
            'radius_max': radius_config['max'],
             'min_gap_width': min_safe_spacing,           # 推荐的安全间距
            'min_physical_gap': min_physical_spacing,     # 绝对最小间距
            'danger_zone_ratio': danger_ratio,            # 危险区域比例
        }

    def generate_power_law_sample(self, alpha: float, min_val: float, max_val: float) -> float:
        """生成幂律分布样本"""
        if min_val <= 0 or max_val <= min_val or alpha <= 0:
            raise ValueError("Invalid parameters for power law distribution")

        u = np.random.uniform(0, 1)

        if abs(alpha - 1.0) < 1e-10:  # alpha ≈ 1 的情况
            result = min_val * (max_val / min_val) ** u
        else:
            # 使用逆变换采样
            ratio = max_val / min_val
            result = min_val * (1 + u * (ratio ** (1 - alpha) - 1)) ** (1 / (1 - alpha))

        return float(np.clip(result, min_val, max_val))

    def check_gap_width(self, new_obs: Dict) -> bool:
        """检查新障碍物是否满足约束：允许部分间隙小于推荐安全间距，但不能小于物理最小间距"""
        required_keys = ['position', 'radius']
        if not all(key in new_obs for key in required_keys):  # 确认障碍物字典有 position 和 radius 两个关键字段
            return False

        danger_ratio = self.params['danger_zone_ratio']

        for existing_obs in self.obstacles:
            center_distance = np.linalg.norm(
                new_obs['position'][:2] - existing_obs['position'][:2]  # 只取前两个元素x，y，忽略高度
            )
            gap_width = center_distance - new_obs['radius'] - existing_obs['radius']

            # 绝对不能小于物理最小间距（会重叠）
            if gap_width < self.params['min_physical_gap']:
                return False

            # 如果间隙小于推荐安全间距，按概率决定是否允许
            if gap_width < self.params['min_gap_width']:
                if np.random.random() > danger_ratio:
                    return False

        return True

    def generate(self, seed: Optional[int] = None) -> List[Dict]:
        '''生成障碍物'''
        if seed is not None:
            np.random.seed(seed)  # 固定随机种子

        self.obstacles = []

        if self.params['density'] == 0:
            return self.obstacles  # 如果该阶段密度为 0（比如基础飞行），直接返回空

        # 计算障碍物数量
        area = self.area_size[0] * self.area_size[1]  # 计算总面积
        lambda_param = self.params['density'] * area  # 计算泊松分布的λ参数（期望障碍物数量）
        num_obstacles = np.random.poisson(lambda_param)  # 从泊松分布中随机生成障碍物数量

        if num_obstacles == 0:  # 如果没有障碍物要放置，直接返回空列表
            return self.obstacles

        successful_placements = 0  # 成功放置的计数器
        max_total_attempts = num_obstacles * self.max_placement_attempts  # 总尝试次数上限

        for obstacle_idx in range(num_obstacles):  # 单个障碍物放置循环
            placed = False  # 标记是否成功放置
            attempts = 0  # 尝试次数计数器
            max_attempts_per_obstacle = max(1, max_total_attempts // num_obstacles)

            while not placed and attempts < max_attempts_per_obstacle:
                attempts += 1

                # 生成位置，确保不超出边界
                margin = max(self.params['radius_max'], self.generation_margin)
                if self.area_size[0] <= 2 * margin or self.area_size[1] <= 2 * margin:
                    break  # 区域太小，无法放置障碍物

                position = np.array([
                    np.random.uniform(margin, self.area_size[0] - margin),
                    np.random.uniform(margin, self.area_size[1] - margin),
                    0
                ])

                # 幂律函数生成半径和高度
                radius = self.generate_power_law_sample(
                    self.params['radius_alpha'],
                    self.params['radius_min'],
                    self.params['radius_max']
                )

                height = self.generate_power_law_sample(
                    self.params['height_alpha'],
                    self.params['height_min'],
                    self.params['height_max']
                )

                temp_obstacle = {
                    'position': position,
                    'radius': radius,
                    'height': height
                }

                # 额外检查：障碍物不应超出区域边界
                if (position[0] - radius < 0 or position[0] + radius > self.area_size[0] or
                        position[1] - radius < 0 or position[1] + radius > self.area_size[1]):
                    continue  # 放弃这次尝试，继续循环

                if self.check_gap_width(temp_obstacle):
                    self.obstacles.append(temp_obstacle)
                    successful_placements += 1
                    placed = True

        return self.obstacles


    def get_obstacle_statistics(self) -> Dict:
        """获取障碍物统计信息"""
        if not self.obstacles:
            return {
                'count': 0,
                'avg_radius': 0,
                'avg_height': 0,
                'max_radius': 0,
                'max_height': 0,
                'min_radius': 0,
                'min_height': 0,
                'density_actual': 0,
                'density_expected': self.params['density'],
                'min_gap_width': None,  # 没有障碍物时为None更合适
                'wingspan_safety_ratio': None,
                'safe_for_guam': None,
                'placement_efficiency': 1.0,  # 没有障碍物时视为100%效率
                'config_used': self.config.get('obstacles', {})
            }

        radii = [obs['radius'] for obs in self.obstacles]  # 所有半径列表
        heights = [obs['height'] for obs in self.obstacles]  # 所有高度列表
        area = self.area_size[0] * self.area_size[1]

        # 计算两障碍物的最小间隙宽度
        min_gap = float('inf')
        if len(self.obstacles) >= 2:
            for i in range(len(self.obstacles)):
                for j in range(i + 1, len(self.obstacles)):
                    obs1, obs2 = self.obstacles[i], self.obstacles[j]
                    center_dist = np.linalg.norm(obs1['position'][:2] - obs2['position'][:2])
                    gap = center_dist - obs1['radius'] - obs2['radius']
                    if gap >= 0:  # 只考虑不重叠的情况
                        min_gap = min(min_gap, gap)

        # 处理边界情况
        if min_gap == float('inf'):
            min_gap = None
            wingspan_safety_ratio = None
            safe_for_guam = None
        else:
            wingspan_safety_ratio = min_gap / self.wingspan
            safe_for_guam = wingspan_safety_ratio >= 1.5  # 布尔标准

        # 计算放置效率
        expected_count = self.params['density'] * area
        placement_efficiency = len(self.obstacles) / max(expected_count, 1)

        return {
            'count': len(self.obstacles),
            'avg_radius': np.mean(radii),
            'avg_height': np.mean(heights),
            'max_radius': np.max(radii),
            'max_height': np.max(heights),
            'min_radius': np.min(radii),
            'min_height': np.min(heights),
            'density_actual': len(self.obstacles) / area,
            'density_expected': self.params['density'],
            'min_gap_width': min_gap,
            'wingspan_safety_ratio': wingspan_safety_ratio,
            'safe_for_guam': safe_for_guam,
            'placement_efficiency': placement_efficiency,
            'stage_name': self.stage.name,
            'area_size': self.area_size,
            'generation_params': self.params.copy(),
            'config_used': self.config.get('obstacles', {})
        }

    def get_nearest_obstacle(self, position: np.ndarray) -> Tuple[float, np.ndarray]:
        '''找到最近障碍物的距离和方向 '''

        # --- 引入安全常量 ---
        SAFE_FAR_DISTANCE = 100000.0

        if not self.obstacles:
            return SAFE_FAR_DISTANCE, np.zeros(2)

        min_distance = float('inf')  # 到最近障碍物表面的实际3D距离
        nearest_bearing = np.zeros(2)  # 指向最近障碍物中心的单位方向向量（2D）

        for obs in self.obstacles:
            # 计算到障碍物边缘的水平距离
            horizontal_dist_to_edge = (
                    np.linalg.norm(position[:2] - obs['position'][:2]) - obs['radius']
            )

            # 3D距离计算
            if position[2] <= obs['height']:
                # 飞机在障碍物高度以下或同等高度
                distance = max(0, horizontal_dist_to_edge)
            else:
                # 飞机在障碍物上方
                vertical_clearance = position[2] - obs['height']
                if horizontal_dist_to_edge <= 0:
                    # 水平投影在障碍物内部
                    distance = vertical_clearance
                else:
                    # 水平投影在障碍物外部
                    distance = np.sqrt(horizontal_dist_to_edge ** 2 + vertical_clearance ** 2)

            if distance < min_distance:
                min_distance = distance
                diff = obs['position'][:2] - position[:2]
                norm_diff = np.linalg.norm(diff)
                if norm_diff > 1e-10:
                    nearest_bearing = diff / norm_diff

        if min_distance == float('inf'):
            min_distance = SAFE_FAR_DISTANCE

        return min_distance, nearest_bearing
