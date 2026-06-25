import numpy as np
from typing import Tuple, Dict, Optional
from .dynamics import LiftCruiseEVTOLDynamics as EVTOLDynamics
from .obstacles import ObstacleGenerator, TrainingStage
from .wind import WindField, WindIntensity, get_wind_intensity_for_stage


class EVTOLEnvironment:
    '''基于Lift+Cruise复合翼的eVTOL环境'''

    def __init__(self, stage: TrainingStage = TrainingStage.BASIC_FLIGHT, config: dict = None):
        self.stage = stage
        self.config = config or {}
        self.setup_stage_parameters()

        # 初始化组件，传递配置
        self.dynamics = EVTOLDynamics(config=config)
        self.obstacle_gen = ObstacleGenerator(self.area_size, stage, wingspan=self.dynamics.wingspan)

        # 根据训练阶段设置风场强度
        if stage.value >= 2:  # 从SIMPLE_NAV阶段开始有风场
            wind_intensity = get_wind_intensity_for_stage(stage.value)
            self.wind_field = WindField(intensity=wind_intensity, config=config)
            print(f"Wind field initialized with {wind_intensity.name} intensity for stage {stage.name}")
        else:
            self.wind_field = None

        # 状态和动作维度
        self.state_dim = 15  # 状态空间维度
        self.action_dim = 4  # 动作空间维度

        # 回合变量
        self.current_step = 0
        self.max_steps = 1500  # 与配置文件同步
        self.target_position = None
        self.start_position = None
        self.current_obstacle_stats = {}
        self.current_W_V = 0.0

        # 奖励权重
        self.reward_weights = {
            'task_completion': 500.0,
            'distance_progress': 5.0,
            'time_penalty': -0.5,

            # 物理约束惩罚
            'collision': -5.0,
            'safety_margin': -0.1,
            'overspeed':-0.5,
            'underspeed': -0.5,
            'altitude_exceeded': -2.0,
            'ground_collision': -5.0,
            'range_exceeded': -5.0,

            # 能源和性能惩罚
            'lift_power': -0.00001,
            'cruise_power': -0.00001,
            'accel_incentive': 0.01,  # 过渡加速激励权重
            'energy_efficiency': -0.0001,
            'transition_mode_penalty': -0.001,

            # 飞行舒适性
            'comfort': -0.01,
            'smoothness': -0.01,
        }

        # 从配置读取目标半径
        self.target_radius = self.config.get('environment', {}).get('target_radius', 15.0)

    def setup_stage_parameters(self):
        '''根据训练阶段设置区域参数'''
        params = {
            TrainingStage.BASIC_FLIGHT: {
                'area_size': (1500, 1500),
                'max_start_distance': 300.0
            },
            TrainingStage.SIMPLE_NAV: {
                'area_size': (2000, 2000),
                'max_start_distance': 750.0
            },
            TrainingStage.COMPLEX_NAV: {
                'area_size': (7000, 7000),
                'max_start_distance': 3000.0
            },
            TrainingStage.GENERALIZATION: {
                'area_size': (25000, 25000),
                'max_start_distance': 15000.0
            }
        }

        stage_params = params[self.stage]
        self.area_size = stage_params['area_size']
        self.max_start_distance = stage_params['max_start_distance']

    def _transition_weight(self) -> float:
        """计算混合因子 W(V)"""
        V = self.dynamics.velocity_magnitude
        v_start = self.config.get('environment', {}).get('transition_start_speed', 20.0)
        v_end = self.config.get('environment', {}).get('transition_end_speed', 44.0)

        if V <= v_start:
            return 0.0
        elif V >= v_end:
            return 1.0
        else:
            # 使用余弦函数确保连续和平滑过渡
            phase = np.pi * (V - v_start) / (v_end - v_start)
            return 0.5 * (1 - np.cos(phase))

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        '''重置环境'''
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0

        # 生成起始和目标位置
        if self.stage == TrainingStage.BASIC_FLIGHT:
            self.start_position = np.array([750.0, 750.0, 0.0], dtype=np.float32)
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(50, self.max_start_distance)
            self.target_position = self.start_position + np.array([
                distance * np.cos(angle),
                distance * np.sin(angle),
                np.random.uniform(50, 300)
            ], dtype=np.float32)
        else:
            # 多次尝试生成安全位置
            max_attempts = 50
            for attempt in range(max_attempts):
                if attempt == 0:
                    self.obstacle_gen.generate(seed)

                safety_margin = self.config.get('environment', {}).get('safety_distance_threshold', 25.0)

                self.start_position = np.array([
                    np.random.uniform(100, self.area_size[0] - 100),
                    np.random.uniform(100, self.area_size[1] - 100),
                    0.0
                ], dtype=np.float32)

                self.target_position = np.array([
                    np.random.uniform(100, self.area_size[0] - 100),
                    np.random.uniform(100, self.area_size[1] - 100),
                    np.random.uniform(50, 500)
                ], dtype=np.float32)

                # 检查位置安全性
                start_safe = self.obstacle_gen.get_nearest_obstacle(self.start_position)[0] > safety_margin
                target_safe = self.obstacle_gen.get_nearest_obstacle(self.target_position)[0] > safety_margin

                if start_safe and target_safe:
                    break

                if attempt == max_attempts - 1:
                    # 使用固定安全位置
                    self.start_position = np.array([
                        self.area_size[0] * 0.2, self.area_size[1] * 0.2, 0.0
                    ], dtype=np.float32)
                    self.target_position = np.array([
                        self.area_size[0] * 0.8, self.area_size[1] * 0.8, 300.0
                    ], dtype=np.float32)

        # 重置动力学模型
        self.dynamics.reset(self.start_position)

        # 记录障碍物统计
        self.current_obstacle_stats = self.obstacle_gen.get_obstacle_statistics()

        return self.get_state()

    def get_state(self) -> np.ndarray:
        '''构建15维状态向量 '''
        state = []

        # 相对位置与距离 R (3D + 1D)
        relative_position = self.target_position - self.dynamics.position
        state.extend(relative_position)
        state.append(np.linalg.norm(relative_position))

        # 运动学状态 (4D)
        state.extend([
            self.dynamics.velocity_magnitude,  # V 速度大小
            self.dynamics.gamma,  # γ 飞行路径角
            self.dynamics.chi,  # χ 航向角
            self.current_W_V   # W(V) 混合因子
        ])

        # 控制姿态 (2D)
        state.extend([
            self.dynamics.alpha,  # α 攻角
            self.dynamics.phi  # φ 滚转角
        ])

        # 风场 (2D)
        if self.wind_field:
            wind_velocity = self.wind_field.get_wind(self.dynamics.position, self.dynamics.dt)
            state.extend(wind_velocity[:2])  # 水平风速分量
        else:
            state.extend([0.0, 0.0])

        # 障碍物感知 (3D)
        min_obstacle_dist, obstacle_bearing = self.obstacle_gen.get_nearest_obstacle(
            self.dynamics.position
        )
        state.append(min_obstacle_dist)
        state.extend(obstacle_bearing)

        return np.array(state, dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        '''执行动作 action: [T_Lift, T_Cruise, α, φ]'''
        self.current_step += 1

        # 获取风场
        wind_velocity = None
        if self.wind_field:
            wind_velocity = self.wind_field.get_wind(self.dynamics.position, self.dynamics.dt)

        # 更新动力学
        dynamics_info = self.dynamics.step(action, wind_velocity)

        # 存储当前的混合因子 W(V)
        self.current_W_V = dynamics_info['aerodynamics']['transition_weight']

        # 通道宽度检查
        min_obstacle_dist, _ = self.obstacle_gen.get_nearest_obstacle(self.dynamics.position)
        required_clearance = self.dynamics.wingspan

        # 如果飞机试图通过小于翼展的地方，记录为违规
        if min_obstacle_dist < required_clearance:
            dynamics_info['violations']['narrow_passage'] = required_clearance - min_obstacle_dist

        # 计算奖励
        reward = self.calculate_reward(dynamics_info)

        # 检查终止条件
        terminated = self.check_termination(dynamics_info)
        truncated = self.current_step >= self.max_steps

        # 获取下一个状态
        next_state = self.get_state()

        info = {
            'position': self.dynamics.position.copy(),
            'distance_to_goal': np.linalg.norm(self.target_position - self.dynamics.position),
            'violations': dynamics_info['violations']
        }

        return next_state, reward, terminated, truncated, info

    def calculate_reward(self, dynamics_info: Dict) -> float:
        '''计算复合奖励函数'''
        reward = 0.0
        violations = dynamics_info['violations']

        # 任务进度与完成
        distance_to_goal = np.linalg.norm(self.target_position - self.dynamics.position)
        if distance_to_goal < self.target_radius:
            reward += self.reward_weights['task_completion']

        # 距离进度奖励
        prev_distance = np.linalg.norm(
            self.target_position - (self.dynamics.position - self.dynamics.velocity * self.dynamics.dt)
        )
        reward += (prev_distance - distance_to_goal) * self.reward_weights['distance_progress']

        # 障碍物安全奖励
        min_obstacle_dist, _ = self.obstacle_gen.get_nearest_obstacle(self.dynamics.position)

        # 定义安全距离阈值
        d_safe = self.config.get('environment', {}).get('collision_distance_threshold', 10.0)
        d_influence = self.config.get('environment', {}).get('safety_distance_threshold', 25.0)

        if min_obstacle_dist < d_safe:
            reward += self.reward_weights['collision']
        elif d_safe <= min_obstacle_dist <= d_influence:
            danger_reward = -abs(self.reward_weights['safety_margin']) * np.exp(
                -(min_obstacle_dist - d_safe) / d_influence)
            reward += danger_reward

        # 物理约束惩罚
        for violation_type, violation_value in violations.items():
            if violation_type in self.reward_weights:
                reward += violation_value * self.reward_weights[violation_type]

        # 能源效率惩罚
        W_V = dynamics_info['aerodynamics']['transition_weight']

        # 垂直起降阶段的推力惩罚
        m_g = self.dynamics.mass * self.dynamics.g

        # 净推力惩罚项：只有当 T_Lift 超过 mg 时，才惩罚多余部分
        T_Lift_excess = np.maximum(0, self.dynamics.thrust_lift - m_g)

        reward += (
                T_Lift_excess * self.reward_weights['lift_power'] * (1 - W_V)
        )

        # 巡航阶段的推力惩罚
        reward += (
                self.dynamics.thrust_cruise * self.reward_weights['cruise_power']
        )

        # 飞行平滑性
        reward += np.linalg.norm(dynamics_info['acceleration']) * self.reward_weights['comfort']
        reward += np.linalg.norm(dynamics_info['jerk']) * self.reward_weights['smoothness']

        # 时间惩罚
        reward += self.reward_weights['time_penalty']

        # 过渡模式激励
        V = self.dynamics.velocity_magnitude
        v_start = self.config.get('environment', {}).get('transition_start_speed', 20.0)
        v_end = self.config.get('environment', {}).get('transition_end_speed', 44.0)

        if v_start <= V <= v_end:
            reward += V * self.reward_weights['accel_incentive'] * (1 - W_V) # 鼓励快速通过过渡区域

        return reward

    def check_termination(self, dynamics_info: Dict) -> bool:
        '''检查终止条件'''
        violations = dynamics_info['violations']

        # 任务完成
        if np.linalg.norm(self.target_position - self.dynamics.position) < self.target_radius:
            return True

        # 严重违规
        critical_violations = [
            'ground_collision',
            'altitude_exceeded',
            'range_exceeded',
            'narrow_passage'
        ]
        for violation in critical_violations:
            if violation in violations:
                return True

        # 碰撞检测
        min_obstacle_dist, _ = self.obstacle_gen.get_nearest_obstacle(self.dynamics.position)
        d_safe = self.config.get('environment', {}).get('collision_distance_threshold', 10.0)
        if min_obstacle_dist < d_safe:
            return True

        # 边界检测
        if (self.dynamics.position[0] < 0 or
                self.dynamics.position[0] > self.area_size[0] or
                self.dynamics.position[1] < 0 or
                self.dynamics.position[1] > self.area_size[1]):
            return True

        return False


# 为了后向兼容
if __name__ == "__main__":
    from obstacles import TrainingStage
    import numpy as np

    # 快速测试环境初始化和基本功能
    print("=== eVTOL Lift+Cruise 环境测试 ===")

    # 使用不同训练阶段测试环境初始化
    for stage in [TrainingStage.BASIC_FLIGHT, TrainingStage.SIMPLE_NAV, TrainingStage.COMPLEX_NAV]:
        print(f"\n测试阶段: {stage.name}")
        env = EVTOLEnvironment(stage=stage)

        # 重置环境
        initial_state = env.reset()
        print(f"初始状态维度: {initial_state.shape}")
        print(f"初始状态: {initial_state}")

        # 执行几个随机动作
        for _ in range(5):
            # 生成随机动作：[T_Lift, T_Cruise, α, φ]
            action = np.random.uniform(-1, 1, 4)
            next_state, reward, terminated, truncated, info = env.step(action)

            print(f"动作: {action}")
            print(f"奖励: {reward}")
            print(f"终止: {terminated}, 截断: {truncated}")

            if terminated or truncated:
                break