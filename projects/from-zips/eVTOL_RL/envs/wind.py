import numpy as np
from enum import Enum
from typing import  Dict

class WindIntensity(Enum):
    """风场强度等级"""
    CALM = 0  # 无风/微风
    NORMAL = 1  # 正常风况
    MODERATE = 2  # 中等风况
    EXTREME = 3  # 极端风况


class UrbanWindField:
    """城市风场模型 """

    def __init__(self, intensity: WindIntensity = WindIntensity.NORMAL, config: dict = None):
        self.intensity = intensity
        self.time = 0.0
        self.config = config or {}

        # 获取风场配置
        self.wind_config = self.config.get('wind', {})

        # 根据风场强度设置参数
        self.setup_wind_parameters()

    def setup_wind_parameters(self):
        """根据风场强度设置参数 """
        # 基础参数（可被配置覆盖）
        self.z0 = self.wind_config.get('reference_height', 140.0)  # 参考高度 (m)
        default_alpha = self.wind_config.get('wind_profile_exponent', 0.29)  # 默认风剖面指数

        # 获取强度配置
        intensity_params = self.wind_config.get('intensity_params', {})
        stage_name = self.intensity.name

        # 默认参数，中间优先级（default文件中的是最高优先级）
        default_params = {
            'CALM': {
                'base_speed': 3.0,
                'wind_profile_exponent': 0.29,
                'gust_start_time': 100.0,
                'gust_duration': 0.5,
                'random_max_amplitude': 0.2,
                'random_frequency': 0.1
            },
            'NORMAL': {
                'base_speed': 6.0,
                'wind_profile_exponent': 0.29,
                'gust_start_time': 1000.0,
                'gust_duration': 0.1,
                'random_max_amplitude': 0.0,
                'random_frequency': 0.0
            },
            'MODERATE': {
                'base_speed': 10.0,
                'wind_profile_exponent': 0.29,
                'gust_start_time': 50.0,
                'gust_duration': 1.0,
                'random_max_amplitude': 0.8,
                'random_frequency': 0.3
            },
            'EXTREME': {
                'base_speed': 25.0,
                'wind_profile_exponent': 0.35,
                'gust_start_time': 2.0,
                'gust_duration': 5.0,
                'random_max_amplitude': 3.0,
                'random_frequency': 1.0
            }
        }

        # 获取当前强度的参数
        stage_params = intensity_params.get(stage_name, default_params.get(stage_name, default_params['NORMAL']))

        # 创建实例变量（配置优先，最低优先级）
        self.v0 = float(stage_params['base_speed'])
        self.alpha = float(stage_params.get('wind_profile_exponent', default_alpha))
        self.gust_start_time = float(stage_params['gust_start_time'])
        self.gust_duration = float(stage_params['gust_duration'])
        self.random_max_amplitude = float(stage_params['random_max_amplitude'])
        self.random_frequency = float(stage_params['random_frequency'])

        # 风向配置
        self.primary_direction = np.array(
            self.wind_config.get('primary_direction', [1.0, 0.0, 0.0]),  # 主风向
            dtype=np.float32
        )
        self.primary_direction = self.primary_direction / np.linalg.norm(self.primary_direction)        # 归一化
        self.direction_variation = float(self.wind_config.get('direction_variation', 0.1))              # 风向变化幅度
        self.direction_frequency = float(self.wind_config.get('direction_frequency', 0.3))              # 风向变化频率
        self.vertical_component_factor = float(self.wind_config.get('vertical_component_factor', 0.2))  # 垂直分量系数

        # 阵风系数参数
        gust_factor_config = self.wind_config.get('gust_factor', {})
        self.gust_coeff_a = float(gust_factor_config.get('coefficient_a', -5.0681))
        self.gust_coeff_b = float(gust_factor_config.get('coefficient_b', 12.2611))

    def mean_wind(self, height):
        """
        计算平均风速

        Args:
            height: 高度 (m)

        Returns:
            float: 平均风速 (m/s)
        """
        if height <= 0:
            return 0.0
        return self.v0 * (height / self.z0) ** self.alpha  # 幂律风速剖面公式v(z) = v₀ × (z/z₀)^α

    def gust_factor(self, height):
        """
        计算阵风系数

        Args:
            height: 高度 (m)

        Returns:
            float: 阵风系数
        """

        return np.exp(self.gust_coeff_a * height + self.gust_coeff_b)

    def gust_wind(self, height, time):
        """
        计算阵风速度

        Args:
            height: 高度 (m)
            time: 时间 (s)

        Returns:
            float: 阵风速度 (m/s)
        """
        P = self.gust_factor(height)
        v_mean = self.mean_wind(height)
        v_max = P * v_mean

        # 阵风时间窗口
        t1 = self.gust_start_time
        T_g = self.gust_duration

        if time < t1:
            return 0.0
        elif t1 <= time <= t1 + T_g:
            # 余弦形式的阵风
            phase = np.pi * (time - t1) / T_g
            return (v_max / 2) * (1 - np.cos(2 * phase))
        else:
            return 0.0

    def random_wind(self, time, seed=None):
        """
        计算随机风速度

        Args:
            time: 时间 (s)
            seed: 随机种子

        Returns:
            float: 随机风速度 (m/s)
        """
        if self.random_max_amplitude == 0.0:
            return 0.0

        if seed is not None:
            np.random.seed(int(seed + time))

        # 随机数 [-1, 1]
        R = 2 * np.random.random() - 1

        # 随机相位 [0, 2π]
        phi = 2 * np.pi * np.random.random()

        return self.random_max_amplitude * R * np.sin(
            2 * np.pi * self.random_frequency * time + phi
        )

    def total_wind_speed(self, height, time, seed=None):
        """
        计算总风速（平均风 + 阵风 + 随机风）

        Args:
            height: 高度 (m)
            time: 时间 (s)
            seed: 随机种子

        Returns:
            dict: 包含各分量和总风速的字典
        """
        v_mean = self.mean_wind(height)
        v_gust = self.gust_wind(height, time)
        v_random = self.random_wind(time, seed)
        v_total = v_mean + v_gust + v_random

        return {
            'mean': v_mean,
            'gust': v_gust,
            'random': v_random,
            'total': max(0, v_total)  # 确保风速不为负，因为v_mean，v_gust，v_random都为标量
        }

    def calculate_wind_direction(self, time):
        """
        计算风向（考虑变化）

        Args:
            time: 时间 (s)

        Returns:
            np.ndarray: 标准化的风向向量
        """
        # 基础风向（主风向）
        base_direction = self.primary_direction.copy()

        # 添加风向变化
        if self.direction_variation > 0:
            variation_x = self.direction_variation * np.sin(self.direction_frequency * time)
            variation_y = self.direction_variation * np.cos(self.direction_frequency * time * 0.7)  # 不同频率避免规律性

            # 在水平面上添加变化
            direction = base_direction + np.array([variation_x, variation_y, 0.0])

            # 重新归一化
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 1e-6:
                direction = direction / direction_norm
            else:
                direction = base_direction
        else:
            direction = base_direction

        return direction

    def get_wind(self, position: np.ndarray, dt: float) -> np.ndarray:
        """获取指定位置的风速向量 """
        self.time += dt

        height = max(position[2], 1)  # 确保高度至少为1m（高度不能为0.且接近0数值不稳定）

        # 获取总风速
        wind_data = self.total_wind_speed(height, self.time)
        total_speed = wind_data['total']

        # 计算风向（考虑变化）
        wind_direction = self.calculate_wind_direction(self.time)

        # 计算三维风速向量
        wind_vector = wind_direction * total_speed

        return wind_vector.astype(np.float32)

    def get_wind_statistics(self) -> Dict:
        """获取风场统计信息"""
        return {
            'intensity': self.intensity.name,
            'current_time': self.time,
            'parameters': {
                'base_speed': self.v0,
                'reference_height': self.z0,
                'wind_profile_exponent': self.alpha,
                'gust_start_time': self.gust_start_time,
                'gust_duration': self.gust_duration,
                'random_max_amplitude': self.random_max_amplitude,
                'random_frequency': self.random_frequency
            },
            'direction_config': {
                'primary_direction': self.primary_direction.tolist(),
                'direction_variation': self.direction_variation,
                'direction_frequency': self.direction_frequency,

            },
            'config_source': self.wind_config
        }

    def reset(self):
        """重置风场时间"""
        self.time = 0.0


# 为了向后兼容，保持WindField名称但直接使用UrbanWindField
class WindField(UrbanWindField):
    """城市风场"""

    def __init__(self, intensity: WindIntensity = WindIntensity.NORMAL,
                 config: dict = None, **kwargs):
        """
        初始化风场

        Args:
            intensity: 风场强度
            config: 配置字典
            **kwargs: 其他参数（为了兼容性，会被忽略）
        """
        super().__init__(intensity=intensity, config=config)


# 训练阶段与风场强度的映射
def get_wind_intensity_for_stage(stage_value: int) -> WindIntensity:
    """根据训练阶段返回对应的风场强度"""
    stage_wind_map = {
        1: WindIntensity.CALM,  # BASIC_FLIGHT: 无风
        2: WindIntensity.NORMAL,  # SIMPLE_NAV: 只有均匀风速，无扰动
        3: WindIntensity.MODERATE,  # COMPLEX_NAV: 添加轻微扰动
        4: WindIntensity.EXTREME  # GENERALIZATION: 完整风场模型
    }
    return stage_wind_map.get(stage_value, WindIntensity.NORMAL)


if __name__ == "__main__":
    # 测试各种风场强度
    print("=== 风场模型测试 ===")

    # 创建测试配置
    test_config = {
        'wind': {
            'intensity_params': {
                'NORMAL': {
                    'base_speed': 8.0,
                    'wind_profile_exponent': 0.29,
                    'gust_start_time': 500.0,
                    'gust_duration': 0.1,
                    'random_max_amplitude': 0.0,
                    'random_frequency': 0.0
                }
            },
            'primary_direction': [0.8, 0.6, 0.0],  # 自定义风向
            'direction_variation': 0.15
        }
    }

    for intensity in WindIntensity:
        print(f"\n=== {intensity.name} 风场测试 ===")
        wind_model = WindField(intensity, config=test_config if intensity == WindIntensity.NORMAL else None)

        # 测试50m高度的风速
        position = np.array([0, 0, 100])  # 50m高度
        wind_vector = wind_model.get_wind(position, dt=0.1)

        # 获取详细风速分析
        result = wind_model.total_wind_speed(height=50, time=10.0, seed=42)
        stats = wind_model.get_wind_statistics()

        print(f"50m高度的风速向量: [{wind_vector[0]:.2f}, {wind_vector[1]:.2f}, {wind_vector[2]:.2f}] m/s")
        print(
            f"风速分析 - 平均: {result['mean']:.2f}, 阵风: {result['gust']:.2f}, 随机: {result['random']:.2f}, 总计: {result['total']:.2f} m/s")
        print(
            f"配置参数 - 基础风速: {stats['parameters']['base_speed']:.1f} m/s, 风向: {stats['direction_config']['primary_direction']}")