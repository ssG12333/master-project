"""
控制输入: [T_Lift, T_Cruise, α, φ]
"""

import numpy as np
from typing import Dict, Optional


class LiftCruiseEVTOLDynamics:
    '''3-DOF质点动力学模型'''

    def __init__(self, config: dict = None, dt: float = 0.1):
        self.dt = float(dt)
        self.g = 9.81  # 重力加速度 (m/s²)

        # 从配置文件读取参数
        if config and 'environment' in config:
            env_config = config['environment']
            self.mass = float(env_config.get('mass', 2653.0))
            self.wingspan = float(env_config.get('wingspan', 14.5))
            self.wing_area = float(env_config.get('wing_area', 17.0))
            self.max_speed = float(env_config.get('max_speed', 100.0))
            self.min_speed = float(env_config.get('min_speed', 0.1))
            self.max_altitude = float(env_config.get('max_altitude', 300.0))
            self.max_range = float(env_config.get('max_range', 50000.0))

            # 推力限制
            self.max_thrust_lift = float(env_config.get('max_thrust_lift', 47000.0))
            self.max_thrust_cruise = float(env_config.get('max_thrust_cruise', 35000.0))

            # 过渡速度参数
            self.v_start = float(env_config.get('transition_start_speed', 20.0))
            self.v_end = float(env_config.get('transition_end_speed', 44.0))

            # 最大角度限制
            self.max_alpha = np.radians(float(env_config.get('max_alpha_deg', 90.0)))
            self.max_phi = np.radians(float(env_config.get('max_phi_deg', 60.0)))

            # 气动参数
            self.CL_alpha = float(env_config.get('lift_slope', 6.65))
            self.CD0 = float(env_config.get('parasitic_drag', 0.07))
            self.k = float(env_config.get('induced_drag_factor', 0.04))
            self.air_density = float(env_config.get('air_density', 1.225))
        else:
            # 默认参数
            self.mass = 2653.0
            self.max_thrust_lift = 47000.0
            self.max_thrust_cruise = 35000.0
            self.v_start = 20.0
            self.v_end = 44.0
            self.max_alpha = np.radians(90.0)
            self.max_phi = np.radians(60.0)
            self.CL_alpha = 6.65
            self.CD0 = 0.07
            self.k = 0.04
            self.air_density = 1.225
            self.wingspan = 14.5
            self.wing_area = 17.0
            self.max_speed = 100.0
            self.min_speed = 0.1
            self.max_altitude = 300.0
            self.max_range = 50000.0

        # 状态变量
        self.position = np.zeros(3, dtype=np.float32)  # [x, y, h]
        self.velocity_magnitude = 0.1  # V (速度大小)
        self.gamma = 0.0  # γ (飞行路径角)
        self.chi = 0.0  # χ (航向角)

        # 控制输入: [T_Lift, T_Cruise, α, φ]
        self.thrust_lift = 0.0
        self.thrust_cruise = 0.0
        self.alpha = 0.0  # 攻角
        self.phi = 0.0  # 滚转角

        # 速度和加速度
        self.velocity = np.zeros(3, dtype=np.float32)
        self.acceleration = np.zeros(3, dtype=np.float32)
        self.total_distance = 0.0

    def _transition_weight(self, V: float) -> float:
        """计算混合因子 W(V)"""
        if V <= self.v_start:
            return 0.0
        elif V >= self.v_end:
            return 1.0
        else:
            # 使用余弦函数确保连续和平滑过渡
            phase = np.pi * (V - self.v_start) / (self.v_end - self.v_start)
            return 0.5 * (1 - np.cos(phase))

    def _compute_aerodynamic_forces(self, V: float, alpha: float) -> tuple:
        """计算气动力 """
        q = 0.5 * self.air_density * V ** 2  # 动压

        # 升力系数
        CL = self.CL_alpha * alpha
        CL = np.clip(CL, -0.5, 1.8)

        # 阻力系数
        CD = self.CD0 + self.k * CL ** 2

        # 升力和阻力
        L = q * self.wing_area * CL
        D = q * self.wing_area * CD

        return L, D

    def step(self, action: np.ndarray, wind_velocity: np.ndarray = None) -> Dict:
        '''更新动力学状态

        Args:
            action: [T_Lift_normalized, T_Cruise_normalized, alpha_normalized, phi_normalized] ∈ [-1, 1]⁴
            wind_velocity: 风速矢量 [vx_wind, vy_wind, vz_wind]
        '''
        if wind_velocity is None:
            wind_velocity = np.zeros(3, dtype=np.float32)

        action = np.array(action, dtype=np.float32)

        # 将控制输入 (从 [-1,1] 映射到物理范围)
        self.thrust_lift = float((action[0] + 1) / 2 * self.max_thrust_lift)
        self.thrust_cruise = float((action[1] + 1) / 2 * self.max_thrust_cruise)
        self.alpha = float(action[2] * self.max_alpha)
        self.phi = float(action[3] * self.max_phi)

        # 限制攻角不超过 +/- 15 度（失速)
        MAX_AOA = np.radians(15.0)
        self.alpha = np.clip(self.alpha, -MAX_AOA, MAX_AOA)

        # 当前状态
        x, y, h = self.position
        V = self.velocity_magnitude
        gamma = self.gamma
        chi = self.chi

        # 数值保护
        V_safe_limit = 1 # 设定 V_safe 的下限
        V_safe = max(V, V_safe_limit)
        h = max(h, 0.0)
        cos_gamma_safe = max(np.cos(gamma), 0.17)  # 用于保护 chi_dot

        # 计算混合因子
        W_V = self._transition_weight(V)

        # 气动力计算
        L, D = self._compute_aerodynamic_forces(V, self.alpha)

        # 力学方程
        # F_gamma: 垂直/升力分量
        F_gamma = (self.thrust_lift ) * (1 - W_V) + L * W_V

        # F_V: 水平/推进分量
        F_V = (self.thrust_cruise * np.cos(self.alpha)) - (D * W_V)

        # 微分方程飞行器构型与模式定义构型Lift+Cruise 复合翼（8个垂直旋翼 TLift, 1个推进螺旋桨 TCruise）动力学模型3 自由度（DOF）safe}} \cos\gamma_{\text{safe}}}$法向力 $F_{\gamma}$ 的侧向分量驱动转弯。$\frac{dx}{dt}, \frac{dy}{dt}, \frac{dh}{dt}$$V \cos\gamma \cos\chi$ 等几何关系积分（位置更新）。
        V_dot = (F_V / self.mass) - self.g * np.sin(gamma)
        gamma_dot = (F_gamma * np.cos(self.phi) - self.mass * self.g * np.cos(gamma)) / (self.mass * V_safe)
        chi_dot = (F_gamma * np.sin(self.phi)) / (self.mass * V_safe * cos_gamma_safe)

        # 即使 V_safe 保护了分母，高推力仍可能导致导数过大，
        # 故对角速率施加物理限制（例如，最大 60 度/秒）
        MAX_ANGULAR_RATE = np.radians(60.0)
        gamma_dot = np.clip(gamma_dot, -MAX_ANGULAR_RATE, MAX_ANGULAR_RATE)
        chi_dot = np.clip(chi_dot, -MAX_ANGULAR_RATE, MAX_ANGULAR_RATE)

        # 状态更新
        new_x = x + V * np.cos(gamma) * np.cos(chi) * self.dt
        new_y = y + V * np.cos(gamma) * np.sin(chi) * self.dt
        new_h = h + V * np.sin(gamma) * self.dt
        new_V = V + V_dot * self.dt
        new_gamma = gamma + gamma_dot * self.dt
        new_chi = chi + chi_dot * self.dt

        # 约束
        new_V = np.clip(new_V, self.min_speed, self.max_speed)
        new_h = max(new_h, 0.0)
        new_gamma = np.clip(new_gamma, -np.pi/2, np.pi/2)

        # 更新状态
        self.position = np.array([new_x, new_y, new_h], dtype=np.float32)
        self.velocity_magnitude = new_V
        self.gamma = new_gamma
        self.chi = new_chi

        # 计算速度和加速度矢量
        self.velocity = np.array([
            new_V * np.cos(new_gamma) * np.cos(new_chi),
            new_V * np.cos(new_gamma) * np.sin(new_chi),
            new_V * np.sin(new_gamma)
        ], dtype=np.float32)

        self.acceleration = np.array([
            V_dot * np.cos(gamma) * np.cos(chi),
            V_dot * np.cos(gamma) * np.sin(chi),
            V_dot * np.sin(gamma)
        ], dtype=np.float32)

        # 更新总距离
        distance_step = V * self.dt
        self.total_distance += distance_step

        # 检查违规
        violations = self.check_violations()

        # 计算抖动
        jerk = np.linalg.norm([V_dot, gamma_dot, chi_dot]) / self.dt

        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'acceleration': self.acceleration.copy(),
            'jerk': np.array([jerk, 0, 0], dtype=np.float32),
            'violations': violations,
            'aerodynamics': {
                'lift': L,
                'drag': D,
                'CL': self.CL_alpha * self.alpha,
                'dynamic_pressure': 0.5 * self.air_density * V ** 2,
                'transition_weight': W_V
            }
        }

    def check_violations(self) -> Dict:
        '''检查物理限制违反'''
        violations = {}

        V = self.velocity_magnitude
        h = self.position[2]

        # 速度限制
        if V > self.max_speed:
            violations['overspeed'] = V - self.max_speed
        elif V < self.min_speed:
            violations['underspeed'] = self.min_speed - V

        # 高度限制
        if h > self.max_altitude:
            violations['altitude_exceeded'] = h - self.max_altitude
        elif h < 0:
            violations['ground_collision'] = -h

        # 航程限制
        if self.total_distance > self.max_range:
            violations['range_exceeded'] = self.total_distance - self.max_range

        return violations

    def reset(self, position: Optional[np.ndarray] = None):
        '''重置状态'''
        if position is not None:
            self.position = np.array(position, dtype=np.float32)
        else:
            self.position = np.zeros(3, dtype=np.float32)

        self.velocity_magnitude = 0.1
        self.gamma = 0.0
        self.chi = 0.0
        self.thrust_lift = 0.0
        self.thrust_cruise = 0.0
        self.alpha = 0.0
        self.phi = 0.0
        self.velocity = np.zeros(3, dtype=np.float32)
        self.acceleration = np.zeros(3, dtype=np.float32)
        self.total_distance = 0.0

    def get_state_vector(self) -> np.ndarray:
        '''获取完整状态向量'''
        return np.array([
            self.position[0],
            self.position[1],
            self.position[2],
            self.velocity_magnitude,
            self.gamma,
            self.chi
        ], dtype=np.float32)

    def get_control_vector(self) -> np.ndarray:
        '''获取当前控制输入'''
        return np.array([
            self.thrust_lift,
            self.thrust_cruise,
            self.alpha,
            self.phi
        ], dtype=np.float32)


# 为了向后兼容
EVTOLDynamics = LiftCruiseEVTOLDynamics

if __name__ == "__main__":
    # 测试代码
    print("=== Lift+Cruise eVTOL动力学模型测试 ===\n")

    # 创建动力学模型
    dynamics = LiftCruiseEVTOLDynamics()

    print("飞行器参数:")
    print(f"  质量: {dynamics.mass} kg")
    print(f"  翼展: {dynamics.wingspan} m")
    print(f"  翼面积: {dynamics.wing_area} m²")
    print(f"  最大垂直推力: {dynamics.max_thrust_lift} N")
    print(f"  最大巡航推力: {dynamics.max_thrust_cruise} N")
    print(f"  转换速度范围: [{dynamics.v_start}, {dynamics.v_end}] m/s\n")

    # 初始化
    dynamics.reset(position=np.array([0, 0, 100]))  # 100m高度

    print("初始状态:")
    print(f"  位置: {dynamics.position}")
    print(f"  速度大小: {dynamics.velocity_magnitude:.2f} m/s")
    print(f"  飞行路径角: {np.degrees(dynamics.gamma):.2f}°")
    print(f"  航向角: {np.degrees(dynamics.chi):.2f}°\n")

    # -------------------------------------------------------------
    # 修正后的场景1A: 验证模型物理正确性 (垂直加速爬升)
    # T_Lift 用于克服重力，T_Cruise 用于加速 V。
    # T_Cruise 最大值 15000N，提供的加速度最多 15000/2653 ≈ 5.65 m/s²
    # 重力阻力最大 9.81 m/s²。因此 T_Cruise 必须为 1.0 且 T_Lift 足够大。

    # 我们用 T_Lift=0.9 (略大于悬停所需) 和 T_Cruise=0.9
    print("场景1A (修正): 垂直爬升并加速 (10秒)")
    dynamics.reset(position=np.array([0, 0, 100]))

    # 计算悬停所需推力：F = m*g = 26027 N. 归一化推力 ≈ 26027 / 47000
    hover_T_lift_norm = (2653 * 9.81 / dynamics.max_thrust_lift) * 1.05  # 略微大于悬停推力

    for i in range(100):
        # [T_Lift, T_Cruise, α, φ]
        action = np.array([hover_T_lift_norm, 0.8, 0.0, 0.0])
        info = dynamics.step(action)

        if i % 20 == 0:
            print(f"  t={i * 0.1:.1f}s: h={dynamics.position[2]:.1f}m, "
                  f"V={dynamics.velocity_magnitude:.1f}m/s, "
                  f"γ={np.degrees(dynamics.gamma):.1f}°")

    # -------------------------------------------------------------
    # 场景1B: 您的原始测试 (垂直悬停尝试，但无巡航推力)
    # 证明 V 锁死是物理正确的结果
    print("\n场景1B (原始): 垂直悬停尝试 (无巡航推力，10秒)")
    dynamics.reset(position=np.array([0, 0, 100]))

    for i in range(100):
        # [T_Lift, T_Cruise, α, φ]
        action = np.array([1.0, 0.0, 0.0, 0.0])  # 最大垂直推力，无巡航推力，无攻角和滚转
        info = dynamics.step(action)

        if i % 20 == 0:
            print(f"  t={i * 0.1:.1f}s: h={dynamics.position[2]:.1f}m, "
                  f"V={dynamics.velocity_magnitude:.1f}m/s, "
                  f"γ={np.degrees(dynamics.gamma):.1f}°")

    # -------------------------------------------------------------
    # 测试场景2: 过渡飞行与转弯 (保持不变)
    print("\n场景2: 过渡飞行和转弯 (10秒)")
    dynamics.reset(position=np.array([0, 0, 100]))
    dynamics.velocity_magnitude = 30.0

    for i in range(100):
        # [T_Lift, T_Cruise, α, φ]
        action = np.array([0.5, 0.7, 0.2, 0.5])  # 中等垂直推力，较大巡航推力，小攻角，右滚转
        info = dynamics.step(action)

        if i % 20 == 0:
            L = info['aerodynamics']['lift']
            D = info['aerodynamics']['drag']
            W = info['aerodynamics']['transition_weight']
            print(f"  t={i * 0.1:.1f}s: pos=({dynamics.position[0]:.1f}, "
                  f"{dynamics.position[1]:.1f}), χ={np.degrees(dynamics.chi):.1f}°, "
                  f"L/D={L / (D + 1e-6):.2f}, W(V)={W:.2f}")
    # --- 在原有代码的基础上添加以下测试场景 ---

    # -------------------------------------------------------------
    # 场景 3: 极限高速爬升与失速边缘 (15秒)
    # -------------------------------------------------------------
    print("\n场景 3: 极限高速爬升 (15秒)")
    dynamics.reset(position=np.array([0, 0, 100]))
    dynamics.velocity_magnitude = 80.0  # 高速起始

    for i in range(150):
        # [T_Lift, T_Cruise, α, φ]
        # 持续最大巡航推力，中等攻角，保持稳定爬升角 (小滚转角)
        action = np.array([0.0, 1.0, 0.1, 0.1])
        info = dynamics.step(action)

        if i % 30 == 0:
            W = info['aerodynamics']['transition_weight']
            L = info['aerodynamics']['lift']
            D = info['aerodynamics']['drag']
            print(f"  t={i * 0.1:.1f}s: V={dynamics.velocity_magnitude:.1f}m/s, "
                  f"γ={np.degrees(dynamics.gamma):.1f}°, L/D={L / (D + 1e-6):.2f}, W={W:.2f}")

    # -------------------------------------------------------------
    # 场景 4: 极限俯冲与拉起 (10秒)
    # -------------------------------------------------------------
    print("\n场景 4: 极限俯冲与拉起 (10秒)")
    dynamics.reset(position=np.array([0, 0, 100]))
    dynamics.velocity_magnitude = 50.0

    # 阶段 1: 俯冲 (5秒)
    for i in range(50):
        # [T_Lift, T_Cruise, α, φ]
        # 无推力，负攻角，持续俯冲
        action = np.array([0.0, 0.0, -0.2, 0.0])
        info = dynamics.step(action)
        if i == 49:
            print(
                f"  t=5.0s (俯冲结束): h={dynamics.position[2]:.1f}m, V={dynamics.velocity_magnitude:.1f}m/s, γ={np.degrees(dynamics.gamma):.1f}°")

    # 阶段 2: 大G拉起 (5秒)
    for i in range(50, 100):
        # [T_Lift, T_Cruise, α, φ]
        # 最大垂直推力，最大正攻角，试图拉起
        action = np.array([1.0, 0.0, 1.0, 0.0])
        info = dynamics.step(action)

        if i % 20 == 0:
            print(
                f"  t={i * 0.1:.1f}s (拉起): h={dynamics.position[2]:.1f}m, V={dynamics.velocity_magnitude:.1f}m/s, γ={np.degrees(dynamics.gamma):.1f}°")

    # -------------------------------------------------------------
    # 场景 5: 零速悬停与横向平移 (10秒)
    # -------------------------------------------------------------
    print("\n场景 5: 零速悬停与横向平移 (10秒)")
    dynamics.reset(position=np.array([0, 0, 100]))
    dynamics.velocity_magnitude = 0.1

    # 计算悬停所需推力：m*g / T_Lift_max
    hover_T_lift_norm = (dynamics.mass * dynamics.g / dynamics.max_thrust_lift)
    print(f"  悬停所需归一化 T_Lift: {hover_T_lift_norm:.3f}")

    for i in range(100):
        # [T_Lift, T_Cruise, α, φ]
        # 悬停推力，无巡航，小滚转角产生横向力
        action = np.array([hover_T_lift_norm, -1.0, 0.0, 0.3])
        info = dynamics.step(action)

        if i % 20 == 0:
            pos = dynamics.position
            print(
                f"  t={i * 0.1:.1f}s: pos=({pos[0]:.1f}, {pos[1]:.1f}), V={dynamics.velocity_magnitude:.1f}m/s, χ={np.degrees(dynamics.chi):.1f}°")

    # -------------------------------------------------------------
    # 场景 6: 临界过渡速度操作 (10秒)
    # -------------------------------------------------------------
    print("\n场景 6: 临界过渡速度操作 (10秒)")
    dynamics.reset(position=np.array([0, 0, 100]))
    dynamics.velocity_magnitude = 30.0  # 位于 [20, 44] 转换区间内

    for i in range(100):
        # [T_Lift, T_Cruise, α, φ]
        # 在过渡区间内施加大推力和大滚转，检查稳定性
        action = np.array([0.7, 0.7, 0.5, 0.8])
        info = dynamics.step(action)

        if i % 20 == 0:
            W = info['aerodynamics']['transition_weight']
            gamma_dot = info['jerk'][0] * dynamics.dt  # 粗略检查导数的平滑性
            print(
                f"  t={i * 0.1:.1f}s: V={dynamics.velocity_magnitude:.1f}m/s, W(V)={W:.2f}, γ_dot≈{gamma_dot:.2f} rad/s")

    # --- 结束 ---



    print("\n测试完成!")