"""
二阶旋转倒立摆 (Rotary Double Inverted Pendulum) - PPO 强化学习控制
包含: 数学建模 (RK4 积分)、Gymnasium 环境搭建、PPO 算法控制、PyGame 可视化
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import os
import torch
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.env_checker import check_env

class RotaryDoublePendulumEnv(gym.Env):
    """
    自定义二阶旋转倒立摆环境
    动力学内核基于等效小车双倒立摆，视觉层映射为旋转臂。
    目标：使两个摆杆保持垂直向上（theta1 = 0, theta2 = 0），并将旋转臂保持在中心位置。
    """
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super(RotaryDoublePendulumEnv, self).__init__()
        self.render_mode = render_mode

        # 物理参数定义 (Mathematical Modeling)
        self.m0 = 1.0    # 旋转臂等效质量
        self.m1 = 0.5    # 摆杆 1 质量
        self.m2 = 0.25   # 摆杆 2 质量
        self.l1 = 0.5    # 摆杆 1 质心长度
        self.l2 = 0.5    # 摆杆 2 质心长度
        self.g = 9.81    # 重力加速度

        # 摩擦系数
        self.b0 = 0.1    # 旋转臂摩擦
        self.b1 = 0.05   # 关节 1 摩擦
        self.b2 = 0.05   # 关节 2 摩擦

        self.max_force = 20.0  # 电机最大输出力矩/力
        self.dt = 0.02         # 控制周期 20ms

        # 状态边界 (终止条件)
        self.x_threshold = 2.0        # 旋转臂最大等效线位移
        self.theta_threshold = 0.5    # 摆角偏离垂直的最大弧度 (~28度)

        # 动作空间：连续力矩输出 [-1.0, 1.0]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # 状态空间观察量 (8维): [x, dx, cos(th1), sin(th1), dth1, cos(th2), sin(th2), dth2]
        high = np.array([self.x_threshold*2, np.inf, 1.0, 1.0, np.inf, 1.0, 1.0, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.state = None
        self.screen = None
        self.clock = None

    def _compute_derivatives(self, state, action):
        """
        拉格朗日方程的非线性动力学矩阵求解
        M(q) * q_ddot = B(q, q_dot, tau)
        """
        x, th1, th2, dx, dth1, dth2 = state
        tau = np.clip(action[0], -1.0, 1.0) * self.max_force

        # 质量惯性矩阵 M(q)
        M11 = self.m0 + self.m1 + self.m2
        M12 = (self.m1 + self.m2) * self.l1 * np.cos(th1)
        M13 = self.m2 * self.l2 * np.cos(th2)

        M21 = M12
        M22 = (self.m1 + self.m2) * self.l1**2
        M23 = self.m2 * self.l1 * self.l2 * np.cos(th1 - th2)

        M31 = M13
        M32 = M23
        M33 = self.m2 * self.l2**2

        M = np.array([[M11, M12, M13],
                      [M21, M22, M23],
                      [M31, M32, M33]])

        # 广义力与科里奥利力/重力向量 B
        B1 = tau + (self.m1 + self.m2) * self.l1 * dth1**2 * np.sin(th1) + self.m2 * self.l2 * dth2**2 * np.sin(th2) - self.b0 * dx
        B2 = -self.m2 * self.l1 * self.l2 * dth2**2 * np.sin(th1 - th2) + (self.m1 + self.m2) * self.g * self.l1 * np.sin(th1) - self.b1 * dth1
        B3 = self.m2 * self.l1 * self.l2 * dth1**2 * np.sin(th1 - th2) + self.m2 * self.g * self.l2 * np.sin(th2) - self.b2 * dth2

        B = np.array([B1, B2, B3])

        # 求解加速度 q_ddot
        try:
            q_ddot = np.linalg.solve(M, B)
        except np.linalg.LinAlgError:
            q_ddot = np.zeros(3) # 防止奇异矩阵导致的崩溃

        return np.array([dx, dth1, dth2, q_ddot[0], q_ddot[1], q_ddot[2]])

    def step(self, action):
        """RK4 四阶龙格库塔积分更新环境"""
        # 1. RK4 积分
        k1 = self._compute_derivatives(self.state, action)
        k2 = self._compute_derivatives(self.state + 0.5 * self.dt * k1, action)
        k3 = self._compute_derivatives(self.state + 0.5 * self.dt * k2, action)
        k4 = self._compute_derivatives(self.state + self.dt * k3, action)

        self.state = self.state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # 提取状态
        x, th1, th2, dx, dth1, dth2 = self.state

        # 规范化角度到 [-pi, pi]
        th1 = ((th1 + np.pi) % (2 * np.pi)) - np.pi
        th2 = ((th2 + np.pi) % (2 * np.pi)) - np.pi
        self.state[1] = th1
        self.state[2] = th2

        # 2. 终止条件判断 (超出安全边界即失败)
        terminated = bool(
            x < -self.x_threshold or x > self.x_threshold
            or th1 < -self.theta_threshold or th1 > self.theta_threshold
            or th2 < -self.theta_threshold or th2 > self.theta_threshold
        )
        truncated = False

        # 3. 奖励函数构造 (Reward Shaping - 极为重要)
        if not terminated:
            # 存活奖励 + 偏离惩罚
            alive_bonus = 10.0
            # 鼓励角度为0，位置为0，速度尽量小
            reward = alive_bonus \
                     - 15.0 * (th1**2) \
                     - 15.0 * (th2**2) \
                     - 1.0 * (x**2) \
                     - 0.1 * (dx**2) \
                     - 0.1 * (dth1**2) \
                     - 0.1 * (dth2**2) \
                     - 0.05 * (action[0]**2)
        else:
            reward = -100.0 # 失败严厉惩罚

        # 4. 返回观察值
        obs = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 初始化在直立位置附近（稳定任务 Curriculum）
        # 如果需要实现完全起摆，可以将 0.05 扩大为 np.pi
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(6,))
        return self._get_obs(), {}

    def _get_obs(self):
        x, th1, th2, dx, dth1, dth2 = self.state
        return np.array([
            x, dx,
            np.cos(th1), np.sin(th1), dth1,
            np.cos(th2), np.sin(th2), dth2
        ], dtype=np.float32)

    def render(self):
        """基于 PyGame 的 2D 投影可视化"""
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.window_width = 800
            self.window_height = 600
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("Rotary Double Inverted Pendulum - PPO")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.screen.fill((240, 248, 255)) # 背景色

        # 获取状态
        x, th1, th2, _, _, _ = self.state

        # 视觉缩放比例
        scale = 150

        # 将 x 映射为旋转臂的角度 (视觉效果)
        # 假设旋转臂长度 R=1.0，则 theta0 = x / R
        arm_radius = 1.2
        theta0 = x / arm_radius

        # 旋转臂中心点 (固定)
        center_x = self.window_width // 2
        center_y = self.window_height // 2 + 100

        # 关节 1：旋转臂末端位置 (投影在 2D 屏幕上画椭圆弧度)
        j1_x = center_x + int(arm_radius * scale * np.sin(theta0))
        # 用余弦制造透视景深感
        j1_y = center_y - int(arm_radius * scale * 0.3 * np.cos(theta0))

        # 关节 2：第一级摆杆末端位置
        j2_x = j1_x + int(self.l1 * scale * np.sin(th1))
        j2_y = j1_y - int(self.l1 * scale * np.cos(th1)) # 屏幕 y 轴向下，减号向上

        # 末端点：第二级摆杆末端位置
        end_x = j2_x + int(self.l2 * scale * np.sin(th2))
        end_y = j2_y - int(self.l2 * scale * np.cos(th2))

        # 绘制中心基座
        pygame.draw.circle(self.screen, (50, 50, 50), (center_x, center_y), 15)
        # 绘制旋转臂轨迹参考线
        pygame.draw.ellipse(self.screen, (200, 200, 200),
                            (center_x - int(arm_radius*scale), center_y - int(arm_radius*scale*0.3),
                             int(arm_radius*scale*2), int(arm_radius*scale*0.6)), 2)

        # 绘制旋转臂
        pygame.draw.line(self.screen, (100, 100, 200), (center_x, center_y), (j1_x, j1_y), 8)
        pygame.draw.circle(self.screen, (200, 50, 50), (j1_x, j1_y), 10)

        # 绘制摆杆 1
        pygame.draw.line(self.screen, (200, 150, 50), (j1_x, j1_y), (j2_x, j2_y), 6)
        pygame.draw.circle(self.screen, (50, 200, 50), (j2_x, j2_y), 8)

        # 绘制摆杆 2
        pygame.draw.line(self.screen, (200, 150, 50), (j2_x, j2_y), (end_x, end_y), 6)
        pygame.draw.circle(self.screen, (50, 50, 200), (end_x, end_y), 8)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None

class TqdmCallback(BaseCallback):
    """
    自定义 tqdm 进度条回调函数，附加训练指标显示
    """
    def __init__(self, total_timesteps: int, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None
        self.episodes = 0

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="PPO 训练进度 (CUDA)", unit="step")

    def _on_step(self):
        # 动态更新进度条，用实际步数减去进度条当前步数
        self.pbar.update(self.model.num_timesteps - self.pbar.n)

        # 统计完成的轮次 (dones)
        dones = self.locals.get("dones")
        if dones is not None:
            self.episodes += np.sum(dones)

        # 从模型的缓存区中获取最近 100 轮的平均奖励和平均步长
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            mean_length = np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])

            # 在进度条右侧显示附加信息
            self.pbar.set_postfix({
                "轮次": self.episodes,
                "平均奖励": f"{mean_reward:.2f}",
                "平均步长": f"{mean_length:.1f}"
            })

        return True

    def _on_training_end(self):
        if self.pbar is not None:
            self.pbar.close()

def train_model():
    """使用 PPO 训练模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在检查环境完整性... (将使用设备: {device})")

    env = RotaryDoublePendulumEnv()
    check_env(env)
    print("环境检查通过！")

    # 构建 PPO 算法，网络架构设计为两层 128 神经元的 MLP
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))

    # 注意：明确指定 device，并将 verbose 设为 0 以防控制台日志打乱 tqdm 进度条
    model = PPO("MlpPolicy", env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=512,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.0,
                policy_kwargs=policy_kwargs,
                device=device,
                verbose=0,
                tensorboard_log="./ppo_rotary_pendulum_tb/")

    # 设置评估回调，每隔 10000 步评估一次，并保存最好模型
    eval_env = RotaryDoublePendulumEnv()
    os.makedirs("./models/", exist_ok=True)
    eval_callback = EvalCallback(eval_env, best_model_save_path='./models/',
                                 log_path='./models/', eval_freq=10000,
                                 deterministic=True, render=False)

    total_timesteps = 500000
    tqdm_callback = TqdmCallback(total_timesteps=total_timesteps)

    print("开始训练 PPO 模型 (预计需要 5-10 分钟收敛)...")
    # 传入 callback 列表
    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, tqdm_callback])

    model.save("./models/ppo_rotary_final")
    print("训练完成并保存！")

def test_model():
    """测试并可视化已经训练好的模型"""
    model_path = "./models/best_model.zip"
    if not os.path.exists(model_path):
        model_path = "./models/ppo_rotary_final.zip"
        if not os.path.exists(model_path):
            print("找不到模型文件，请先运行训练模式！")
            return

    env = RotaryDoublePendulumEnv(render_mode="human")
    model = PPO.load(model_path, env=env)

    obs, _ = env.reset()
    print("开始渲染...")
    try:
        for i in range(2000):
            # deterministic=True 保证测试时使用网络输出的均值，消除高斯噪声震荡
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__ == '__main__':
    # =========================================================================
    # 模式控制开关 (MODE Control)
    # 第一次运行请将 MODE 设为 "train"，训练完成后改为 "test" 进行可视化验证
    # =========================================================================
    MODE = "test"  # 改为 "test" 来观察训练好的动画

    if MODE == "train":
        train_model()
    elif MODE == "test":
        test_model()