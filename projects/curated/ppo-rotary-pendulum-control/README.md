# PPO 二阶旋转倒立摆控制

## 项目简介

本项目构建了一个**二阶旋转倒立摆 (Rotary Double Inverted Pendulum)** 的 Gymnasium 自定义环境, 并使用 Stable-Baselines3 的 PPO (Proximal Policy Optimization) 算法训练控制策略。项目涵盖了从物理建模 (拉格朗日力学, RK4 积分) 到强化学习训练 (PPO, 自定义回调) 再到可视化 (PyGame 实时渲染) 的完整流程。

原始目录：`二阶旋转倒立摆/`

## 技术栈

- Python 3.8+, PyTorch
- Gymnasium (自定义环境)
- Stable-Baselines3 (PPO)
- NumPy (矩阵运算, RK4 积分)
- PyGame (物理可视化)
- tqdm (训练进度条)

## 自定义环境: RotaryDoublePendulumEnv

### 物理参数

```python
class RotaryDoublePendulumEnv(gym.Env):
    def __init__(self, render_mode=None):
        self.m0 = 1.0     # 旋转臂等效质量 (kg)
        self.m1 = 0.5     # 摆杆 1 质量 (kg)
        self.m2 = 0.25    # 摆杆 2 质量 (kg)
        self.l1 = 0.5     # 摆杆 1 质心长度 (m)
        self.l2 = 0.5     # 摆杆 2 质心长度 (m)
        self.g = 9.81     # 重力加速度 (m/s^2)

        self.b0 = 0.1     # 旋转臂摩擦系数
        self.b1 = 0.05    # 关节 1 摩擦系数
        self.b2 = 0.05    # 关节 2 摩擦系数

        self.max_force = 20.0  # 电机最大输出力矩 (N)
        self.dt = 0.02         # 控制周期 20ms (50Hz)
```

### 动作空间与状态空间

**动作空间** (连续):
```python
self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
```
- 一维连续动作, 范围 [-1.0, 1.0], 乘以 `max_force` 得到实际力矩

**状态空间** (8 维连续):
```python
# [x, dx, cos(th1), sin(th1), dth1, cos(th2), sin(th2), dth2]
high = np.array([4.0, inf, 1.0, 1.0, inf, 1.0, 1.0, inf], dtype=np.float32)
self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
```

| 维度 | 变量 | 含义 | 范围 |
|------|------|------|------|
| 0 | x | 旋转臂等效线位移 | [-4.0, 4.0] |
| 1 | dx | 旋转臂线速度 | (-inf, inf) |
| 2 | cos(th1) | 摆杆1角度余弦 | [-1.0, 1.0] |
| 3 | sin(th1) | 摆杆1角度正弦 | [-1.0, 1.0] |
| 4 | dth1 | 摆杆1角速度 | (-inf, inf) |
| 5 | cos(th2) | 摆杆2角度余弦 | [-1.0, 1.0] |
| 6 | sin(th2) | 摆杆2角度正弦 | [-1.0, 1.0] |
| 7 | dth2 | 摆杆2角速度 | (-inf, inf) |

使用三角函数编码角度以避免角度周期性边界问题。

### 动力学模型 (拉格朗日方程)

系统动力学通过拉格朗日方法建模, 求解 `M(q) * q_ddot = B(q, q_dot, tau)`:

```python
def _compute_derivatives(self, state, action):
    x, th1, th2, dx, dth1, dth2 = state
    tau = np.clip(action[0], -1.0, 1.0) * self.max_force

    # 质量惯性矩阵 M(q) (3x3)
    M11 = self.m0 + self.m1 + self.m2
    M12 = (self.m1 + self.m2) * self.l1 * np.cos(th1)
    M13 = self.m2 * self.l2 * np.cos(th2)
    M22 = (self.m1 + self.m2) * self.l1**2
    M23 = self.m2 * self.l1 * self.l2 * np.cos(th1 - th2)
    M33 = self.m2 * self.l2**2
    # ... 对称填充

    # 广义力向量 B (3x1)
    B1 = tau + ... - self.b0 * dx
    B2 = ... + (self.m1+self.m2)*self.g*self.l1*np.sin(th1) - self.b1*dth1
    B3 = ... + self.m2*self.g*self.l2*np.sin(th2) - self.b2*dth2

    q_ddot = np.linalg.solve(M, B)
    return np.array([dx, dth1, dth2, q_ddot[0], q_ddot[1], q_ddot[2]])
```

### RK4 积分更新

```python
def step(self, action):
    k1 = self._compute_derivatives(self.state, action)
    k2 = self._compute_derivatives(self.state + 0.5*self.dt*k1, action)
    k3 = self._compute_derivatives(self.state + 0.5*self.dt*k2, action)
    k4 = self._compute_derivatives(self.state + self.dt*k3, action)

    self.state = self.state + (self.dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
```

角度规范化到 [-pi, pi]:
```python
th1 = ((th1 + np.pi) % (2 * np.pi)) - np.pi
th2 = ((th2 + np.pi) % (2 * np.pi)) - np.pi
```

### 终止条件

```python
terminated = bool(
    x < -self.x_threshold or x > self.x_threshold       # |x| > 2.0
    or th1 < -self.theta_threshold or th1 > self.theta_threshold  # |th1| > 0.5 rad
    or th2 < -self.theta_threshold or th2 > self.theta_threshold  # |th2| > 0.5 rad
)
```

超出安全边界 (~28 度) 即判定为失败。

### 奖励函数

```python
if not terminated:
    reward = 10.0                     # 存活奖励
             - 15.0 * (th1**2)        # 摆杆1 角度偏离惩罚
             - 15.0 * (th2**2)        # 摆杆2 角度偏离惩罚
             - 1.0 * (x**2)           # 旋转臂偏离中心惩罚
             - 0.1 * (dx**2)          # 速度惩罚
             - 0.1 * (dth1**2)        # 角速度惩罚
             - 0.1 * (dth2**2)        # 角速度惩罚
             - 0.05 * (action[0]**2)  # 控制能耗惩罚
else:
    reward = -100.0  # 失败惩罚
```

奖励函数引导策略: (1) 保持两个摆杆垂直向上 (th1=0, th2=0), (2) 旋转臂居中 (x=0), (3) 速度尽可能小, (4) 控制动作尽可能小 (节能)。

### 环境重置

```python
def reset(self, seed=None, options=None):
    self.state = np.random.uniform(low=-0.05, high=0.05, size=(6,))
    return self._get_obs(), {}
```

初始化在直立位置附近 (小扰动 [-0.05, 0.05]), 实现稳定控制任务的课程学习。将扰动范围扩大到 [-pi, pi] 可实现完全起摆任务。

## PPO 训练配置

```python
policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))

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
    tensorboard_log="./ppo_rotary_pendulum_tb/")
```

### 超参数总表

| 参数 | 值 | 说明 |
|------|-----|------|
| **PPO 算法** | | |
| `learning_rate` | 3e-4 | Adam 学习率 |
| `n_steps` | 2048 | 每轮每环境采样步数 |
| `batch_size` | 512 | 训练批次大小 |
| `n_epochs` | 10 | 每轮 SGD 更新次数 |
| `gamma` | 0.99 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE lambda 参数 |
| `clip_range` | 0.2 | PPO 裁剪范围 |
| `ent_coef` | 0.0 | 熵正则系数 (关闭) |
| **网络架构** | | |
| Actor 网络 | [128, 128] | 2 层 MLP, 每层 128 神经元 |
| Critic 网络 | [128, 128] | 2 层 MLP, 每层 128 神经元 |
| **训练配置** | | |
| `total_timesteps` | 500,000 | 总训练时间步 |
| `eval_freq` | 10,000 | 评估间隔 |
| `device` | CUDA (if available) | 自动选择 GPU/CPU |

### 自定义进度条回调

```python
class TqdmCallback(BaseCallback):
    def _on_step(self):
        self.pbar.set_postfix({
            "轮次": self.episodes,
            "平均奖励": f"{mean_reward:.2f}",
            "平均步长": f"{mean_length:.1f}"
        })
```

实时显示: 已完成的轮次数、最近 100 轮平均奖励、平均步长。

### 模型保存与评估

```python
eval_callback = EvalCallback(eval_env, best_model_save_path='./models/',
    log_path='./models/', eval_freq=10000, deterministic=True)

model.learn(total_timesteps=500000, callback=[eval_callback, tqdm_callback])
model.save("./models/ppo_rotary_final")
```

- 每 10000 步评估一次最佳模型
- 最终模型和最佳模型分别保存为 `ppo_rotary_final.zip` 和 `best_model.zip`

## 测试与可视化

```python
def test_model():
    env = RotaryDoublePendulumEnv(render_mode="human")
    model = PPO.load(model_path, env=env)

    for i in range(2000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
```

测试时使用 `deterministic=True` (网络输出均值, 不含高斯噪声), PyGame 实时渲染:

```
旋转臂 (蓝色粗线) -> 关节1 (红色圆点)
  -> 摆杆1 (橙色线) -> 关节2 (绿色圆点)
    -> 摆杆2 (橙色线) -> 末端 (蓝色圆点)
```

渲染帧率: 50 FPS (与 20ms 控制周期对齐)。

## 关键文件

```
ppo-rotary-pendulum-control/
├── ppo_rotary_pendulum.py   # 全部代码: 环境定义, PPO 训练, 评估回调, 测试, PyGame 渲染
└── README.md                # 本文件
```

## 运行方式

```bash
# 安装依赖
pip install gymnasium stable-baselines3 pygame torch numpy tqdm

# 训练模式 (生成模型)
# 在 ppo_rotary_pendulum.py 中设置 MODE = "train"
python ppo_rotary_pendulum.py

# 测试模式 (加载模型并可视化)
# 在 ppo_rotary_pendulum.py 中设置 MODE = "test"
python ppo_rotary_pendulum.py
```

## 知识点

- **物理建模**: 使用拉格朗日力学推导二阶倒立摆动力学方程, 3x3 质量惯性矩阵求解
- **数值积分**: RK4 (四阶龙格-库塔) 方法, 20ms 控制周期
- **自定义 Gymnasium 环境**: 继承 `gym.Env`, 实现 `step`, `reset`, `render`, `_get_obs`
- **连续控制 PPO**: Stable-Baselines3 的 PPO 实现在连续动作空间上的应用
- **奖励函数设计**: 角度偏离 + 位移 + 速度 + 能耗的加权惩罚, 以及存活奖励和失败惩罚
- **PyGame 可视化**: 将高维状态映射为 2D 投影渲染, 支持实时交互
- **训练流程**: EvalCallback 定期评估 + 自定义 TqdmCallback 进度监控
