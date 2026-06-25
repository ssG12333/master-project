# Hybrid A* 泊车路径规划仿真

## 项目简介

融合 Hybrid A* 搜索与 Reeds-Shepp 曲线的车辆运动学路径规划。支持 3 种泊车场景（超车、U 型调头、直角转弯），包含 Python + MATLAB 双版本对比仿真。

## 代码架构

### 三场景定义 (`hybrid_astar_main.py`)

```python
SCENARIOS = {
    1: {  # 超车 Overtaking — 3×10
        "start_pose": [0.5, 0.5, 0], "goal_pose": [9.5, 0.5, 0],
        "min_r": 0.5,
    },
    2: {  # U 型调头 U-turn — 10×5, 障碍墙分隔
        "start_pose": [0.5, 0.5, 0], "goal_pose": [4.5, 0.5, 0],
        "min_r": 0.8,
    },
    3: {  # 直角转弯 Turning Corner — 10×10, L型障碍
        "start_pose": [0.5, 9.5, 0], "goal_pose": [9.5, 0.5, 0],
        "min_r": 1.0,
    },
}

def mod2pi(x):
    """角度归一化 [-π, π]"""
```

### Hybrid A* 搜索 (`motion_model.py`)

```python
def find_route(start_pose, goal_pose, obstacles, min_r):
    """Hybrid A* 主搜索
    状态: (x, y, θ) 连续位姿
    动作: 前进/后退 × 左转/直行/右转 (6 个离散动作)
    g-cost:  路径长度 + 方向切换惩罚
    h-cost:  max(astar_2d_distance, reeds_shepp_length)
    碰撞检测: 车辆矩形 (length×width) vs 障碍物栅格
    终止条件: 位姿误差 < ε 且 Reeds-Shepp 无碰撞可达
    """
```

### A* 启发式 (`astar_search.py`)

```python
def astar_distance(start, goal, obstacles, grid_size):
    """2D A* 最短距离 (8 方向连通)
    用作 Hybrid A* 的 admissible h-cost 下界估计
    """
```

### Reeds-Shepp 曲线 (`reeds_shepp.py`)

```python
def reeds_shepp(start, goal, min_radius):
    """48 种路径类型全搜索
    CSC:  Circular-Straight-Circular (8×)
    CCC:  Circular-Circular-Circular (4×)
    CCCC: 4-segment (8×)
    CCSC: Circular-Circular-Straight-Circular (16×)
    CCSCC: 5-segment (12×)
    返回: 最短路径的曲线段序列 + 总长度
    """
```

### MATLAB 对比 (`matlab_hybrid_astar/`)

```matlab
% Astar_fun.m:       A* 搜索实现
% CCC.m, CCCC.m, CCSC.m, CCSCC.m, CSC.m: 各路径类型
% FindRSPath.m:      Reeds-Shepp 最优路径查找
% find_route_fun.m:  Hybrid A* 主函数
```

## 技术栈

| 类别 | 技术 |
|------|------|
| 路径搜索 | Hybrid A* (连续位姿 + 车辆运动学约束) |
| 最短曲线 | Reeds-Shepp (48 种类型, 允许倒车) |
| 启发式 | max(A* 2D 距离, Reeds-Shepp 无碰撞长度) |
| 碰撞检测 | 车辆矩形 AABB vs 栅格障碍物 |
| Python | numpy, matplotlib |
| MATLAB | 独立 .m 文件对比验证 |

## 运行方式

```bash
pip install numpy matplotlib
python hybrid_astar_main.py    # 3 场景自动运行 + 路径可视化

# MATLAB: 打开 matlab_hybrid_astar/ 文件夹运行
```

## 关键文件

| 文件 | 说明 |
|------|------|
| `hybrid_astar_main.py` | 主程序: 场景定义 + 可视化渲染 |
| `astar_search.py` | A* 2D 启发式距离 |
| `reeds_shepp.py` | Reeds-Shepp 曲线 (48 种路径) |
| `motion_model.py` | 车辆运动模型 + Hybrid A* 搜索 |
| `matlab_hybrid_astar/` | MATLAB 对比仿真 (9 .m 文件) |
