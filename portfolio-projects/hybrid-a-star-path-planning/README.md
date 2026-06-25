# Hybrid A* 泊车路径规划仿真

## 项目简介

本项目实现面向车辆泊车场景的 Hybrid A* 路径规划，结合 A* 启发式、车辆运动学约束、Reeds-Shepp 曲线和多场景可视化评估。项目适合展示传统路径规划、车辆运动模型和仿真分析能力。

原始目录：`改进a星路径规划/`

## 技术栈

- Python, MATLAB
- Hybrid A*, A*
- Reeds-Shepp 曲线
- NumPy, matplotlib
- CSV 指标统计与路径可视化

## 主要功能

- 基于车辆位姿状态进行 Hybrid A* 搜索。
- 引入 Reeds-Shepp 曲线满足车辆最小转弯半径约束。
- 支持多种障碍物场景、起终点和车辆尺寸配置。
- 输出路径、搜索树、障碍物距离、性能对比表等结果。
- 保留 MATLAB 版本作为算法对照实现。

## 工作链路

1. 构建栅格地图、障碍物和车辆运动约束。
2. 使用 A* 距离作为启发式估计。
3. 按车辆运动模型扩展候选轨迹。
4. 使用 Reeds-Shepp 曲线连接局部路径并计算代价。
5. 搜索得到可行泊车路径后输出轨迹和指标。
6. 对多个场景生成路径图、搜索树和性能对比表。

## 知识点

- Hybrid A* 搜索框架。
- 车辆非完整约束和最小转弯半径。
- Reeds-Shepp 路径族。
- 启发式函数设计和碰撞检测。
- 路径规划结果可视化与性能评估。

## 关键文件

- `hybrid_astar_main.py`：主算法、场景运行和结果分析。
- `astar_search.py`：A* 启发式距离计算。
- `motion_model.py`：车辆运动扩展。
- `reeds_shepp.py`：Reeds-Shepp 曲线实现。
- `visualization.py`：路径和车辆轮廓可视化。
- `matlab_hybrid_astar/`：MATLAB 对照实现。
- `scenario_*_vehicle_path.png`、`performance_data.csv`：示例结果。

## 整理说明

- 已移除论文、视频、缓存和调试日志。
- 当前目录保留 Python/MATLAB 核心算法和少量结果图。
- 批量中间帧和演示视频不入库。
