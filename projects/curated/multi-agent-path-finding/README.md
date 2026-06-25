# 多智能体路径规划仿真系统 (MAPF)

本科毕业设计 — 基于 Conflict-Based Search (CBS) 和 A* 的多 AGV 物流分拣路径规划可视化仿真系统。

**在线 Demo**: https://mapf-simulation-system.pages.dev/

## 代码架构

### CBS 冲突搜索 (`CBS.js`, `CBS_v2.js`)

```javascript
class CBS {
    constructor(env) {
        this.openSet = [];    // CT Node 优先队列
        this.closedSet = [];
    }

    search() {
        // 1. 初始化根 CT Node (无约束)
        // 2. 对所有 agent 用 A* 计算初始路径
        // 3. while openSet not empty:
        //    a. 取最小 cost 的 CT Node P
        //    b. 检测冲突 (顶点冲突 / 边冲突)
        //    c. 无冲突 → 返回路径方案
        //    d. 有冲突 → 分裂为两个子 CT Node:
        //       - 子节点 A: agent_i 禁止在时间 t 占据顶点 v
        //       - 子节点 B: agent_j 禁止在时间 t 占据顶点 v
        //    e. 对受影响的 agent 重新规划 (A* 带约束)
    }
}
```

**V2 优化 (CBS_v2.js)**:
- Cardinal Conflict 优先选择
- BP (Bypass) 机制: 假分裂 → 若新路径减少冲突则替换原路径，不分裂
- 全局冲突表缓存
- 转弯代价惩罚

### A* 搜索 (`AStar.js`, `AStar_v2.js`)

```javascript
class AStar {
    // 带约束的 A* 路径搜索
    // 约束类型:
    //   - 顶点约束: agent 不能在时间 t 占据 (x,y)
    //   - 边约束: agent 不能从 (x1,y1) 移动到 (x2,y2) 在时间 t
    //
    // 启发式: 曼哈顿距离
    // V2 优化: dirs 方向优化、优先选择 h 值小的路径
}
```

### 环境与智能体 (`Environment.js`, `Agent.js`)

```javascript
class Environment {
    // 栅格地图管理
    // agent_dict: {id → Agent}
    // constraint_dict: {id → Constraints}
    // obstacles: Set of (x,y)

    calcSolution(checkConflict) // 所有 agent 并行 A*
    getFirstConflict(solution)  // 检测最早冲突
    calcSolutionCost(solution)  // 总路径长度 + 转弯代价
}

class Agent {
    // start, end, path[], taskList[]
    // velocity, color, waitCount, turnCount
}
```

### Python 地图处理 (`map_processor.py`)

```python
# 从 YAML/JSON 读取预定义地图 → 转换为仿真系统格式
# 支持批量地图生成与导出
```

### 可视化 (`sketch.js`)

```javascript
// p5.js 渲染:
//   - 栅格地图 (可配置行/列/障碍物比例)
//   - Agent 运动动画 (速度可调)
//   - 路径颜色编码
//   - 单步执行/连续运行模式
//   - 统计面板: 等待次数、转弯次数
//   - 地图保存/加载 (JSON 格式)
```

## 实验结果

| 地图规模 | 障碍物比例 | Agent 数量 | 测试组数 |
|---------|:---:|:---:|:---:|
| 8×8 | 1% | 2-7 | 每组 5 组 |
| 20×20 | 1% | 4-20 | 每组 3 组 |
| 20×20 | 10% | 4-12 | 每组 4 组 |
| 50×50 | 1% | 10-30 | 每组 3 组 |

## 技术栈

| 类别 | 技术 |
|------|------|
| 路径规划 | CBS (Conflict-Based Search), A* |
| 多智能体 | MAPF (Multi-Agent Path Finding), 约束传播 |
| 可视化 | p5.js, JavaScript, HTML5 Canvas |
| 地图处理 | Python (map_processor.py) |
| 优化 | Cardinal Conflict, BP, 全局冲突表 |

## 运行方式

```bash
# Web 版: 直接在浏览器打开 index.html
# 或使用 p5.js IDE 运行

# 在线体验: https://mapf-simulation-system.pages.dev/
```

## 参考文献

- Sharon et al. "Conflict-based search for optimal multi-agent pathfinding." AIJ, 2015.
- Boyarski et al. "ICBS: The improved conflict-based search algorithm for multi-agent pathfinding." SoCS, 2015.

## 原始目录

`D:\010\master\code\MultiAgentPathFinding\`
