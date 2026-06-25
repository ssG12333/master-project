# AS 路由网络分析

## 技术方向

复杂网络分析、互联网 AS 关系建模、图计算、网络可视化。

## 技术栈

- Python
- NetworkX
- PyTorch
- scikit-learn
- pandas, NumPy
- Matplotlib
- 社区发现算法

## 工作链路

1. 使用 `crawl_peers.py` 爬取或整理 AS Peering 数据。
2. 将 AS 关系构建为图结构。
3. 计算度分布、中心性、社群结构、横向/纵向关系等指标。
4. 使用可视化脚本输出网络图和统计图。
5. 将分析结果用于路由行为研究和论文说明。

## 关键内容

- `crawl_peers.py`：数据抓取。
- `第一种.py`, `第二种.py`：分析方法脚本。
- `as_visualizations_*`：网络分析图和统计结果。

## 后续整理

- 合并两个版本的分析输出。
- 增加数据来源、字段说明和复现实验命令。

