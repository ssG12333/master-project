# 山体滑坡检测

## 技术方向

遥感/自然灾害识别、YOLO 目标检测、地质灾害监测。

## 技术栈

- Python
- YOLOv8 / Ultralytics
- PyTorch
- OpenCV
- Roboflow YOLO 数据集格式

## 工作链路

1. 整理山体滑坡图像数据和 YOLO 标注。
2. 使用 `data/v1/data.yaml` 配置训练集、验证集、测试集和类别。
3. 基于 YOLOv8 训练滑坡检测模型。
4. 对图像进行推理，输出滑坡区域检测框。
5. 使用验证指标和可视化结果评估模型效果。

## 数据类别

- `Landslides`

## 关键内容

- `configs/data.yaml`：数据集配置。
- `requirements.txt`：依赖说明。
- `demo.py`, `测试/测试.py`：检测演示和测试脚本。
- `test.jpg`, `测试/`：少量样例图片。

## 整理说明

- 已移除完整 Ultralytics 框架副本、训练数据和批量输出图。
- 当前目录保留数据配置、依赖说明、演示脚本和少量样例图。
- 权重和完整数据集不入库，后续可通过 Release 或外部链接补充。
