# 安全帽检测及预警系统

## 技术方向

计算机视觉、安全生产监测、YOLO 目标检测、违规预警。

## 技术栈

- Python
- YOLOv8 / Ultralytics
- PyTorch
- OpenCV
- Roboflow/YOLO 数据集格式
- DeepSORT 目标跟踪

## 工作链路

1. 整理安全帽、人员、工服等目标检测数据集。
2. 使用 `data.yaml` 配置类别和训练、验证、测试划分。
3. 基于 YOLOv8 训练安全装备检测模型。
4. 对图片或视频进行推理，识别 `Hardhat`、`NO-Hardhat`、`Person`、`Suit` 等类别。
5. 根据检测类别触发未佩戴安全帽、未穿工服等违规预警。

## 关键内容

- `configs/datasets/*.yaml`：多版本安全帽数据集配置。
- `ultralytics/app.py`, `ultralytics/demo.py`：检测系统入口和演示脚本。
- `ultralytics/track/predict.py`, `ultralytics/track/demo.py`：检测 + 跟踪链路。
- `ultralytics/track/deep_sort_pytorch/`：DeepSORT 跟踪相关代码。
- `ultralytics/results/`, `ultralytics/alerts/`：少量检测结果和告警示例图。

## 整理说明

- 已移除完整 Ultralytics 框架副本、训练集图片、运行输出和权重文件。
- 当前目录保留业务入口、数据配置、跟踪模块和少量结果图。
- 权重文件不入库，后续可用外部下载链接或 Release 管理。
