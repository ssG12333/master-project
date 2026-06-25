# YOLO 肺结核与胸片疾病检测系统

## 项目简介

本项目基于 YOLOv8 构建胸片疾病检测系统，并封装为 PyQt5 桌面应用。系统可识别肺结核、肺炎、健康等类别，并在图像上绘制检测结果，辅助完成医学影像目标检测实验展示。

原始目录：`肺结核yolo/`

## 技术栈

- Python
- YOLOv8 / Ultralytics
- PyQt5
- OpenCV
- PIL
- Roboflow YOLO 格式数据集

## 主要功能

- 登录窗口和检测主界面。
- 读取胸片图像并调用 YOLO 模型推理。
- 显示疾病类别、置信度和检测框。
- 使用中文标签映射展示结果。
- 支持结果历史记录和图像可视化。

## 工作链路

1. 使用胸片 YOLO 数据集训练疾病检测模型。
2. PyQt5 界面加载待检测胸片。
3. OpenCV/PIL 完成图像读取、缩放和中文标签绘制。
4. YOLOv8 输出疾病类别、置信度和检测框。
5. 界面展示检测结果、类别中文映射和置信度。

## 数据与类别

已识别到两套胸片相关数据配置：

- `Pneumonia Bacteria`, `Pneumonia Virus`, `Sick`, `healthy`, `tuberculosis`
- `Consolidation`, `Effusion`, `Pneumonia`, `Tuberculosis`

## 知识点

- 医学影像目标检测任务建模。
- YOLOv8 模型加载和推理。
- PyQt5 图像显示和交互界面。
- 检测标签中文映射。
- OpenCV 与 PIL 混合绘制中文文本。

## 后续清理

- 去除数据集和权重文件，保留数据配置样例。
- 增加模型训练命令和推理示例。
- 补充医学场景说明和指标截图。
