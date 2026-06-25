# 羽毛球动作识别

## 技术方向

运动动作识别、YOLO 目标检测/姿态识别、体育智能分析。

## 技术栈

- Python
- YOLOv8 / Ultralytics
- PyTorch
- OpenCV
- Roboflow YOLO 数据集格式

## 工作链路

1. 整理羽毛球动作类别数据集。
2. 使用 `data.yaml` 定义训练、验证、测试集和动作类别。
3. 基于 YOLO 训练动作识别或姿态检测模型。
4. 对图片/视频进行推理，识别发球、防守、挑球、杀球等动作。
5. 输出类别、置信度和可视化结果。

## 数据类别

版本一包含：

- `BackhandClear`, `BackhandLift`, `BackhandServe`
- `ForehandClear`, `ForehandLift`
- `ReadyPosition`, `Smash`

版本二包含：

- `backhand-general`, `defense`, `lift`, `offense`, `serve`, `smash`

## 关键内容

- `data/V1/data.yaml`, `data/V2/data.yaml`
- `requirements.txt`
- `ultralytics/`

## 后续整理

- 明确最终采用的数据版本。
- 补充动作识别效果图和模型指标。

