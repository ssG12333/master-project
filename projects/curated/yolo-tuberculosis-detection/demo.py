from ultralytics import YOLO
import torch
# 加载模
model = YOLO('D://010//master//代做//肺结核//ultralytics//cfg//models//v8//yolov8l_fusion_transformerx3_hsi.yaml')
print("文件加载成功！")
# 开始训练模型
model.train(data=r'D:\010\master\代做\肺结核//data\v1\data.yaml', workers=0, device='cuda', amp=False)
# 进行预测

