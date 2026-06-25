from ultralytics import YOLO
import torch
# 加载模
model = YOLO('D://010//master//代做//羽毛球识别//ultralytics//cfg//models//v8//yolov8.yaml').load('D://010//master//代做//羽毛球识别//yolov8n.pt')
print("文件加载成功！")
# 开始训练模型
model.train(data=r'D:\010\master\代做\羽毛球识别//data\V2\data.yaml', workers=0, device='cuda', amp=False)
# 进行预测

