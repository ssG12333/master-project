from ultralytics import YOLO
import torch
# 加载模
model = YOLO('D://010//master//代做//山体滑坡检测//ultralytics//cfg//models//v8//yolov8.yaml').load('D://010//master//代做//山体滑坡检测//best.pt')
print("文件加载成功！")
# 开始训练模型
model.train(data=r'D:\010\master\代做\山体滑坡检测//data\v1\data.yaml', workers=0, device='cuda', amp=False)
# 进行预测


