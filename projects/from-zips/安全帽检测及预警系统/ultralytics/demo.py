from ultralytics import YOLO
import torch
# 加载模
model = YOLO('D://010//master//代做//ultralytics//ultralytics//cfg//models//v8//yolov8.yaml').load('D://010//master//代做//ultralytics//yolov8n.pt')
print("文件加载成功！")
# 开始训练模型
model.train(data=r'D:\010\master\代做\ultralytics//data\v5\data.yaml', workers=0, device='cuda', amp=False)
# 进行预测

model.predict(source='test3.mp4', device='cuda', save_txt=True, show=True, save=True)
