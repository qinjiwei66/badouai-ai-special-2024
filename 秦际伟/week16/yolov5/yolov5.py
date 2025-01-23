import torch
from PIL import Image


# 加载预训练的YOLOv5模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 切换到推理模式
model.eval()

# 定义要检测的图像路径
image_path = 'path/to/image.jpg'

# 加载图像并进行预测
image = Image.open(image_path)
results = model(image)

# 输出检测结果
print(results.pandas().xyxy[0])

# 显示图像和检测结果
results.show()