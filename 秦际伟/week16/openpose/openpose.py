import torch 
import torchvision.transforms as transforms
import cv2
import numpy as np 


# 加载预训练的OpenPose模型
module = torch.hub.load('CMU-Visual-Computing-Lab/openpose', 'pose_resnet50', pretrained=True)
module.eval()


# 图像预处理
def preprocess_image(image):
    # 将图像转换为PyTorch张量
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    # 添加一个额外的维度，以匹配模型的输入形状
    image = image.unsqueeze(0)
    return image


# 读取图像
image = cv2.imread('image.jpg')
image_tensor = preprocess_image(image)

# 运行OpenPose模型
with torch.no_grad():
    output = module(image_tensor)

# 解析输出
heatmaps = output[0].cpu().numpy()
keypoints = np.argmax(heatmaps, axis=0)
for i in range(keypoints.shape[0]):
    y, x = np.unravel_index(np.argmax(heatmaps[i]), heatmaps[i].shape)
    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

# 显示结果
cv2.imshow('OpenPose', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

