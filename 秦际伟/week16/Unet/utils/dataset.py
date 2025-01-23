import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random


class ISBI_Loader(Dataset):
    # 初始化数据集的路径和图像文件的路径列表
    def __init__(self, data_path):
        # 初始化函数，接受data_path路径下的参数
        self.data_path = data_path
        # 初始化一个空列表，用于存储数据路径
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))


    # 用于对图像进行随机翻转的数据增强操作
    def augment(self, image, flipCode):
        # 使用cv2.flip函数进行图像翻转，flipCode为1表示水平翻转，0表示垂直翻转，-1表示水平垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    
    # 首先根据索引获取图像和标签的路径
    # 然后读取图像和标签，并进行灰度转换和形状调整
    # 接着，根据随机选择的 flipCode 进行数据增强操作
    # 最后返回增强后的图像和标签。
    def __getitem__(self, index):
        # 根据索引index获取图像和标签的路径
        image_path = self.imgs_path[index]
        label_path = image_path.replace('image', 'label')
        # 使用cv2.imread函数读取图像和标签，以灰度模式读取
        image = cv2.imread(image_path)  
        label = cv2.imread(label_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])

        if label.max() > 1:
            label = label / 255

        # 随机选择是否进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])

        if flipCode!= 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        return image, label


    # 返回图像路径列表的长度，即数据集的大小
    def __len__(self):
        # 返回图像路径列表的长度，即数据集的大小
        return len(self.imgs_path)


# 创建了 ISBI_Loader 类的实例，并打印出数据集的大小。
# 然后，使用 torch.utils.data.DataLoader 创建了一个数据加载器，
# 并遍历数据加载器，打印出每个批次的图像形状
if __name__ == "__main__":
    ISBI_dataset = ISBI_Loader("data/train/")
    print("数据个数：", len(ISBI_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=ISBI_dataset,
                                               batch_size=2,
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)