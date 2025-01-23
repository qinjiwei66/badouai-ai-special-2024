from unet_parts import *

class UNet(nn.Module):
    # 定义U-Net模型结构，接受输入通道数n_channels和输出类别数n_classes，以及是否使用双线性插值bilinear参数
    # 初始化了U-Net模型的各个组件，包括卷积层、下采样层、上采样层和输出层
    # 其中，DoubleConv、Down、Up和OutConv分别是U-Net模型的四个组件，用于构建编码器和解码器部分的卷积层和池化层
    # 最后，通过OutConv将输出通道数转换为n_classes的输出
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    # 定义U-Net模型的前向传播过程，接受输入x，经过一系列卷积层、池化层和上采样层，最终输出预测的概率值logits
    # 定义了模型的前向传播过程，即输入数据通过各个层的处理，最终得到输出结果
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits



if __name__ == '__main__':
    net = UNet(n_channels=3, n_classes=2)
    print(net)