import matplotlib.pyplot as plt
import torch
from torch import nn

# 创建二维卷积层，配置输入输出通道数、卷积核大小等参数
conv_layer = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=(3, 3), stride=1, padding=0, dilation=1, groups=1, bias=True)

# 读取图像并转换为张量
img = plt.imread('img/img.png')

# 将图像数据转换为浮点型张量并进行维度变换：HWC 格式转为 CHW 格式，并添加批次维度
img = torch.tensor(img, dtype=torch.float32)
print(f'img:{img}, img.shape:{img.shape}')
img = img.permute(2, 0, 1).unsqueeze(0)
print('-'*50)
print(f'img:{img}, img.shape:{img.shape}')

# 执行卷积操作生成特征图
feature_map = conv_layer(img)
print(f' feature_map.shape:{feature_map.shape}')

# 创建第二个卷积层，增加输出通道数并设置填充
# padding(3,3)
conv_layer2 = nn.Conv2d(in_channels=2, out_channels=200, kernel_size=(3, 3), stride=1, padding=(3, 3), dilation=1, groups=1, bias=True)

# 对第一个卷积的输出再次进行卷积
feature_map2 = conv_layer2(feature_map)
print(f' feature_map2.shape:{feature_map2.shape}')
