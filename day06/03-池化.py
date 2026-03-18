import matplotlib.pyplot as plt
import torch
from torch import nn



# 构建三维输入张量并转换为浮点型
inputs = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                       [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
                       [[11, 22, 33], [44, 55, 66], [77, 88, 99]]]).float()
print(inputs.shape)
# 创建最大池化层，配置卷积核大小、步幅等参数
pool = nn.MaxPool2d(kernel_size=(2, 2), stride=1, padding=0, dilation=1, return_indices=False, ceil_mode=False)
# 执行最大池化操作
max_out = pool(inputs)
print(max_out)

# 创建平均池化层，配置卷积核大小和步幅
avg_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=1, padding=0, ceil_mode=False)
# 执行平均池化操作
avg_out = avg_pool(inputs)
print(avg_out)
