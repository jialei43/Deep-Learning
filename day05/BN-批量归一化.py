"""
演示
    批量归一化，属于正则化方法，缓解 模型过拟合，主要用于CV计算机视觉，比如CNN网络中

批量归一化：
    先标准化为均值0，方差1的分布，然后再进行重构(缩放+平移，缩放系数和平移系数都是可学习的)，进一步提取特征

涉及到的API：
    CV计算机视觉领域
    BatchNorm1d: 处理2D/3D数据，主要用于处理文本数据/序列数据，NCS
    BatchNorm2d: 处理4D数据，主要用于二维卷积神经网络Conv2d来处理图片数据，NCHW,(N,C,H,W), N批次大小，C通道数，H高度，W宽度
    BatchNorm3d: 处理5D数据，主要用于三维卷积神经网络Conv3d,来处理高维的视觉数据，比如医学图像，卫星遥感视频，接收形状为 NCDHW/NCTHW 的张量作为输入。

需要掌握的：
    BatchNorm2d

"""

# 导包
import torch
import torch.nn as nn

# 定义函数，演示 BatchNorm2d 批量归一化, 处理4D数据，NCHW
def demo01():
    # 1.创建数据，模拟NCHW图片数据 torch.randint(0,256,size=(2,3,32,32))
    x = torch.randint(0,256,size=(1,3,32,32)).to(dtype=torch.float32)
    print(f"原始数据x:{x}, shape: {x.shape}")
    # 2.创建BatchNorm2d层
    # running_mean = (1 - momentum) * running_mean + momentum * batch_mean
    # running_var = (1 - momentum) * running_var + momentum * batch_var
    bn = nn.BatchNorm2d(
        num_features=3, # 通道数，C
        eps=1e-05,  # 极小值，防止除零
        momentum=0.1,   # 动量，控制参数的更新
        affine=True,    # 缩放参数和平移参数是否可学习
    )
    # 3.输入数据到BatchNorm2d层中
    y = bn(x)   # (2,3,32,32)
    # 4.打印输出
    print(f"处理后的数据y:{y}, shape: {y.shape}")
    ...

# 定义函数，演示 BatchNorm1d 批量归一化, 处理2D/3D数据，NCS
def demo02():
    # 1.创建数据，模拟序列数据
    x = torch.randint(0,256,size=(1,3,10)).to(dtype=torch.float32)
    print(f"原始数据x:{x}, shape: {x.shape}")
    # 2.创建BatchNorm1d层
    bn = nn.BatchNorm1d(
        num_features=3,
        eps=1e-05,  # 极小值，防止除零
        momentum=0.1,   # 动量，控制参数的更新
        affine=True,    # 缩放参数和平移参数是否可学习
    )
    # 3.输入数据到BatchNorm2d层中
    y = bn(x)   # (1,3,10)
    # 4.打印输出
    print(f"处理后的数据y:{y}, shape: {y.shape}")
    ...

# 测试
if __name__ == '__main__':
    demo01()
    demo02()