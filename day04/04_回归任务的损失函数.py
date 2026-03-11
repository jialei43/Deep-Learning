"""
演示 回归任务的 损失函数
MAE:
    Mean Absolute Error, 平均绝对误差
    误差绝对值的平均数
    类似于L1正则化，权重可以降到0，具有稀疏性
    缺点：
        在0点不平滑，可能错过最小值
MSE: Mean Squared Error, 均方误差
    误差平方之和 / 样本总数
    缺点:
        如果差值过大，可能导致梯度爆炸，无法有效训练模型
Smooth L1:
    结合了MAE和MSE, MSE:(-1,1), MAE:(-∞,1]U[1,+∞)
    解决L1(MAE)在0点不平滑、以及L2(MSE)的梯度爆炸问题

如何选择回归任务的损失函数
    MSE > Smooth L1 > MAE
需要掌握：
    MSE,nn.MSELoss

"""

# 导包
import numpy as np
import torch
from torch import nn

# 1.定义函数，演示 MAE
def demo01():
    # 1.创建 样本的真实值
    # 4个样本
    y_true = torch.tensor([20.0,21.0,22.0,23.0],dtype=torch.float32)
    # 2.创建 模型的预测值
    y_pred = torch.tensor([23.0,22.0,23.0,24.0],dtype=torch.float32,requires_grad=True)
    # 3.定义损失函数对象
    loss_fn = nn.L1Loss()
    # 4.计算损失值
    loss = loss_fn(y_pred,y_true)
    # 5.打印损失值
    print(f"损失值：{loss}")    # 1

# 2.定义函数，演示 MSE
def demo02():
    # 1.创建 样本的真实值
    # 4个样本
    y_true = torch.tensor([20.0,21.0,22.0,23.0],dtype=torch.float32)
    # 2.创建 模型的预测值
    y_pred = torch.tensor([23.0,22.0,23.0,24.0],dtype=torch.float32,requires_grad=True)
    # 3.定义损失函数对象
    loss_fn = nn.MSELoss()
    # 4.计算损失值
    loss = loss_fn(y_pred,y_true)
    # 5.打印损失值
    print(f"损失值：{loss}")    # 1

# 3.定义函数，演示 Smooth L1 Loss
def demo03():
    # 1.创建 样本的真实值
    # 4个样本
    y_true = torch.tensor([20.0,21.0,22.0,23.0],dtype=torch.float32)
    # 2.创建 模型的预测值
    y_pred = torch.tensor([23.0,22.0,23.0,24.0],dtype=torch.float32,requires_grad=True)
    # 3.定义损失函数对象
    loss_fn = nn.SmoothL1Loss()
    # 4.计算损失值
    loss = loss_fn(y_pred,y_true)
    # 5.打印损失值
    print(f"损失值：{loss}")    # 1

# 测试
if __name__ == "__main__":
    demo01()
    demo02()
    demo03()