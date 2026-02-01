"""
演示
    自动微分模块 如何计算梯度/导数
介绍
    梯度：损失函数对模型参数的导数/梯度，针对某个确定的模型参数，损失函数的导函数的数值
        模型参数就是权重w，偏置b
    损失函数对模型参数的梯度不需要手动计算，是loss.backward反向传播过程中pytorch调用自动微分模块来自动计算的

过程：
    1.定义模型权重参数
    2.定义一个损失函数
    3.backward()反向传播计算导数
    4.w.grad获取到数值/梯度
"""

import torch

# 1.定义变量 初始权重
w = torch.tensor([10,20], requires_grad=True, dtype=torch.float32)
print(f'w:{w},requiers_grad:{w.requires_grad}')

# 2.定义损失函数
loss = w**2 + 10
# loss = 2*w = [2*w1,2*w2]
# loss_mean = 2*w = [w1,w2]
print(f'loss_mean:{loss.mean()}')
# 3.反向传播计算导数,调用自动微分模块 loss必须是一个标量数值，所以使用mean让它成为一个标量
loss.mean().backward()
print(f'w.grad:{w.grad}')

w.data = w.data - 0.01 * w.grad
print(f'w:{w}')
