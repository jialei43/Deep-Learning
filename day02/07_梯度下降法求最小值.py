"""
演示
    梯度下降发，求损失函数的最小值
    循环实现，计算梯度，更新参数w
    #求y=x**2+20的极小值点并打印y是最小值时w的值(梯度)
    #1定义点 x=10 requires_grad=True dtype=torch.float32
    #2定义函数y=x**2+20
    #3利用梯度下降法循环迭代1000次求最小值
        #3-1前向传播
        #3-2梯度清零x.grad.zero_(
        #3-3 反向传播
        # 3-4更新参数x.data =x.data - 0.01*x.grad
"""

# 导包
import torch

#1定义点 x=10 requires_grad=True dtype=torch.float32
w = torch.tensor([10.0,20.0],requires_grad=True,dtype=torch.float32)
#2定义函数y=x**2+20
loss = w**2 + 20
#3利用梯度下降法循环迭代1000次求最小值
print(f'初始值：w:{w.data}, w.grad:{w.grad},loss_mean:{loss.mean()}')
for i in range(1000):
    #3-1前向传播
    loss = w**2 + 20
    #3-2梯度清零x.grad.zero_()
    if w.grad is not None:
        w.grad.zero_()
    # w.grad.zero_()
    #3-3 反向传播
    loss.mean().backward()
    # 3-4更新参数x.data =x.data - 0.01*x.grad
    w.data = w.data - 0.01*w.grad
    print(f'第{i+1}次迭代：w:{w.data},w.grad:{w.grad.data}, loss_mean:{loss.mean()}')

