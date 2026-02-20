"""
演示
    自动微分的真实应用场景，
    也就是使用pytorch框架来实现损失函数MSE，以及反向传播loss.backward()
工作流：
    1.前向传播：把输入数据输入模型得到预测值
    2.计算损失：把真实值和预测值传入损失函数得到损失值
    3.梯度清零：pytorch默认梯度累加，清零上次梯度后再获取当前梯度
    4.反向传播：loss.backward()过程中自动调用自动微分模块来计算梯度（损失函数对模型参数的导数）
    5.更新参数：更新模型的参数，包括 权重w 和 偏置b，公式：w新 = w旧 - 学习率*梯度

"""
import torch

# 设置随机种子
torch.manual_seed(42)

# 定义x 表示：特征，输入数据
x = torch.ones(2,5)
print(f'x:{x},shape:{x.shape},dtype:{x.dtype}')

# 定义y，表示真实值

y = torch.zeros(2,1)
print(f'y:{y},shape:{y.shape}')

# 3.初始化 权重为w 和 偏置 b，x@w + b ，（2，1）
w = torch.randn(5,1,requires_grad=True,dtype=torch.float32)
print(f'w:{w}')
b = torch.randn(1,requires_grad=True,dtype=torch.float32)
print(f'b:{b}')

# 4.前向传播，计算预测值 z
z = x @ w + b
print(f'z:{z},shape:{z.shape}')

# 5 .计算损失值
loss_fn = torch.nn.MSELoss()
loss = loss_fn(z,y)
print(f'loss:{loss}')

# 6.自动微分，执行反向传播，获取梯度：loss对w,b的梯度，注意损失函数MSELoss内部会自动求平均值，不需要loss.mean()
loss.backward()

# 7.打印loss对w，b的梯度/导数
print(f'grad:{w.grad}')
print(f'b:{b.grad}')

# 后续就是：梯度下降法更新模型参数，w新 = w旧 - 学习率 * 梯度

