"""
演示：
    常用的激活函数
    w新 = w旧 - 学习率*梯度

    梯度消失：（Vanishing Gradient）是指在神经网络训练过程中，梯度在反向传播过程中逐渐变得非常小，接近零，
    导致权重更新的速度变得非常慢，甚至无法有效更新。这种现象通常发生在深层神经网络中，
    尤其是在使用某些激活函数时（如 Sigmoid 或 Tanh）。

sigmoid
    用于二分类任务的输出层，通常用于浅层网络，不超过5层，容易出现梯度消失，目前使用较少
    输出范围(0,1),导数范围(0,0.25]


ReLu
    常用于隐藏层，目前使用最多
    计算公式max(0,x),输出范围[0,+正无穷),导数{0,1}
    优势：
        1.计算简单，模型收敛快
        2.中输入的导数恒为1，缓解梯度消失问题
    DeadRuLU问题：
        随着训练的推进，部分输入会落入到负区域，导致对应权重无法更新。被称作"死亡神经元"，实际很少发生

tanh
    用于隐藏层，浅层网络，同样存在梯度消失问题,很少使用，主流网络中只有RNN网络(LSTM,GRU)中使用了tanh激活
    输出范围是(-1,1),导数范围是(0,1]

softmax
    用于单标签多分类任务的输出层，二分类任务上可以替代sigmoid
    输出各个类别的概率分布，范围(0,1),并且概率总和为1

激活函数的选择方法
    对于隐藏层:	ReLU -> Leaky ReLU/PReLU -> (Tanh)

对于输出层:
    多分类问题选择 softmax(优先)
    二分类问题选择 sigmoid 或 softmax
    回归问题多数选择 identity (就是没有激活函数)，正数ReLU，区间Sigmoid/Tanh


细节：
    绘制激活函数图像时出现以下提示，需要将 anaconda3/Lib/site-packages/torch/lib目录下的libiomp5md.dll文件删除
    OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.

"""
# 导包
import torch

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False

# 定义函数，演示sigmoid激活函数的函数图像和导函数图像
def demo1():
    # 1.创建画布
    fig,axes = plt.subplots(1,2,figsize=(12,5))
    # 2.生成x值，-10到1哦之间，总数量100
    x = torch.linspace(-10,10,100,dtype=torch.float)
    # 3.计算y值，sigmoid激活函数
    y = torch.sigmoid(x)
    # 4.绘制函数图像在第一个子图像
    axes[0].plot(x,y)
    axes[0].set_title('sigmoid激活函数')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].grid()
    # 绘制导函数图像，在第二个子图上
    x = torch.linspace(-10, 10, 100, requires_grad=True,dtype=torch.float)
    y = torch.sigmoid(x)
    y.sum().backward() # 反向传播获取导数，x.grad
    axes[1].plot(x.detach(),x.grad)
    axes[1].set_title('sigmoid激活函数导数')
    axes[1].grid()
    plt.show()
    
# 定义函数，演示relu激活函数的函数图像和导函数图像
def demo2():
    # 1.创建画布
    fig,axes = plt.subplots(1,2,figsize=(12,5))
    # 2.生成x值，-10到1哦之间，总数量100
    x = torch.linspace(-10,10,100,dtype=torch.float)
    # 3.计算y值，relu激活函数
    y = torch.relu(x)
    # 4.绘制函数图像在第一个子图像
    axes[0].plot(x,y)
    axes[0].set_title('relu激活函数')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].grid()
    # 绘制导函数图像，在第二个子图上
    x = torch.linspace(-10, 10, 100, requires_grad=True,dtype=torch.float)
    y = torch.relu(x)
    y.sum().backward() # 反向传播获取导数，x.grad
    axes[1].plot(x.detach(),x.grad)
    axes[1].set_title('relu激活函数导数')
    axes[1].grid()
    plt.show()


# 定义函数，演示tanh激活函数的函数图像和导函数图像
def demo3():
    # 1.创建画布
    fig,axes = plt.subplots(1,2,figsize=(12,5))
    # 2.生成x值，-10到1哦之间，总数量100
    x = torch.linspace(-10,10,100,dtype=torch.float)
    # 3.计算y值，tanh激活函数
    y = torch.tanh(x)
    # 4.绘制函数图像在第一个子图像
    axes[0].plot(x,y)
    axes[0].set_title('tanh激活函数')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].grid()
    # 绘制导函数图像，在第二个子图上
    x = torch.linspace(-10, 10, 100, requires_grad=True,dtype=torch.float)
    y = torch.tanh(x)
    y.sum().backward() # 反向传播获取导数，x.grad
    axes[1].plot(x.detach(),x.grad)
    axes[1].set_title('tanh激活函数导数')
    axes[1].grid()
    plt.show()

# 定义函数，演示softmax激活函数的函数图像和导函数图像
def demo4():
    # 1.创建画布
    fig,axes = plt.subplots(1,2,figsize=(12,5))
    # 2.定义x值，模拟模型的预测分数logits
    x = torch.tensor([0.2,-0.9,10,-10,1,2,3,4,5,6])
    # 3.计算y值，softmax激活函数
    # dim = -1：通常用于多分类问题，沿着最后一个维度进行Softmax操作，每个样本的类别得分将归一化，确保每个样本的分类得分和为1。假设 x 是一个二维张量，形状为 (batch_size, num_classes)，表示每个样本的分类得分。dim=-1 会沿着每一行（即每个样本）应用。
    # dim = 0：沿着第一个维度进行Softmax操作，通常用于一些特定的场景，比如对整个批次的某个维度进行归一化。表示沿着第一个维度（即批量维度）应用 Softmax。在这种情况下，Softmax 会在所有样本（即批量中的所有数据）之间进行归一化，但这种应用较少见，通常只有在处理特定场景时才会使用。
    # 其他维度：对于高维张量，dim可以指定任意维度，允许我们在任何维度上应用，Softmax操作。
    y = torch.softmax(x,dim=-1) # 通常特征维度/轴为-1
    # 4.绘制函数图像在第一个子图像
    axes[0].bar(range(len(y)),y)
    axes[0].set_title('softmax激活函数')
    axes[0].grid()
    # 5.绘制softmax(softmax)函数图像，在第二个子图上
    z = torch.softmax(y, dim=-1)
    axes[1].bar(range(len(z)), z)
    axes[1].set_title('softmax(softmax)激活函数')
    axes[1].grid()
    plt.show()

if __name__ == '__main__':
    # demo1()
    # demo2()
    # demo3()
    demo4()