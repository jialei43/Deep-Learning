import torch
import matplotlib.pyplot as plt

import matplotlib

matplotlib.use('TkAgg')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False


def sigmoid():
    # 激活函数
    x = torch.linspace(-10, 10, 100)
    y = torch.sigmoid(x)

    plt.plot(x.numpy(), y.numpy())
    plt.grid()
    plt.show()

    # 激活函数的导数
    x = torch.linspace(-10, 10, 100, requires_grad=True)
    y = torch.sigmoid(x).sum().backward()

    # plt.plot(x.detach().numpy(), x.grad.numpy())
    # plt.grid()
    # plt.show()


def tanh():
    # plt.subplots(1, 2)
    # 激活函数
    x = torch.linspace(-10, 10, 100)
    y = torch.tanh(x)

    # plt.plot(x.numpy(), y.numpy())
    # plt.grid()
    # plt.show()

    # 激活函数的导数
    x = torch.linspace(-10, 10, 100, requires_grad=True)
    y = torch.tanh(x).sum().backward()

    plt.plot(x.detach().numpy(), x.grad.numpy())
    plt.grid()
    plt.show()


def Relu():
    # 1.创建画布
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # 2.生成x值，-10到1哦之间，总数量100
    x = torch.linspace(-10, 20, 100, dtype=torch.float)
    # 3.计算y值，relu激活函数
    y = torch.relu(x)
    # 4.绘制函数图像在第一个子图像
    axes[0].plot(x, y)
    axes[0].set_title('relu激活函数')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].grid()
    # 绘制导函数图像，在第二个子图上
    x = torch.linspace(-10, 20, 100, requires_grad=True, dtype=torch.float)
    y = torch.relu(x)
    y.sum().backward()  # 反向传播获取导数，x.grad
    axes[1].plot(x.detach(), x.grad)
    axes[1].set_title('relu激活函数导数')
    axes[1].grid()
    plt.show()


def softmax():
    score = torch.tensor([0.2,0.02,0.15,0.15,1.3,0.5,0.06,1.1,0.05,3.75])
    print(torch.softmax(score, dim=0))


if __name__ == '__main__':
    sigmoid()
    # tanh()
    # Relu()
    # softmax()