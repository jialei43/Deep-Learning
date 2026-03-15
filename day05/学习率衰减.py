import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False
# 定义函数，演示 等间隔学习率衰减策略
def demo01():
    # 参数初始化
    lr = 0.1
    epochs = 200
    iteration = 10

    # 数据准备
    x = torch.tensor([1.0], dtype=torch.float32)
    y_true = torch.tensor([0.0], dtype=torch.float32)
    w = torch.tensor([1.0], dtype=torch.float32, requires_grad=True)
    # 损失函数
    loss_fn = nn.MSELoss()
    # 优化器
    optimizer = optim.SGD([w], lr=lr)
    # 学习率调度器
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)


    # 模型训练
    # 定义列表，记录训练轮数和学习率
    lr_list = []
    epoch_list = []
    for epoch in range(epochs):
        # 0.获取当前轮数 和 学习率，保存到记录列表中
        epoch_list.append(epoch)
        lr_list.append(lr_scheduler.get_last_lr()[0])
        for i in range(iteration):
            # 前向传播
            y_pred = w * x
            # 计算损失
            loss = loss_fn(y_pred, y_true)
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数:w新 = w旧-学习率*梯度
            optimizer.step()
        # 更新学习率
        lr_scheduler.step()

    # 可视化学习率变化
    plt.plot(epoch_list, lr_list)
    plt.title("等间隔学习率衰减")
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.show()
# 指定间隔学习率衰减策略
def demo02():
    # 参数初始化
    lr = 0.1
    epochs = 200
    iteration = 10

    # 数据准备
    x = torch.tensor([1.0], dtype=torch.float32)
    y_true = torch.tensor([0.0], dtype=torch.float32)
    w = torch.tensor([1.0], dtype=torch.float32, requires_grad=True)
    # 损失函数
    loss_fn = nn.MSELoss()
    # 优化器
    optimizer = optim.SGD([w], lr=lr)
    # 学习率调度器
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 180], gamma=0.5)

    # 模型训练
    # 定义列表，记录训练轮数和学习率
    lr_list = []
    epoch_list = []
    for epoch in range(epochs):
        # 0.获取当前轮数 和 学习率，保存到记录列表中
        epoch_list.append(epoch)
        lr_list.append(lr_scheduler.get_last_lr()[0])
        for i in range(iteration):
            # 前向传播
            y_pred = w * x
            # 计算损失
            loss = loss_fn(y_pred, y_true)
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数:w新 = w旧-学习率*梯度
            optimizer.step()
        # 更新学习率
        lr_scheduler.step()

    # 可视化学习率变化
    plt.plot(epoch_list, lr_list)
    plt.title("指定间隔学习率衰减")
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.show()

# 指数间隔学习率衰减策略
def demo03():
    # 参数初始化
    lr = 0.1
    epochs = 200
    iteration = 10

    # 数据准备
    x = torch.tensor([1.0], dtype=torch.float32)
    y_true = torch.tensor([0.0], dtype=torch.float32)
    w = torch.tensor([1.0], dtype=torch.float32, requires_grad=True)
    # 损失函数
    loss_fn = nn.MSELoss()
    # 优化器
    optimizer = optim.SGD([w], lr=lr)
    # 学习率调度器
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer,  gamma=0.99)

    # 模型训练
    # 定义列表，记录训练轮数和学习率
    lr_list = []
    epoch_list = []
    for epoch in range(epochs):
        # 0.获取当前轮数 和 学习率，保存到记录列表中
        epoch_list.append(epoch)
        lr_list.append(lr_scheduler.get_last_lr()[0])
        for i in range(iteration):
            # 前向传播
            y_pred = w * x
            # 计算损失
            loss = loss_fn(y_pred, y_true)
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数:w新 = w旧-学习率*梯度
            optimizer.step()
        # 更新学习率
        lr_scheduler.step()

    # 可视化学习率变化
    plt.plot(epoch_list, lr_list)
    plt.title("指数学习率衰减")
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.show()

if __name__ == '__main__':
    # demo01()
    # demo02()
    demo03()