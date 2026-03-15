"""
演示
    学习率衰减策略/学习率调度策略
介绍：
    之前学的优化器，可以调节学习率：Adam/AdamW, AdaGrad, RMSProp,
    Momentum(调节梯度)
    也可以手动调节学习率：
        等间隔学习率衰减
        指定间隔学习率衰减
        指数间隔学习率衰减
        周期重启的余弦退火策略
等间隔学习率衰减
    固定间隔来衰减学习率，当前学习率lr = 上一次的学习率lr*γ, 比如γ=0.5

指定间隔学习率衰减
    人为指定间隔来衰减学习率，当前学习率lr = 上一次的学习率lr*γ, 比如γ=0.5

指数间隔学习率衰减
    当前轮次的lr = 上个轮次lr*γ, 比如γ=0.95

周期重启的余弦退火策略
    带有周期重启的余弦曲线，可以让学习率重新启动，有可能冲过局部最小值

神经网络的训练流程：
    1.准备数据集
    2.构建神经网络模型
    3.设置损失函数和优化器，以及学习率调度器
    4.模型训练
        1.前向传播
        2.计算损失
        3.梯度清零
        4.反向传播
        5.更新参数:w新 = w旧-学习率*梯度
        6.更新学习率
    5.模型测试

需要掌握：
    无

"""


# 导包
import torch
import torch.nn as nn
import torch.optim as optim # 优化器模块，提供各种优化器对象，比如SGD,Adam
import matplotlib.pyplot as plt # 绘图
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False

# 1.定义函数，演示 等间隔学习率衰减策略
def demo01():
    # 0.初始化参数
    lr = 0.1
    epochs = 200
    iteration = 10
    # 1.准备数据集
    x = torch.tensor([1.0],dtype=torch.float32)
    y_true = torch.tensor([0.0],dtype=torch.float32)
    # 2.构建神经网络模型
    # 创建张量，模拟网络参数
    w = torch.tensor([1.0],dtype=torch.float32,requires_grad=True)
    # 3.设置损失函数和优化器，以及学习率调度器
    # 损失函数
    loss_fn = nn.MSELoss()
    # 优化器
    optimizer = optim.SGD([w],lr=lr)
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.5)
    # 4.模型训练
    # 定义列表，记录训练轮数和学习率
    lr_list = []
    epoch_list = []
    for epoch in range(epochs):
        # 0.获取当前轮数 和 学习率，保存到记录列表中
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr()[0])
        # 遍历批次
        for i in range(iteration):
            # 1.前向传播
            y_pred = w * x
            # 2.计算损失
            # 3.梯度清零
            # 4.反向传播
            # 5.更新参数:w新 = w旧-学习率*梯度
        # 6.更新学习率
        scheduler.step()
    # 5.可视化学习率变化
    plt.plot(epoch_list,lr_list)
    plt.title("等间隔学习率衰减")
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.show()

# 2.定义函数，演示 指定间隔学习率衰减策略
def demo02():
    # 0.初始化参数
    lr = 0.1
    epochs = 200
    iteration = 10
    # 1.准备数据集
    x = torch.tensor([1.0],dtype=torch.float32)
    y_true = torch.tensor([0.0],dtype=torch.float32)
    # 2.构建神经网络模型
    # 创建张量，模拟网络参数
    w = torch.tensor([1.0],dtype=torch.float32,requires_grad=True)
    # 3.设置损失函数和优化器，以及学习率调度器
    # 损失函数
    loss_fn = nn.MSELoss()
    # 优化器
    optimizer = optim.SGD([w],lr=lr)
    # 学习率调度器
    # scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.5)
    # 指定间隔学习率衰减策略
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[50,125,160],gamma=0.5)
    # 4.模型训练
    # 定义列表，记录训练轮数和学习率
    lr_list = []
    epoch_list = []
    for epoch in range(epochs):
        # 0.获取当前轮数 和 学习率，保存到记录列表中
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr()[0])
        # 遍历批次
        for i in range(iteration):
            # 1.前向传播
            y_pred = w * x
            # 2.计算损失
            # 3.梯度清零
            # 4.反向传播
            # 5.更新参数:w新 = w旧-学习率*梯度
        # 6.更新学习率
        scheduler.step()
    # 5.可视化学习率变化
    plt.plot(epoch_list,lr_list)
    plt.title("指定间隔学习率衰减")
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.show()

# 3.定义函数，演示 指数间隔学习率衰减策略
def demo03():
    # 0.初始化参数
    lr = 0.1
    epochs = 200
    iteration = 10
    # 1.准备数据集
    x = torch.tensor([1.0],dtype=torch.float32)
    y_true = torch.tensor([0.0],dtype=torch.float32)
    # 2.构建神经网络模型
    # 创建张量，模拟网络参数
    w = torch.tensor([1.0],dtype=torch.float32,requires_grad=True)
    # 3.设置损失函数和优化器，以及学习率调度器
    # 损失函数
    loss_fn = nn.MSELoss()
    # 优化器
    optimizer = optim.SGD([w],lr=lr)
    # 学习率调度器
    # scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.5)
    # 指定间隔学习率衰减策略
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[50,125,160],gamma=0.5)
    # 指数间隔学习率衰减策略
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.95)
    # 4.模型训练
    # 定义列表，记录训练轮数和学习率
    lr_list = []
    epoch_list = []
    for epoch in range(epochs):
        # 0.获取当前轮数 和 学习率，保存到记录列表中
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr()[0])
        # 遍历批次
        for i in range(iteration):
            # 1.前向传播
            y_pred = w * x
            # 2.计算损失
            # 3.梯度清零
            # 4.反向传播
            # 5.更新参数:w新 = w旧-学习率*梯度
        # 6.更新学习率
        scheduler.step()
    # 5.可视化学习率变化
    plt.plot(epoch_list,lr_list)
    plt.title("指数间隔学习率衰减")
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.show()

# 4.定义函数，演示 周期重启的余弦退火策略
def demo04():
    # 0.初始化参数
    lr = 0.1
    epochs = 200
    iteration = 10
    # 1.准备数据集
    x = torch.tensor([1.0],dtype=torch.float32)
    y_true = torch.tensor([0.0],dtype=torch.float32)
    # 2.构建神经网络模型
    # 创建张量，模拟网络参数
    w = torch.tensor([1.0],dtype=torch.float32,requires_grad=True)
    # 3.设置损失函数和优化器，以及学习率调度器
    # 损失函数
    loss_fn = nn.MSELoss()
    # 优化器
    optimizer = optim.SGD([w],lr=lr)
    # 学习率调度器
    # scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.5)
    # 指定间隔学习率衰减策略
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[50,125,160],gamma=0.5)
    # 指数间隔学习率衰减策略
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.95)
    # 周期重启的余弦退火策略
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,  # 优化器对象
        T_0=50, # 第一个周期的轮数
        eta_min=0
    )
    # 4.模型训练
    # 定义列表，记录训练轮数和学习率
    lr_list = []
    epoch_list = []
    for epoch in range(epochs):
        # 0.获取当前轮数 和 学习率，保存到记录列表中
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr()[0])
        # 遍历批次
        for i in range(iteration):
            # 1.前向传播
            y_pred = w * x
            # 2.计算损失
            # 3.梯度清零
            # 4.反向传播
            # 5.更新参数:w新 = w旧-学习率*梯度
        # 6.更新学习率
        scheduler.step()
    # 5.可视化学习率变化
    plt.plot(epoch_list,lr_list)
    plt.title("周期重启的余弦退火策略")
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.show()

# 测试
if __name__ == '__main__':
    demo01()
    demo02()
    demo03()
    demo04()