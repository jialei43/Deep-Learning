"""
案例：
    通过代码演示学习率 对 梯度（梯度优化）的影响

结论：
    学习率越小，梯度下降越慢
    学习率越大，梯度下降越快，可能会越过最小值，造成震荡，甚至不收敛，造成梯度爆炸
"""
import torch


def dm01():
    print("演示学习率对梯度（梯度优化）的影响")
    x = torch.tensor([1.0], requires_grad=True)
    # 记录迭代次数，画曲线
    iter_rec,loss_rec,x_rec=list(),list(),list()

    # 实验学习率：0.01,0.1,0.5,1.0,2.0,5.0,10.0,20.0,50.0,100.0
    lr = 0.01
    # lr = 0.1 #正常梯度下降
    # lr =0.125 # 学习率设置为0.125，一下子求出最优解
                # x = 0 y = 0 在x=0处梯度等于0，x = x -lr*x.grad 就不更新了
                # 后续再多少次迭代，都固定在最优点
    # lr =0.2   # x从2.0一下子跨过0点，到了左侧的负数区域
    # lr =0.3   #梯度越大越爆炸
    max_iter = 4
    for i in range(max_iter):
        y = func(x)
        # 1.计算损失函数
        y = x ** 2
        # 2.记录迭代次数
        iter_rec.append(i)
        # 3.记录损失函数
        loss_rec.append(y.item())
        # 4.记录参数
        x_rec.append(x.item())
        # 5.梯度清零
        x.grad.data.zero_()
        # 6.反向传播
        y.backward()
        # 7.更新参数 