"""
演示
    梯度下降法的优化方法

介绍：
    梯度下降法，由损失函数的导数/梯度、学习率 来更新模型参数
    w新 = w旧 - 学习率 * 梯度
    存在问题：
        1.碰到平缓区域，梯度较小，参数更新慢
        2.碰到 “鞍点” ，梯度为 0，参数无法更新
        3.碰到局部最小值，无法跳出，参数不是最优
梯度下降优化方法：
    1.调整梯度，比如使用指数移动加权平均来替代原始梯度
        动量法Momentum
            梯度的指数移动加权平均：st = β*st-1 + (1-β)*gt
            更新参数：wt = wt-1 - η*st
    2.调整学习率
        AdaGrad自适应学习率(Adaptive Gradient Estimation):
            累计平方梯度：
                St = St-1 + gt*gt
                其中，St为本次的累计平方梯度，St-1为上次的累计平方梯度，gt为本次的梯度
            学习率：
                η = η/(sqrt(St)+σ)
            更新参数公式：
                Wt = Wt-1 - 调整后的η * gt
            缺点：
                可能会使得学习率过早、过量的降低，导致模型训练后期学习率太小，较难找到最优解
        RMSProp(Root Mean Square Propagation)自适应学习率：
            对AdaGrad的优化，在计算累计平方梯度时，加入 调和权重系数
            指数加权累计平方梯度：
                St = β*St-1 + (1-β)*gt*gt
            其中St为本次的累计平方梯度，St-1为上次的累计平方梯度，gt为本次的梯度，β为 调和权重系数
            学习率：
                η = η/(sqrt(St)+σ)
            更新参数公式：
                Wt = Wt-1 - 调整后的η * gt
            优点：通过引入 调和权重系数， 控制历史梯度 对 累计平方梯度的影响，防止累计平方梯度过大，从而防止了学习率过早降低
        Adam(Adaptive Moment Estimation)自适应矩估计:
            同时优化 学习率 和 梯度
            公式：
                梯度(一阶矩)：
                    Mt = β1 * Mt-1 +(1-β1)*gt
                    Mt_hat = Mt / (1-β1^t)
                学习率(二阶矩):
                    St = β2 * St-1 + (1-β2)*gt*gt
                    St_hat = St / (1-β2^t)
                更新参数公式：
                    Wt = Wt-1 - 学习率/(sqrt(St_hat)+σ) * 调整后的梯度Mt_hat
            就是 RMSProp + Momentum
        AdamW:
            Adam的优化，解偶了权重衰减，在更新参数时直接添加了权重衰减项。
            公式:
                梯度(使用一阶矩)：
                    Mt = β1 * Mt-1 +(1-β1)*gt
                    Mt_hat = Mt / (1-β1^t)
                学习率(使用二阶矩):
                    St = β2 * St-1 + (1-β2)*gt*gt
                    St_hat = St / (1-β2^t)
                更新参数公式：
                    Wt = Wt-1 - 学习率/(sqrt(St_hat)+σ) * 调整后的梯度Mt_hat - η * λ * Wt-1

梯度下降法的工作流：
    1.初始化权重参数 torch.tensor(w=[1.0],requires_grad=True,dtype=torch.float32)
    2.定义损失函数 nn.mse_loss/(w**2)/2
    3.创建优化器对象 tensor.optim.SGD/AdaGrad/RMSProp/AdamW
    4.梯度清零 optimizer.zero_grad()
    5.反向传播,计算梯度 loss.sum().backward()
    6.更新权重参数 optimizer.step()
需要掌握：
    Adam,AdamW
    Momentum

"""
# 导包
import torch
import torch.nn as nn
import torch.optim as optim  # 优化器模块，实现梯度下降法以及梯度下降的优化方法
from sympy.physics.vector.printing import params


# 定义函数，演示梯度下降优化方法 ——>动量法（momentum）
def dm01_momentum():
    print("动量法Momentum")
    # 1.初始化权重函数
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    # 2.定义损失函数
    # loss_fn = (w**2)/2
    # 3.创建优化器对象
    optimizer = optim.SGD(params=[w], lr=0.1, momentum=0.9)
    # # 4.梯度清零
    # optimizer.zero_grad()
    # # 5.反向传播，计算梯度
    # loss_fn.sum().backward()
    # # 6.更新权重
    # optimizer.step()
    # print(f'第一次权重更新：loss:{loss_fn.sum().item()},w:{w.data},grad:{w.grad.data}')

    # 迭代30次
    for i in range(10):
        loss = ((w ** 2) * 0.5).sum()
        # 4.梯度清零
        optimizer.zero_grad()
        # 5.反向传播，计算梯度
        loss.mean().backward()
        # 6.更新参数，第一次更新参数
        optimizer.step()
        # if w.grad.data < 0:
        #     break
        print(f"第{i}次更新参数：loss:{loss},w:{w.data}, w.grad:{w.grad.data}")


def dm02_Adagrad():
    print("AdaGrad自适应学习率")
    # 1.初始化模型参数w
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    # 2.定义损失函数
    # loss_fn = (w ** 2) / 2
    # 3.创建优化器对象
    optimizer = optim.Adagrad(params=[w], lr=0.1)
    # 迭代30次
    for i in range(30):
        loss = ((w ** 2) * 0.5).sum()
        # 4.梯度清零
        optimizer.zero_grad()
        # 5.反向传播，计算梯度
        loss.mean().backward()
        # 6.更新参数，第一次更新参数
        optimizer.step()
        print(f"第{i}次更新参数：loss:{loss},w:{w.data}, w.grad:{w.grad.data}")

def dm03_RMSProp():
    print("RMSProp自适应学习率")
    # 1.初始化模型参数w
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    # 2.定义损失函数
    # loss_fn = (w ** 2) / 2
    # 3.创建优化器对象
    optimzer = optim.RMSprop(params=[w], lr=0.1, alpha=0.9, eps=1e-6)
    # 迭代30次
    for i in range(30):
        loss = ((w ** 2) * 0.5).sum()
        # 4.梯度清零
        optimzer.zero_grad()
        # 5.反向传播，计算梯度
        loss.sum().backward()
        # 6.更新参数
        optimzer.step()
        print(f"第{i}次更新参数：loss:{loss},w:{w.data}, w.grad:{w.grad.data}")

def dm04_Adam():
    print("Adam自适应矩估计")
    # 1.初始化模型参数w
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    # 2.定义损失函数
    # loss_fn = (w ** 2) / 2
    # 3.创建优化器对象 beatas=(梯度的衰减系数,学习率的衰减系数),eps=1e-6(小常数)
    optimizer = optim.Adam(params=[w], lr=0.1, betas=(0.9, 0.999), eps=1e-6)
    # 迭代30次
    for i in range(30):
        loss = ((w ** 2) * 0.5).sum()
        # 4.梯度清零
        optimizer.zero_grad()
        # 5.反向传播，计算梯度
        loss.sum().backward()
        # 6.更新参数
        optimizer.step()
        print(f"第{i}次更新参数：loss:{loss},w:{w.data}, w.grad:{w.grad.data}")

if __name__ == '__main__':
    dm01_momentum()
    dm02_Adagrad()
    dm03_RMSProp()
    dm04_Adam()