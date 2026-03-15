"""
案例：
    演示 学习率 的大小，对模型训练过程中的 梯度下降的路径 的影响

结论：
    学习率 太小， 梯度下降更新很慢
    学习率 太大， 梯度下降更新很快，可能会 梯度震荡，甚至 梯度爆炸
    通常，学习率设为1e-2~1e-5

"""


import torch
import matplotlib.pyplot as plt

def func(x_t):
    return torch.pow(2*x_t, 2)  # y = 4 x ^2

# 采用较小的学习率，梯度下降的速度慢
# 采用较大的学习率，梯度下降太快越过了最小值点，导致不收敛，甚至震荡
def demo01():
    # 定义不同的学习率进行实验
    learning_rates = [0.01, 0.1, 0.125, 0.2, 0.3]
    max_iteration = 4

    # 创建图像和子图
    fig, axes = plt.subplots(2, len(learning_rates), figsize=(5 * len(learning_rates), 8))

    for idx, lr in enumerate(learning_rates):
        # 重置参数
        x = torch.tensor([2.], requires_grad=True)
        iter_rec, loss_rec, x_rec = list(), list(), list()

        # 梯度下降过程
        for i in range(max_iteration):
            y = func(x)
            y.backward()

            x_rec.append(x.item())
            x.data.sub_(lr * x.grad)
            x.grad.zero_()

            iter_rec.append(i)
            loss_rec.append(y.item())

        # 绘制损失曲线
        axes[0, idx].plot(iter_rec, loss_rec, '-ro')
        axes[0, idx].set_title(f'LR={lr}\nLoss Curve')
        axes[0, idx].set_xlabel("Iteration")
        axes[0, idx].set_ylabel("Loss")
        axes[0, idx].grid(True)

        # 绘制函数曲线和下降轨迹
        x_t = torch.linspace(-3, 3, 100)
        y_t = func(x_t)
        axes[1, idx].plot(x_t.detach().numpy(), y_t.detach().numpy(), 'b-', alpha=0.7, label="y = 4*x²")

        y_rec = [func(torch.tensor(i)).item() for i in x_rec]
        axes[1, idx].plot(x_rec, y_rec, '-ro', label="Gradient Descent")
        axes[1, idx].set_title(f'LR={lr}\nOptimization Path')
        axes[1, idx].set_xlabel("x")
        axes[1, idx].set_ylabel("y")
        axes[1, idx].grid(True)
        axes[1, idx].legend()

    plt.tight_layout()
    plt.show()
# 测试
if __name__ == '__main__':
    demo01()