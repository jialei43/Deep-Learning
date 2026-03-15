"""
演示
    指数移动加权平均, 使用近30天的天气温度数据.
结论
    1.对于指数移动加权平均值，β值（调节系数）越大，移动加权平均值越平缓
    2.特例：β=0，就是原始值；β=1，就是第一天的数据直线
    3.距离当前点越远，对指数移动加权平均的影响越小

"""

# 导包
import torch
import matplotlib.pyplot as plt

import matplotlib
# matplotlib.use('TkAgg')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False

ELEMENT_NUMBER = 30

# 1. 实际平均温度
def demo01():
    # 固定随机数种子
    torch.manual_seed(0)
    # 产生30天的随机温度
    temperature = torch.randn(size=[ELEMENT_NUMBER, ]) * 10
    print(temperature)
    # 绘制平均温度
    days = torch.arange(1, ELEMENT_NUMBER + 1, 1)
    plt.plot(days, temperature, color='r')
    plt.scatter(days, temperature)
    plt.tight_layout()
    plt.show()

# 2. 指数加权平均温度
def demo02(beta=0.9):
    # 固定随机数种子，确保每次运行得到相同的随机温度数据
    torch.manual_seed(24)
    # 产生30天的随机温度数据，通过乘以10放大温度范围
    temperature = torch.randn(size=[ELEMENT_NUMBER, ]) * 10
    print(temperature)

    # 存储指数加权平均后的温度值
    exp_weight_avg = []
    # 遍历每天的温度数据，idx从0开始计数
    for idx, temp in enumerate(temperature):
        # 第一天的指数加权平均值就是当天温度本身
        if idx == 0:
            exp_weight_avg.append(temp)
            continue
        # 从第二天开始，按照指数加权平均公式计算：
        # V_t = β * V_{t-1} + (1-β) * θ_t
        # 其中 V_t 是当前时刻的加权平均值，V_{t-1} 是上一时刻的加权平均值
        # θ_t 是当前时刻的实际温度值，β 是权重参数(0<=β<1)
        # idx-1 是因为要访问前一天的加权平均值
        new_temp = exp_weight_avg[idx - 1] * beta + (1 - beta) * temp
        exp_weight_avg.append(new_temp)

    # 创建天数序列用于绘图
    days = torch.arange(1, ELEMENT_NUMBER + 1, 1)
    # 设置图表标题，显示当前使用的beta值
    plt.title(f"beta={beta}")
    # 绘制指数加权平均温度曲线（红色）
    plt.plot(days, exp_weight_avg, color='r')
    # 绘制原始温度数据点
    plt.scatter(days, temperature)
    plt.tight_layout()
    # 显示图表
    plt.show()

if __name__ == '__main__':
    demo01()
    demo02(beta=0)
    demo02(beta=0.5)
    demo02(beta=0.9)
    demo02(beta=1)