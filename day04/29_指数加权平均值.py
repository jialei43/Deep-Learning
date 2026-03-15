import torch
import matplotlib.pyplot as plt

import matplotlib
# matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False

#设置随机种子
torch.random.manual_seed(24)
temperture = torch.randn(30)*10
ELEMENT_NUMBER = 30
days = torch.arange(1, ELEMENT_NUMBER + 1, 1)
# 1.创建画布
fig,axes = plt.subplots(1,5,figsize=(48,20))
# plt.legend(ncol=2,  # 设置为2列，即每行显示2个
#           fontsize=10,
#           loc='upper right',
#           frameon=True,
#           fancybox=True,
#           shadow=True)
axes[4].plot(days,temperture)
axes[4].set_title('原始温度')
axes[4].set_xlabel('x')
axes[4].set_ylabel('y')
axes[4].grid()
plt.scatter(days,temperture)

# 2.创建指数加权平均对象

# 存储权重初始化值
beta = [0.9,0.95,0.99,0.999]
#存储所有权重对应的指数加权平均后的温度值
all_beta_exp_weight_avg = []
for i in beta:
    # 存储指数加权平均后的温度值
    exp_weight_avg = []
    # 遍历每天的温度数据，idx从0开始计数
    for idx, temp in enumerate(temperture):
        # 第一天的指数加权平均值就是当天温度本身
        if idx == 0:
            exp_weight_avg.append(temp)
            continue
        # 从第二天开始，按照指数加权平均公式计算：
        # V_t = β * V_{t-1} + (1-β) * θ_t
        # 其中 V_t 是当前时刻的加权平均值，V_{t-1} 是上一时刻的加权
        new_temp = exp_weight_avg[idx - 1] * i + (1 - i) * temp
        exp_weight_avg.append(new_temp)
    all_beta_exp_weight_avg.append(exp_weight_avg)
    
axes[0].plot(days,all_beta_exp_weight_avg[0])
axes[0].set_title(f'权重:{beta[0]}')
axes[0].set_xlabel('天')
axes[0].set_ylabel('温度')
axes[0].grid()
axes[1].plot(days,all_beta_exp_weight_avg[1])
axes[1].set_title(f'权重:{beta[1]}')
axes[1].set_xlabel('天')
axes[1].set_ylabel('温度')
axes[1].grid()
axes[2].plot(days,all_beta_exp_weight_avg[2])
axes[2].set_title(f'权重:{beta[2]}')
axes[2].set_xlabel('天')
axes[2].set_ylabel('温度')
axes[2].grid()
axes[3].plot(days,all_beta_exp_weight_avg[3])
axes[3].set_title(f'权重:{beta[3]}')
axes[3].set_xlabel('天')
axes[3].set_ylabel('温度')
axes[3].grid()

plt.show()

    

