"""
案例：
    演示 参数初始化 的7种方法
参数初始化的作用：
    1. 防止 梯度消失 或 梯度爆炸
    2. 提高收敛速度
    3. 打破对称性
参数初始化的方法：
    1. 均匀分布初始化
    2. 正态分布初始化
    3. 全0初始化
    4. 全1初始化
    5. 固定值初始化
    6. kaiming 初始化，也叫做 HE 初始化
        正态分布的he初始化
        均匀分布的he初始化
    7. xavier 初始化，也叫做 Glorot初始化
        正态化的Xavier初始化
        均匀分布的Xavier初始化
需要掌握：
    无
总结：
    权重w:
        使用relu/prelu的网络层，kaiming初始化
        非relu的网络层，Xavier初始化
    偏置b:
        全0初始化：1.简单；2.计算快

"""
# 导包
import torch.nn as nn

# 1.定义函数，演示 均匀分布初始化
def demo01():
    print("均匀分布初始化")
    # 1.定义一个线性层，输入维度3，输出维度5
    linear = nn.Linear(3,5)
    # 2.对权重w进行初始化
    nn.init.uniform_(linear.weight)
    # 3.对偏置b进行初始化
    nn.init.uniform_(linear.bias)
    # 4.打印权重和偏置
    print(f"w:{linear.weight}, shape: {linear.weight.shape}")
    print(f"b:{linear.bias}, shape: {linear.bias.shape}")
    ...

# 2.定义函数，演示 正态分布初始化
def demo02():
    print("正态分布初始化")
    # 1.定义一个线性层，输入维度3，输出维度5
    linear = nn.Linear(3,5)
    # 2.对权重w进行初始化
    nn.init.normal_(linear.weight)
    # 3.对偏置b进行初始化
    nn.init.normal_(linear.bias)
    # 4.打印权重和偏置
    print(f"w:{linear.weight}, shape: {linear.weight.shape}")
    print(f"b:{linear.bias}, shape: {linear.bias.shape}")
    ...

# 3.定义函数，演示 全0分布初始化
def demo03():
    print("全0分布初始化")
    # 1.定义一个线性层，输入维度3，输出维度5
    linear = nn.Linear(3,5)
    # 2.对权重w进行初始化
    # nn.init.normal_(linear.weight)
    # 3.对偏置b进行初始化
    nn.init.zeros_(linear.bias)
    # 4.打印权重和偏置
    # print(f"w:{linear.weight}, shape: {linear.weight.shape}")
    print(f"b:{linear.bias}, shape: {linear.bias.shape}")
    ...

# 5.定义函数，演示 固定值初始化
def demo05():
    print("固定值初始化")
    # 1.定义一个线性层，输入维度3，输出维度5
    linear = nn.Linear(3,5)
    # 2.对权重w进行初始化
    # nn.init.normal_(linear.weight)
    # 3.对偏置b进行初始化
    nn.init.constant_(linear.bias,val=0.5)
    # 4.打印权重和偏置
    # print(f"w:{linear.weight}, shape: {linear.weight.shape}")
    print(f"b:{linear.bias}, shape: {linear.bias.shape}")
    ...

# 6.定义函数，演示 kaiming初始化
def demo06():
    print("kaiming初始化")
    # 1.定义一个线性层，输入维度3，输出维度5
    linear = nn.Linear(3,5)
    # 2.对权重w进行初始化,kaiming正态初始化
    nn.init.kaiming_normal_(linear.weight)
    # 3.打印权重和偏置
    print(f"w:{linear.weight}, shape: {linear.weight.shape}")
    # print(f"b:{linear.bias}, shape: {linear.bias.shape}")
    # 4.对权重w进行初始化,kaiming均匀初始化
    nn.init.kaiming_uniform_(linear.weight)
    print(f"w:{linear.weight}, shape: {linear.weight.shape}")
    ...

# 7.定义函数，演示 xavier初始化
def demo07():
    print("xavier初始化")
    # 1.定义一个线性层，输入维度3，输出维度5
    linear = nn.Linear(3,5)
    # 2.对权重w进行初始化,xavier正态初始化
    nn.init.xavier_normal_(linear.weight)
    # 3.打印权重和偏置
    print(f"w:{linear.weight}, shape: {linear.weight.shape}")
    # print(f"b:{linear.bias}, shape: {linear.bias.shape}")
    # 4.对权重w进行初始化,xavier均匀初始化
    nn.init.xavier_uniform_(linear.weight)
    print(f"w:{linear.weight}, shape: {linear.weight.shape}")
    ...

# 测试
if __name__ == '__main__':
    demo01()
    # demo02()
    # demo03()
    # demo05()
    # demo06()
    # demo07()