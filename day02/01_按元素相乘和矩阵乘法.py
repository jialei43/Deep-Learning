"""
演示
    按元素相乘 和 矩阵相乘
按元素相乘：
    按元素相乘（Hadamard）的是相同形状的张量对应位置的元素相乘，使用mul 或 运算符*实现
    要求：俩个张量的形状一样：A:(m,n)  B(m,n)
    API:
        t1*t2
        t1.mul(t2)
        torch.mul(t1,t2)
需要掌握：
    t1*t2
"""

import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 1.定义函数，演示 张量的按元素相乘
def demo01():
    # 设置随机种子
    torch.manual_seed(42)
    # 创建两个张量
    t1 = torch.randint(low=0,high=10,size=(2,3),device=device)
    print(f't1:{t1},shape:{t1.shape},type:{type(t1)},dtype:{t1.dtype},device:{t1.device}')
    t2 = torch.randint(low=0,high=10,size=(2,3),device=device)
    print(f't2:{t2},shape:{t2.shape},type:{type(t2)},dtype:{t2.dtype},device:{t2.device}')
    print('-'*34)
    # 3.演示张量的按元素相乘
    # t3 = t1*t2
    # t3 = torch.mul(t1, t2)
    t3 = t1.mul(t2)
    print(f't3:{t3},shape:{t3.shape},type:{type(t3)},dtype:{t3.dtype},device:{t3.device}')


# 1.定义函数，演示 张量的矩阵乘法
def demo02():
    # 设置随机种子
    torch.manual_seed(42)
    # 创建两个张量
    t1 = torch.randint(low=0,high=10,size=(3,4),device=device).float()
    print(f't1:{t1},shape:{t1.shape},type:{type(t1)},dtype:{t1.dtype},device:{t1.device}')
    t2 = torch.randint(low=0,high=10,size=(4,3),device=device).float()
    print(f't2:{t2},shape:{t2.shape},type:{type(t2)},dtype:{t2.dtype},device:{t2.device}')
    print('-'*34)
    # 3.演示张量的按元素相乘
    # t3 = t1@t2
    # t3 = torch.matmul(t1, t2)
    # t3 = torch.mm(t1, t2)
    t3 = t1.matmul(t2)
    print(f't3:{t3},shape:{t3.shape},type:{type(t3)},dtype:{t3.dtype},device:{t3.device}')

if __name__ == '__main__':
    # demo01()
    demo02()