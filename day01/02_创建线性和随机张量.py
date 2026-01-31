"""
演示：
    创建 线性 和 随机张量
    涉及到API:
            torch.arange()和torch.linspace() 创建线性张量
            torch.manual_seed() 设置随机种子
            torch.rand/randn() 创建随机浮点类型张量
            torch.randint(low=0, high=,size=()) 创建随机整数类型张量
"""

import torch

# 1 定义函数，使用torch.arange()和torch.linspace()创建线性张量
def demo01():
    # 1.创建0-9线性张量
    t1 = torch.arange(start=1, end=10, step=1,device=torch.device('mps'))
    print(f't1:{t1},shape:{t1.shape},type:{type(t1)},dtype:{t1.dtype},device:{t1.device}')
    print('-'*34)
    # 2.创建0-1线性张量
    t2 = torch.linspace(start=1, end=10, steps=1,device=torch.device('mps'))
    print(f't2:{t2},shape:{t2.shape},type:{type(t2)},dtype:{t2.dtype},device:{t1.device}')

# 2.定义函数 创建随机张量
def demo02():
    # 设置随机种子
    torch.manual_seed(42)
    # 1.创建随机张量
    t1 = torch.rand(size=(2,3),device=torch.device('mps'))
    print(f't1:{t1},shape:{t1.shape},type:{type(t1)},dtype:{t1.dtype},device:{t1.device}')
    print('-'*34)
    # 2.创建随机张量
    t2 = torch.randn(size=(2,3),device=torch.device('mps'))
    print(f't2:{t2},shape:{t2.shape},type:{type(t2)},dtype:{t2.dtype},device:{t2.device}')
    print('-'*34)
    # 3.创建随机整数张量
    t3 = torch.randint(low=0, high=10, size=(2,3),device=torch.device('mps'))
    print(f't3:{t3},shape:{t3.shape},type:{type(t3)},dtype:{t3.dtype},device:{t3.device}')
    print('-'*34)
    # 4.打印随机种子
    print(f'随机种子:{torch.initial_seed()}')


if __name__ == '__main__':
    # demo01()
    demo02()