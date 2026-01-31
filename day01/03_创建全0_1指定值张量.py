"""
演示：
    创建 全0，全1，指定值 张量
    涉及到的API:
        torch.zeros 和 torch.zeros_like 创建0张量
        torch.ones 和 torch.ones_like 创建0张量
        torch.full 和 torch.full_like 创建0张量

用处：
    网络参数初始化，偏置参数初始化
    y=kx+b (b初始化为0)
"""
import torch

# 定义函数，创建全0的张量
def demo01():
    t1 = torch.zeros(size=(2,3),device=torch.device('mps'))
    print(f't1:{t1},shape:{t1.shape},type:{type(t1)},dtype:{t1.dtype},device:{t1.device}')
    print('-'*34)
    # 2.创建一个随机张量
    t2 = torch.randn(size=(3,4),device=torch.device('mps'))
    print(f't2:{t2},shape:{t2.shape},type:{type(t2)},dtype:{t2.dtype},device:{t2.device}')
    print('-'*34)
    # 3.创建torch.zeros_like 这样的全0张量
    t3 = torch.zeros_like(t2)
    print(f't3:{t3},shape:{t3.shape},type:{type(t3)},dtype:{t3.dtype},device:{t3.device}')

def demo02():
    t1 = torch.ones(size=(2,3),device=torch.device('mps'))
    print(f't1:{t1},shape:{t1.shape},type:{type(t1)},dtype:{t1.dtype},device:{t1.device}')
    print('-'*34)
    # 2.创建一个随机张量
    t2 = torch.randn(size=(3,4),device=torch.device('mps'))
    print(f't2:{t2},shape:{t2.shape},type:{type(t2)},dtype:{t2.dtype},device:{t2.device}')
    print('-'*34)
    # 3.创建torch.ones_like 这样的全0张量
    t3 = torch.ones_like(t2)
    print(f't3:{t3},shape:{t3.shape},type:{type(t3)},dtype:{t3.dtype},device:{t3.device}')


def demo03():
    t1 = torch.full(size=(2,3),fill_value=10,device=torch.device('mps'))
    print(f't1:{t1},shape:{t1.shape},type:{type(t1)},dtype:{t1.dtype},device:{t1.device}')
    print('-'*34)
    # 2.创建一个随机张量
    t2 = torch.randn(size=(3,4),device=torch.device('mps'))
    print(f't2:{t2},shape:{t2.shape},type:{type(t2)},dtype:{t2.dtype},device:{t2.device}')
    print('-'*34)
    # 3.创建torch.ones_like 这样的全0张量
    t3 = torch.full_like(t2,10,device=torch.device('mps'))
    print(f't3:{t3},shape:{t3.shape},type:{type(t3)},dtype:{t3.dtype},device:{t3.device}')



if __name__ == '__main__':
    # demo01()
    # demo02()
    demo03()