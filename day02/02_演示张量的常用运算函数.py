"""
演示：
    张量的常用函数

涉及到API:
    sum(),mean(),max(),min()  -> 有dim参数，可以指定维度/轴进行操作，会改变张量的形状
    pow()/**,sqrt(),exp(),log(),log2(),log10() ->无dim参数，对张量中的所有元素进行运算，不会改变张量的形状
要掌握的：
    sum(),mean(),max(),min()
    **

"""
import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 1.定义函数，演示 常用的运算函数
def demo01():
    # 1.创建一个3D张量
    t1 = torch.randint(low=0,high=10,size=(3,4,5),device=device)
    print(f't1:{t1},shape:{t1.shape},dtype:{t1.dtype},device:{t1.device}')
    print('-'*34)
    # 2.求和
    t2 = t1.sum()
    print(f't2:{t2},shape:{t2.shape},dtype:{t2.dtype},device:{t2.device}')
    print('-'*34)
    t3 = t1.sum(dim=0)
    print(f't3:{t3},shape:{t3.shape},dtype:{t3.dtype},device:{t3.device},dim:{t3.ndim}')
    print('-'*34)
    t4 = t1.float().mean(dim=0)
    print(f't4:{t4},shape:{t4.shape},dtype:{t4.dtype},device:{t4.device},dim:{t4.ndim}')
    print('-'*34)
    t5 = t1.max()
    print(f't5:{t5}')
    print('-'*34)
    t6 = t1.max(dim=0)
    print(f't6:{t6}')

if __name__ == '__main__':
    demo01()