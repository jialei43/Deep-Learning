"""
演示
    张量的拼接
涉及到的API:
    torch.cat()
        在指定的维度/轴拼接张量，不改变张量的维度2D/3D，会改变拼接的维度/轴的长度，除了要拼接的维度，其他维度形状必须相同
        (m,n)+(k,n)->(m+k,n)\
    torch.stack()
        在一个新的维度/轴堆叠张量，增加一个新的维度，比如2D->3D,要求输入的张量的形状必须相同
        (m,n)+(m,n)->(2,m,n)

需要掌握的：
    torch.cat
    torch.stack
"""
import torch

# 1.定义函数，演示张量的拼接
def demo01():
    torch.manual_seed(42)
    # 1.创建两个张量
    t1 = torch.randint(low=0,high=10,size=(3,4),device=torch.device('mps'))
    print(f't1:{t1},shape:{t1.shape}')
    t2 = torch.randint(low=0,high=10,size=(5,4),device=torch.device('mps'))
    print(f't2:{t2},shape:{t2.shape}')
    print('='*68)
    t3 = torch.randint(low=0,high=10,size=(2,4),device=torch.device('mps'))
    print(f't3:{t3},shape:{t3.shape},')
    # 使用torch.cat()拼接
    t4 = torch.cat([t1,t2,t3],dim=0)
    print(f't4:{t4},shape:{t4.shape}')
    # 当dim =1 时，拼接的其他轴/维度长度必须一致
    print('='*68)
    t5 = torch.randint(low=0, high=10, size=(5, 3), device=torch.device('mps'))
    print(f't5:{t5},shape:{t5.shape}')
    t6 = torch.randint(low=0, high=10, size=(5, 4), device=torch.device('mps'))
    print(f't6:{t6},shape:{t6.shape}')
    t7 = torch.cat([t5, t6], dim=1)
    print(f't7:{t7},shape:{t7.shape}')
    print('='*68)

    # 使用torch.stack()拼接 沿着那个维护进行拼接，那个维度的值就是拼接的维度/轴的值，其他维度的值不变
    t1 = torch.randint(low=0, high=10, size=(3, 4), device=torch.device('mps'))
    t2 = torch.randint(low=0, high=10, size=(3, 4), device=torch.device('mps'))
    t8 = torch.stack([t1,t2],dim=0)
    print(f't8:{t8},shape:{t8.shape}')
    print('='*68)
    t9 = torch.stack([t1,t2],dim=1)
    print(f't9:{t9},shape:{t9.shape}')
    print('='*68)
    t10 = torch.stack([t1,t2],dim=2)
    print(f't10:{t10},shape:{t10.shape}')
    print('='*68)



if __name__ == '__main__':
    demo01()