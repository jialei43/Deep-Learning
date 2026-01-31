"""
演示：
    张量的基本运算 + - * / -（取负）
    涉及到的API: add、sub、mul、div、neg
                add_、sub_、mul_、div_、neg_（带线换线的版本会修改原数据）
注意：
    张量 和标量的运算：就是张量的每个元素和这个标量进行运算

"""

import torch

# 1.定义函数，演示张量的基本运算
def demo01():
    t1 = torch.tensor([1,2,3],device=torch.device('mps'))
    t2 = torch.tensor([4,5,6],device=torch.device('mps'))
    print(f't1:{t1},shape:{t1.shape},dtype:{t1.dtype},device:{t1.device}')
    print(f't2:{t2},shape:{t2.shape},dtype:{t2.dtype},device:{t2.device}')
    print('-'*34)
    t3 = t1.add(t2)
    print(f't3:{t3},shape:{t3.shape},dtype:{t3.dtype},device:{t3.device}')
    print('-'*34)
    # 设置随机因子
    torch.manual_seed(22)
    t4 = torch.rand(size=(2,3),device=torch.device('mps'))
    print(f't4:{t4},shape:{t4.shape},dtype:{t4.dtype},device:{t4.device}')
    t5 = t4 + 1
    print(f't5:{t5},shape:{t5.shape},dtype:{t5.dtype},device:{t5.device}')
    print('-'*34)
    t6 = t4.add_(1)
    print(f't6:{t6},shape:{t6.shape},dtype:{t6.dtype},device:{t6.device}')
    print(f't4:{t4},shape:{t4.shape},dtype:{t4.dtype},device:{t4.device}')

if __name__ == '__main__':
    demo01()