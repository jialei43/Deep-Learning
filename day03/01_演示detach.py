'''
演示
    detach()函数功能，解决开启梯度计算requires_grad = True的张量，无法转换为numpy
    对象的问题
介绍：
    一个张量Tensor，开启梯度计算requires_grad = True,就无法Tensor.numpy(),需要Tensor.detach().numpy()
    来转换为numpy数组
需要掌握
    n1 = torch.Tensor([1,2,3]).detach().numpy()
'''

# 导包
import torch

w = torch.tensor([1,2,3],requires_grad=True,dtype=torch.float32 )
print(f'w:{w},shape:{w.shape},requires_grad:{w.requires_grad},device:{w.device}')

# 转换为numpy
# n1 = w.numpy()
# print(f'n1:{n1},shape:{n1.shape},dtype:{n1.dtype}')

# 使用detach().numpy()来转换为numpy数组
t1 = w.detach()
print(f't1:{t1},shape:{t1.shape},requires_grad:{t1.requires_grad},device:{t1.device}')
n2 = w.detach().numpy()
print(f'n2:{n2},shape:{n2.shape},dtype:{n2.dtype}')
print('-'*34)
# 查看是否共享内存/浅拷贝
n2[0] = 28.0
print(f'w:{w},shape:{w.shape}')
print(f'n2:{n2},shape:{n2.shape}')
