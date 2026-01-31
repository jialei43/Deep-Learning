"""
演示：
    张量和numpy 之间的相互转换，以及只有一个数值的张量转换为python数值
    涉及到API:
    张量转numpy
        张量对象.numpy() 共享内存，浅拷贝
        张量对象.numpy().copy() 不共享内存，深拷贝
    numpy转为张量
        torch.form_numpy(numpy数组对象) 共享内存，浅拷贝
        torch.tensor(numpy数组对象) 不共享内存，深拷贝
    一个数值的张量转换 python数值：
        张量对象.item()

"""
import torch

# 1. 定义函数，演示 张量和numpy数组之间转换
def demo01():
    # 0.设置随机种子
    torch.manual_seed(42)
    # 1.创建一个张量
    t1 = torch.rand(size=(2,3))
    print(f't1:{t1},shape:{t1.shape},dtype:{t1.dtype},device:{t1.device}')
    print('-'*34)
    # 2.将张量转为numpy数组
    np_array = tensor_to_numpy(t1,False)
    print(f'np_array:{np_array},shape:{np_array.shape},dtype:{np_array.dtype}')
    # 验证是否是浅拷贝
    np_array[0,0]=100
    print(f'np_array:{np_array},shape:{np_array.shape},dtype:{np_array.dtype}')
    print(f't1:{t1},shape:{t1.shape},dtype:{t1.dtype},device:{t1.device}')
    print('-'*34)
    # 3.使用nunpy().copy()进行深拷贝
    np_array2 = tensor_to_numpy(t1,True)
    print(f'np_array2:{np_array2},shape:{np_array2.shape},dtype:{np_array2.dtype}')
    # 验证是否是深拷贝
    np_array2[0, 0] = 200
    print(f'np_array2:{np_array2},shape:{np_array2.shape},dtype:{np_array2.dtype}')
    print(f't1:{t1},shape:{t1.shape},dtype:{t1.dtype},device:{t1.device}')
    print('-'*34)
    # 4.将numpy数组转为张量
    t2 = torch.from_numpy(np_array2)
    print(f't2:{t2},shape:{t2.shape},dtype:{t2.dtype},device:{t2.device}')
    print('-'*34)
    # 5.将张量转为python数值
    t3 = torch.tensor(28)
    python_value = t3.item()
    print(f'python_value:{python_value},type:{type(python_value)}')




def tensor_to_numpy(t1,copy=True):
    np_array = None
    if t1.is_mps or t1.device.type == 'mps':
        if copy:
            np_array = t1.cpu().numpy().copy()
        else:
            np_array = t1.cpu().numpy()
    else:
        if copy:
            np_array = t1.detach().cpu().numpy().copy()
        else:
            np_array = t1.detach().cpu().numpy()

    return np_array

    # 1.创建一个张量
    t1 = torch.tensor(10,device=torch.device('mps'))
    print(f't1:{t1},shape:{t1.shape},dtype:{t1.dtype},device:{t1.device}')
    print('-'*34)
    # 2.将张量转为python数值
    python_value = t1.item()
    print(f'python_value:{python_value},type:{type(python_value)}')
if __name__ == '__main__':
    demo01()