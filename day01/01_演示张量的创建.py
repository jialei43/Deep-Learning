"""
演示：
    张量的创建基本方式

张量
    pytorch中存储数据的一种容器，元素必须是数值，元素类型必须相同
涉及到的API:
    要掌握的：
        torch.tensor

"""
# 导包
import torch

# 定义函数，演示torch.tensor创建张量

def demo01():
    # 1.创建0维度/0D张量，也就是标量
    tensor = torch.tensor(1)
    print(f'0D-t1:{tensor},shape:{tensor.shape}')
    # 2.创建1维度/1D张量，也就是向量
    tensor2 = torch.tensor([1,2,3])
    print(f'1D-t1:{tensor2},shape:{tensor2.shape}')
    # 3.创建2维度/2D张量，也就是矩阵
    tensor3 = torch.tensor([[1,2,3],[4,5,6]])
    print(f'2D-t1:{tensor3},shape:{tensor3.shape}')
    # 4.创建3维度/3D张量，也就是张量
    tensor4 = torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
    print(f'3D-t1:{tensor4},shape:{tensor4.shape}，type:{type(tensor4)},dtype:{tensor4.dtype}')
    print('---------------------------------------------------------------------------------------------------------')
    # 5.创建指定形状的张量（3，4）
    tensor5 = torch.Tensor(3,4)
    print(f'3,4-t1:{tensor5},shape:{tensor5.shape}')

def demo02():
    # 1.创建0维度/0D张量，也就是标量
    tensor = torch.tensor(1)
    print(f'0D-t1:{tensor},shape:{tensor.shape}')
    # 2.创建1维度/1D张量，也就是向量
    tensor2 = torch.tensor([1,2,3])
    print(f'1D-t1:{tensor2},shape:{tensor2.shape}')
    # 3.创建2维度/2D张量，也就是矩阵
    tensor3 = torch.tensor([[1,2,3],[4,5,6]])
    print(f'2D-t1:{tensor3},shape:{tensor3.shape}')
    # 4.创建3维度/3D张量，也就是张量
    tensor4 = torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
    print(f'3D-t1:{tensor4},shape:{tensor4.shape}，type:{type(tensor4)},dtype:{tensor4.dtype}')
    print('---------------------------------------------------------------------------------------------------------')
    # 5.创建指定形状的张量（3，4）
    tensor5 = torch.Tensor(3,4)
    print(f'3,4-t1:{tensor5},shape:{tensor5.shape}')

def demo03():
    # 1.创建0维度/0D张量，也就是标量
    tensor = torch.IntTensor(1)
    print(f'0D-t1:{tensor},shape:{tensor.shape}')
    # 2.创建1维度/1D张量，也就是向量
    tensor2 = torch.IntTensor([1,2,3])
    print(f'1D-t1:{tensor2},shape:{tensor2.shape}')
    # 3.创建2维度/2D张量，也就是矩阵
    tensor3 = torch.FloatTensor([[1,2,3],[4,5,6]])
    print(f'2D-t1:{tensor3},shape:{tensor3.shape}')
    # 4.创建3维度/3D张量，也就是张量
    tensor4 = torch.DoubleTensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
    print(f'3D-t1:{tensor4},shape:{tensor4.shape}，type:{type(tensor4)},dtype:{tensor4.dtype}')
    print('---------------------------------------------------------------------------------------------------------')
    # 5.创建指定形状的张量（3，4）
    tensor5 = torch.Tensor(3,4)
    print(f'3,4-t1:{tensor5},shape:{tensor5.shape}')

# 4.定义函数，演示张量的属性
# type:对象类型，shape:形状，dtype:数据类型，device:设备，requires_grad:是否开启梯度计算
def demo04():
    # 1.创建一个张量，(2,3)
    t1 = torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float32,device="mps",requires_grad=True)
    # t1 = torch.tensor([[1,2,3],[4,5,6]])
    print(f"t1:{t1}, type: {type(t1)}, shape: {t1.shape}, dtype: {t1.dtype}, device: {t1.device}, requires_grad: {t1.requires_grad}")
    # dtype: int32
    # device:cpu
    # requires_grad: False


if __name__ == '__main__':
    # demo01()
    # demo02()
    # demo03()
    demo04()
