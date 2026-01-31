"""
演示：
    张量元素类型的转换
    涉及到的API:
        data.to(torch.float32)
        data.type(torch.DoubleTensor)
        data.half/double/float/short/int/long()
    要掌握：
        data.to(torch.float32)
        data.type(torch.DoubleTensor)
"""
import torch
# 演示张量的元素类型转换

# 检查 MPS 是否可用
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

def demo01():
    # 创建张量时，指定元素类型
    t1 = torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float32)
    print(f't1:{t1},shape:{t1.shape},dtype:{t1.dtype},device:{t1.device},type:{type(t1)}')
    # 2.使用torch.to 实现张量元素类型转换
    t2 = t1.to(dtype=torch.int64)
    t2 = t2.to(device)
    print(f't2:{t2},shape:{t2.shape},dtype:{t2.dtype},device:{t2.device},type:{type(t2)}')

    # 3.使用torch.type
    t3 = t1.type(dtype=torch.DoubleTensor)
    print(f't3:{t3},shape:{t3.shape},dtype:{t3.dtype},device:{t3.device},type:{type(t3)}')

    # 4.使用张量元素类型的方法
    t4 = t1.half()
    print(f't4:{t4},shape:{t4.shape},dtype:{t4.dtype},device:{t4.device},type:{type(t4)}')

if __name__ == '__main__':
    demo01()